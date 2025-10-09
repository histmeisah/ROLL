# ROLL Agentic Pipeline 详细指南

## 目录
1. [环境创建与测试机制](#1-环境创建与测试机制)
2. [Agentic数据流与Async/Sync实现](#2-agentic数据流与asyncsync实现)
3. [VLM与LLM在Agentic中的差异](#3-vlm与llm在agentic中的差异)
4. [Reward与Trajectory记录实现](#4-reward与trajectory记录实现)
5. [环境类型区分机制](#5-环境类型区分机制)
6. [环境测试方法](#6-环境测试方法)

---

## 1. 环境创建与测试机制

### 1.1 环境创建架构

ROLL的环境系统采用了**分层抽象设计**：

```python
# 基础环境抽象
class BaseEnv(ABC):
    @abstractmethod
    def reset(self, seed=None, **kwargs) -> Tuple[Any, dict]:
        pass
    
    @abstractmethod
    def step(self, action: str) -> Tuple[Any, float, bool, bool, Dict]:
        pass
```

#### 环境注册机制
```python
# roll/agentic/env/__init__.py
REGISTERED_ENVS = {
    "sokoban": SokobanEnv,
    "frozen_lake": FrozenLakeEnv,
    "webshop": WebShopEnv,
    # 更多环境...
}
```

#### 环境配置系统
```python
# 环境配置示例 (Sokoban)
@dataclass
class SokobanEnvConfig(BaseEnvConfig):
    dim_room: Tuple[int, int] = (6, 6)
    max_steps: int = 100
    num_boxes: int = 3
    env_instruction: str = "You are solving the Sokoban puzzle..."
    action_pattern: str = r"<answer>(.*?)</answer>"
```

### 1.2 测试Agent实现

#### Random Agent
```python
@register_llm_proxy("random")
class RandomProxy(BaseLLMProxy):
    def generate(self, messages, lm_input, generation_config):
        response_text = f"{random.choice(self.available_actions)}"
        responses = self.tokenizer([response_text], return_tensors="pt")
        lm_input.batch["responses"] = responses["input_ids"]
        return lm_input
```

#### Test Agent (Policy Proxy)
```python
@register_llm_proxy("policy")
class PolicyProxy(BaseLLMProxy):
    def generate(self, messages, lm_input, generation_config):
        lm_input.meta_info["generation_config"] = generation_config
        lm_output = ray.get(self.generate_scheduler.generate_one_request.remote(data=lm_input))
        return lm_output
```

### 1.3 环境测试流程

```python
# 测试环境示例
def test_frozen_lake():
    config = FrozenLakeEnvConfig(size=4, p=0.8, is_slippery=False, map_seed=42)
    env = FrozenLakeEnv(config)
    
    obs, _ = env.reset(seed=42)
    while True:
        action = input("Enter action: ")
        obs, reward, done, info = env.step(action)
        print(f"Observation: {obs}, Reward: {reward}, Done: {done}")
        if done:
            break
```

---

## 2. Agentic数据流与Async/Sync实现

### 2.1 整体数据流架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Environment   │───▶│  LLM Proxy      │───▶│  Rollout Cache  │
│   Workers       │    │  (Policy/Random)│    │  (Trajectory)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  GroupQueue     │    │  RequestScheduler│    │  DataProto      │
│  (Batch Mgmt)   │    │  (Async Gen)    │    │  (Serialization)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2.2 Async/Sync实现机制

#### Async训练模式
```python
# RolloutScheduler中的Async实现
async def _start_env_manager(self, global_step):
    if self.config.async_generation_ratio > 0 and self.mode == "train":
        # Async training: 只调用一次run_rollout_loop
        self.rollout_refs = self.es_manager.run_rollout_loop(None, seed, blocking=False)
    else:
        # Sync training: 每次get_batch都调用run_rollout_loop
        self.rollout_refs = self.es_manager.run_rollout_loop(global_step, seed, blocking=False)
```

#### Sync训练模式
```python
# 同步训练流程
async def get_batch(self, data: DataProto, batch_size):
    # 1. 启动环境管理器
    await self._start_env_manager(global_step)
    
    # 2. 收集轨迹数据
    batch = await self.env_output_queue.get_batch(batch_size)
    
    # 3. 停止环境管理器
    await self._stop_env_manager()
    
    return batch
```

### 2.3 Async配置参数

```yaml
# 配置文件中的async设置
async_generation_ratio: 1  # 0表示同步，>0表示异步
```

#### Async实现机制
```python
# roll/configs/base_config.py
async_generation_ratio: float = field(
    default=0,
    metadata={
        "help": "The ratio of ahead generation requests in pipeline, "
        "0 means synchronous pipeline. currently only integer is supported."
    },
)
```

#### Async vs Sync逻辑
```python
# roll/agentic/rollout/rollout_scheduler.py
if config.async_generation_ratio > 0 and self.mode == "train":
    # Async training: 使用GroupQueue实现速率限制
    # 最多有 `async_generation_ratio - 1` 个进行中的步骤
    max_group_num = env_manager_config.max_traj_per_env * (config.async_generation_ratio - 1)
    if max_group_num == 0:
        queue_cls = PipeGroupQueue
    else:
        queue_cls = BoundedGroupQueue
else:
    # Sync training: 不需要速率限制
    max_group_num = env_manager_config.max_traj_per_env
    queue_cls = PipeGroupQueue
```

### 2.4 数据流关键组件

#### GroupQueue管理
```python
class BoundedGroupQueue(GroupQueue):
    async def put(self, episode_id, start_step, rollout):
        # 按episode_id分组存储
        if episode_id not in self.groups:
            self.groups[episode_id] = []
        self.groups[episode_id].append(rollout)
        
        # 当组满时释放信号
        if len(self.groups[episode_id]) == self.group_size:
            self.completed.release()
    
    async def get(self):
        # 获取完整的组
        await self.completed.acquire()
        # 返回最早完成的组
        target = min(episode_id for episode_id, rollouts in self.groups.items() 
                    if len(rollouts) >= self.group_size)
        return self.groups.pop(target)
```

#### 轨迹缓存机制
```python
@dataclass
class RolloutCache:
    env_id: int
    group_id: int
    tag: str
    history: List[Dict] = field(default_factory=list)  # 每步信息
    frames: List = field(default_factory=list)         # 渲染帧
    truncated: bool = False
    terminated: bool = False
    step: int = 0
```

---

## 3. VLM与LLM在Agentic中的差异

### 3.1 架构差异

#### LLM环境管理器 (TrajEnvManager)
```python
class TrajEnvManager(BaseEnvManager):
    def __init__(self, tokenizer, ...):
        self.tokenizer = tokenizer
        # 纯文本处理
    
    def format_messages(self, history):
        # 纯文本消息格式
        messages = [
            {"role": "system", "content": self.agent_system_template},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        return messages
```

#### VLM环境管理器 (VLTrajEnvManager)
```python
class VLTrajEnvManager(TrajEnvManager):
    def __init__(self, tokenizer, processor, ...):
        self.tokenizer = tokenizer
        self.processor = processor  # 多模态处理器
        self.collator = DataCollatorWithPaddingForMM(...)
    
    def format_messages(self, history):
        # 多模态消息格式
        messages = [
            {"role": "system", "content": self.agent_system_template},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": text_content},
                    {"type": "image", "image": image_data}
                ]
            }
        ]
        return messages
```

### 3.2 数据处理差异

#### LLM数据处理
```python
# TrajEnvManager.make_decision
def make_decision(self, rollout_cache: RolloutCache):
    messages = self.format_messages(rollout_cache.history)
    
    # 纯文本tokenization
    lm_input_texts = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = self.tokenizer(lm_input_texts, return_tensors="pt")
    
    return self.llm_proxy.generate(messages, lm_input, generation_config)
```

#### VLM数据处理
```python
# VLTrajEnvManager.make_decision
def make_decision(self, rollout_cache: RolloutCache):
    messages = self.format_messages(rollout_cache.history)
    
    # 多模态collation
    batch = self.collator(messages)
    lm_input = DataProto.from_single_dict(inputs)
    
    return self.llm_proxy.generate(messages, lm_input, generation_config)
```

### 3.3 环境渲染差异

#### LLM环境渲染
```python
def render(self, mode="text"):
    if mode == "text":
        return "\n".join("".join(self.GRID_LOOKUP.get(cell, "?") for cell in row) 
                        for row in room.tolist())
```

#### VLM环境渲染
```python
def render(self, mode="rgb_array"):
    if mode == "rgb_array":
        return self.get_image(mode="rgb_array", scale=1)  # 返回图像数据
    elif mode == "text":
        return super().render(mode="text")
```

---

## 4. Reward与Trajectory记录实现

### 4.1 Reward计算机制

#### 环境Reward
```python
# 环境step中的reward计算
def step(self, action: str):
    next_state, reward, terminated, truncated, info = self.env.step(action)
    
    # 基础环境reward
    self.rollout_cache.history[-1]['reward'] = reward
    
    # 格式惩罚
    if not info['metrics'].get("action_is_valid", True):
        self.rollout_cache.history[-1]['penalty'] = self.worker_config.format_penalty
    
    return self.rollout_cache
```

#### Reward归一化
```python
# AgenticPipeline中的reward处理
def run(self):
    # 按组归一化reward
    grouping = self.pipeline_config.reward_normalization.grouping
    batch_grouped = batch.group_by(keys=grouping)
    
    for group_name, group_batch in batch_grouped.items():
        score_norm_fn = get_score_normalize_fn(rn_cfg=self.pipeline_config.reward_normalization)
        scores = group_batch.batch["scores"].clone()
        penalty = group_batch.batch["penalty"]
        
        # 累积reward + penalty
        acc_scores = scores.sum(dim=-1)
        normalized_acc_scores = acc_scores + penalty
```

### 4.2 Trajectory记录实现

#### 轨迹缓存结构
```python
@dataclass
class RolloutCache:
    env_id: int
    group_id: int
    tag: str
    history: List[Dict] = field(default_factory=list)  # 轨迹历史
    frames: List = field(default_factory=list)         # 渲染帧
    truncated: bool = False
    terminated: bool = False
    step: int = 0
```

#### 轨迹记录流程
```python
# TrajEnvManager中的轨迹记录
def run_rollout_loop(self, data: DataProto):
    while self.running:
        # 1. 决策
        lm_output = self.make_decision(rollout_cache)
        
        # 2. 环境步进
        rollout_cache = self.step(lm_output)
        
        # 3. 检查终止条件
        if rollout_cache.terminated or stop_reason == GenerateStopReason.MAX_LENGTH:
            # 4. 格式化轨迹
            rollout = self.formulate_rollouts(rollout_cache)
            
            # 5. 发送到输出队列
            ray.get(self.output_queue.put.remote(
                self.env_config['group_id'], 
                self.episode_id, 
                start_step, 
                rollout
            ))
```

#### 轨迹格式化
```python
def formulate_rollouts(self, rollout_cache: RolloutCache):
    # 1. 处理历史记录
    history = rollout_cache.history[:-1]  # 移除最后的状态
    
    # 2. 计算累积reward
    scores = [i['reward'] for i in self.rollout_cache.history]
    episode_score = sum(scores)
    
    # 3. 计算累积penalty
    penalty = [i['penalty'] for i in self.rollout_cache.history]
    episode_penalty = sum(penalty)
    
    # 4. 格式化消息
    messages = self.format_messages(history)
    
    # 5. Tokenization
    lm_input_texts = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False)
    inputs = self.tokenizer(lm_input_texts, return_tensors="pt")
    
    # 6. 生成response mask
    response_masks = token_ids_to_assistant_mask(messages, token_ids_split, self.tokenizer)
    
    # 7. 构建DataProto
    rollout = DataProto()
    rollout.batch = TensorDict({
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "response_mask": torch.tensor(response_masks, dtype=torch.bool),
        "scores": torch.tensor([episode_score]),
        "penalty": torch.tensor([episode_penalty])
    })
    
    return rollout
```

### 4.3 数据流总结

```
Environment Step ──▶ Reward Calculation ──▶ Trajectory Cache ──▶ Batch Collection
       │                    │                       │                    │
       ▼                    ▼                       ▼                    ▼
   State/Observation   Environment Reward    History Recording    GroupQueue
       │                    │                       │                    │
       ▼                    ▼                       ▼                    ▼
   LLM Decision      Format Penalty         RolloutCache         DataProto
       │                    │                       │                    │
       ▼                    ▼                       ▼                    ▼
   Action Execution   Normalized Reward     Trajectory Format    Training Batch
```

---

## 5. 环境类型区分机制

### 5.1 通过配置文件声明环境类型

ROLL通过配置文件中的 `env_manager_cls` 字段来明确指定每个环境使用哪种环境管理器：

#### LLM环境配置
```yaml
# examples/qwen2.5-0.5B-agentic/agentic_val_sokoban.yaml
custom_envs:
  SimpleSokoban:
    env_type: sokoban
    env_manager_cls: roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager  # LLM环境管理器
    env_config:
      env_instruction: "You are solving the Sokoban puzzle..."
      render_mode: "text"  # 文本渲染模式
```

#### VLM环境配置
```yaml
# examples/qwen2.5-vl-3B-agentic/agentic_val_sokoban.yaml
custom_envs:
  SimpleSokoban:
    env_type: sokoban
    env_manager_cls: roll.pipeline.agentic.env_manager.vl_traj_env_manager.VLTrajEnvManager  # VLM环境管理器
    env_config:
      env_instruction: "You are solving the Sokoban puzzle..."
      render_mode: "rgb_array"  # 图像渲染模式
```

### 5.2 环境管理器选择逻辑

```python
# roll/pipeline/agentic/environment_worker.py
def create_env_manager(env_id, env_config):
    self.logger.info(f"use env_manager_cls: {env_config['env_manager_cls']}")
    env_manager_cls = safe_import_class(env_config["env_manager_cls"])

    assert env_manager_cls is not None

    if env_manager_cls == TrajEnvManager:
        # LLM环境管理器
        return env_id, env_manager_cls(
            worker_config=self.worker_config,
            pipeline_config=pipeline_config,
            env_config=env_config,
            tokenizer=copy.deepcopy(self.tokenizer),  # 只需要tokenizer
            generate_scheduler=generate_scheduler,
            output_queue=output_queue,
            thread_lock=self.thread_lock,
            mode=mode
        )
    elif env_manager_cls == VLTrajEnvManager:
        # VLM环境管理器
        tokenizer = copy.deepcopy(self.tokenizer)
        processor = copy.deepcopy(self.processor)  # 需要processor处理图像
        return env_id, env_manager_cls(
            worker_config=self.worker_config,
            pipeline_config=pipeline_config,
            env_config=env_config,
            tokenizer=tokenizer,
            processor=processor,  # 额外的图像处理器
            generate_scheduler=generate_scheduler,
            output_queue=output_queue,
            thread_lock=self.thread_lock,
            mode=mode
        )
```

### 5.3 默认环境管理器设置

```python
# roll/pipeline/agentic/agentic_config.py
def make_env_configs(self, env_manager_config: EnvManagerConfig):
    # ...
    entry.update({
        "tag": tag,
        "group_id": group_id,
        "env_id": env_id,
        "config": env_config,
        "env_class": env_class,
        "env_manager_cls": cfg_template.get("env_manager_cls", 
            "roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager"),  # 默认LLM管理器
        "group_seed": group_seeds[group_id],
    })
```

### 5.4 环境类型区分的具体差异

#### 渲染模式差异

**LLM环境渲染**:
```python
# 环境配置中指定文本渲染
env_config:
  render_mode: "text"  # 返回文本描述

# 环境实现
def render(self, mode="text"):
    if mode == "text":
        return "\n".join("".join(self.GRID_LOOKUP.get(cell, "?") for cell in row) 
                        for row in room.tolist())
```

**VLM环境渲染**:
```python
# 环境配置中指定图像渲染
env_config:
  render_mode: "rgb_array"  # 返回图像数组

# 环境实现
def render(self, mode="rgb_array"):
    if mode == "rgb_array":
        return self.get_image(mode="rgb_array", scale=1)  # 返回numpy图像数组
```

#### 消息格式差异

**LLM消息格式**:
```python
# TrajEnvManager.format_messages
def format_messages(self, history: List[Dict]):
    messages = [
        {"role": "system", "content": self.agent_system_template},
        {"role": "user", "content": user_content},  # 纯文本内容
        {"role": "assistant", "content": assistant_content}
    ]
    return messages
```

**VLM消息格式**:
```python
# VLTrajEnvManager.format_messages
def format_messages(self, history: List[Dict]):
    messages = [
        {"role": "system", "content": self.agent_system_template},
        {
            "role": "user", 
            "content": [  # 多模态内容列表
                {"type": "text", "text": text_content},
                {"type": "image", "image": base64_image}
            ]
        }
    ]
    return messages
```

#### 数据处理差异

**LLM数据处理**:
```python
# TrajEnvManager.make_decision
def make_decision(self, rollout_cache: RolloutCache):
    messages = self.format_messages(rollout_cache.history)
    
    # 纯文本tokenization
    lm_input_texts = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = self.tokenizer(lm_input_texts, return_tensors="pt")
    
    return self.llm_proxy.generate(messages, lm_input, generation_config)
```

**VLM数据处理**:
```python
# VLTrajEnvManager.make_decision
def make_decision(self, rollout_cache: RolloutCache):
    messages = self.format_messages(rollout_cache.history)
    
    # 提取图像数据
    images = []
    for message in messages:
        if message["role"] == "user":
            content = message["content"]
            images.extend([content[i].pop("image_PIL") for i in range(len(content)) 
                         if content[i]["type"] == "image"])
    
    # 多模态collation
    features = [{
        self.collator.prompt_key: lm_input_texts,
        self.collator.image_key: images,
        self.collator.image_flag_key: True
    }]
    inputs = self.collator(features)
    lm_input = DataProto.from_single_dict(inputs)
    
    return self.llm_proxy.generate(messages, lm_input, generation_config)
```

#### 环境管理器初始化差异

**LLM环境管理器**:
```python
class TrajEnvManager(BaseEnvManager):
    def __init__(self, tokenizer, ...):
        self.tokenizer = tokenizer
        # 只需要tokenizer
```

**VLM环境管理器**:
```python
class VLTrajEnvManager(TrajEnvManager):
    def __init__(self, tokenizer, processor, ...):
        self.tokenizer = tokenizer
        self.processor = processor  # 多模态处理器
        self.collator = DataCollatorWithPaddingForMM(
            tokenizer=self.tokenizer,
            processor=self.processor,
            answer_key=None,
            extra_data_provider=get_extra_data_provider(...)
        )
```

### 5.5 环境创建流程

#### 配置解析阶段
```python
# 1. 从配置文件读取环境类型
cfg_template = self.custom_envs[tag]
env_manager_cls = cfg_template.get("env_manager_cls", 
    "roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager")

# 2. 创建环境配置
env_config = REGISTERED_ENV_CONFIGS[env_class](**cfg_template.env_config)

# 3. 构建环境条目
entry = {
    "env_manager_cls": env_manager_cls,  # 关键：指定环境管理器类型
    "env_class": env_class,
    "config": env_config,
    # ... 其他配置
}
```

#### 环境管理器创建阶段
```python
# EnvironmentWorker.initialize
def create_env_manager(env_id, env_config):
    env_manager_cls = safe_import_class(env_config["env_manager_cls"])
    
    if env_manager_cls == TrajEnvManager:
        # 创建LLM环境管理器
        return TrajEnvManager(tokenizer=tokenizer, ...)
    elif env_manager_cls == VLTrajEnvManager:
        # 创建VLM环境管理器
        return VLTrajEnvManager(tokenizer=tokenizer, processor=processor, ...)
```

---

## 6. 环境测试方法

### 6.1 环境测试代码位置

#### 基础环境测试
```python
# tests/agentic/env/test_frozen_lake.py
def test_frozen_lake():
    config = FrozenLakeEnvConfig(size=4, p=0.8, is_slippery=False, map_seed=42)
    env = FrozenLakeEnv(config)
    
    obs, _ = env.reset(seed=42)
    while True:
        action = input("Enter action: ")
        obs, reward, done, info = env.step(action)
        print(f"Observation: {obs}, Reward: {reward}, Done: {done}")
        if done:
            break
```

#### 环境管理器测试
```python
# tests/agentic/env_manager/test_traj_env_manager.py
def test_debug_traj_env_manager():
    # 1. 配置设置
    pipeline_config = make_pipeline_config(config_path, config_name, AgenticConfig)
    pipeline_config.async_generation_ratio = 2
    
    # 2. 初始化组件
    worker_config = pipeline_config.train_env_manager
    tokenizer = default_tokenizer_provider(model_args=worker_config.model_args)
    output_queue = GroupQueueManager.remote(...)
    
    # 3. 创建环境管理器
    env_manager = TrajEnvManager(
        worker_config=worker_config,
        pipeline_config=pipeline_config,
        env_config=worker_config.env_configs[0][0],
        tokenizer=tokenizer,
        generate_scheduler=generate_scheduler,
        output_queue=output_queue,
        thread_lock=threading.Lock(),
        mode="train"
    )
    
    # 4. 运行测试
    data = DataProto(meta_info={"current_step": 0, "seed": 0})
    env_manager.run_rollout_loop(data=data)
    
    # 5. 获取结果
    batch = ray.get(output_queue.get_batch.remote(batch_size=pipeline_config.rollout_batch_size))
    print(batch)
```

### 6.2 如何测试自己的环境

#### 创建环境测试脚本

```python
# 自定义环境测试示例
def test_custom_env():
    # 1. 环境配置
    config = CustomEnvConfig(
        max_steps=100,
        env_instruction="Your environment instruction...",
        action_pattern=r"<answer>(.*?)</answer>"
    )
    
    # 2. 创建环境
    env = CustomEnv(config)
    
    # 3. 测试环境基本功能
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    
    # 4. 测试动作执行
    test_actions = ["Up", "Down", "Left", "Right"]
    for action in test_actions:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}")
        if terminated:
            break
    
    # 5. 测试渲染
    rendered = env.render(mode="text")
    print(f"Rendered state: {rendered}")
```

#### 集成到ROLL测试框架

```python
# tests/agentic/env/test_custom_env.py
def test_custom_env_integration():
    # 1. 使用ROLL的测试配置
    config = CustomEnvConfig()
    env = CustomEnv(config)
    
    # 2. 测试与LLM代理的集成
    tokenizer = default_tokenizer_provider(model_args=worker_config.model_args)
    
    # 3. 测试环境管理器
    env_manager = TrajEnvManager(
        worker_config=worker_config,
        pipeline_config=pipeline_config,
        env_config=env_config,
        tokenizer=tokenizer,
        generate_scheduler=generate_scheduler,
        output_queue=output_queue,
        thread_lock=threading.Lock(),
        mode="train"
    )
    
    # 4. 运行完整测试
    data = DataProto(meta_info={"current_step": 0, "seed": 0})
    env_manager.run_rollout_loop(data=data)
```

#### 测试配置文件

```yaml
# tests/agentic/env_manager/custom_env_debug.yaml
rollout_batch_size: 32
sequence_length: 8192
pretrain: Qwen/Qwen2.5-0.5B-Instruct

train_env_manager:
  format_penalty: -0.15
  max_env_num_per_worker: 1
  num_env_groups: 1
  group_size: 1
  tags: [CustomEnv]
  num_groups_partition: [1]
  llm_proxy:
    proxy_type: random  # 使用random代理进行测试

custom_envs:
  CustomEnv:
    env_type: custom_env
    max_tokens_per_step: 128
    user_prompt_format: "<answer> [your answer] </answer>"
    added_text: ""
    env_config:
      env_instruction: "Your environment instruction..."
      action_pattern: "^(.*)$"
      max_steps: 10
```

### 6.3 测试运行方法

#### 基础环境测试
```bash
# 运行基础环境测试
python tests/agentic/env/test_frozen_lake.py

# 运行环境管理器测试
python tests/agentic/env_manager/test_traj_env_manager.py
```

#### 自定义环境测试
```bash
# 1. 创建测试环境
conda create -n python310_torch260_em python=3.10

# 2. 安装依赖
pip install torch torchvision torchaudio py-cpuinfo
pip install -r requirements_em_local_debug.txt

# 3. 运行测试
python tests/agentic/env_manager/test_traj_env_manager.py
```

### 6.4 测试验证要点

1. **环境基本功能**: reset, step, render等基本方法
2. **动作解析**: 确保动作能被正确解析和执行
3. **奖励计算**: 验证奖励计算逻辑
4. **终止条件**: 检查环境终止条件
5. **多模态支持**: 如果支持VLM，测试图像处理
6. **并发安全**: 测试多线程环境下的安全性

---

## 总结

ROLL的agentic pipeline实现了一个完整的、可扩展的强化学习系统：

1. **环境系统**: 支持多种环境类型，通过统一接口实现可扩展性
2. **异步架构**: 通过Ray实现分布式计算，支持大规模并行训练
3. **多模态支持**: VLM和LLM通过不同的环境管理器实现差异化处理
4. **轨迹管理**: 完整的轨迹记录和reward计算系统，支持复杂的强化学习算法
5. **配置驱动**: 通过配置文件灵活指定环境类型和管理器
6. **测试框架**: 完整的测试体系，支持环境验证和集成测试

这个架构设计使得ROLL能够高效地处理大规模agentic强化学习任务，同时保持良好的可扩展性和易用性。
