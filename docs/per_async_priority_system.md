# PER异步优先级更新系统

## 1. 概述

本文档描述ROLL框架中基于Prioritized Experience Replay (PER)的异步优先级更新系统。该系统结合了**样本新鲜度（Freshness）**和**训练信号（Advantage/TD-Error）**两个维度来动态调整采样优先级，并通过**异步刷新机制**避免CPU-GPU空闲瓶颈。

### 1.1 核心创新

1. **Reward-Fresh优先级函数**：结合奖励信号和时间衰减，为off-policy LLM RL量身定制
2. **异步Age Decay刷新**：利用GPU训练期间的CPU空闲窗口，通过ThreadPoolExecutor在后台刷新优先级
3. **训练后优先级更新**：基于训练产生的advantage/td_error信号动态更新已采样轨迹的优先级
4. **完整的优先级生命周期**：从存储到衰减到更新，形成闭环

### 1.2 解决的核心问题

在off-policy LLM RL训练中，replay buffer中的样本随时间推移会变得"陈旧"（policy drift）。传统方法面临两个问题：

- **采样效率低**：均匀采样无法区分高价值和低价值样本
- **优先级过时**：存储时计算的优先级无法反映样本在当前策略下的实际价值

本系统通过**三阶段优先级更新**解决这些问题。

---

## 2. 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                       PER 异步优先级更新系统 - 完整架构                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Training Step N                                                                    │
│  ═══════════════                                                                    │
│                                                                                     │
│  ┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐ │
│  │  Phase 1: Rollout   │     │  Phase 2: Ref/Old    │     │  Phase 3: Advantage  │ │
│  │  (VLLM Inference)   │────►│  LogProbs Compute    │────►│  Computation         │ │
│  │  GPU: Busy          │     │  GPU: Busy           │     │  CPU: Busy           │ │
│  │  CPU: Env Interact  │     │  CPU: Wait           │     │  GPU: Wait           │ │
│  └─────────────────────┘     └──────────────────────┘     └──────────────────────┘ │
│                                                                       │             │
│                                                                       ▼             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │  Phase 4: Actor Training + Async Age Decay Refresh                          │   │
│  │  ┌──────────────────────────────┐  ┌──────────────────────────────────────┐ │   │
│  │  │  GPU Thread                  │  │  CPU Background Thread               │ │   │
│  │  │  ┌────────────────────────┐  │  │  ┌──────────────────────────────┐   │ │   │
│  │  │  │  Actor Train           │  │  │  │  ThreadPoolExecutor          │   │ │   │
│  │  │  │  (PPO/GRPO Update)     │  │  │  │  ┌──────────────────────┐   │   │ │   │
│  │  │  │                        │  │  │  │  │ refresh_all_age_decay│   │   │ │   │
│  │  │  │  • Forward pass        │  │  │  │  │                      │   │   │ │   │
│  │  │  │  • Loss computation    │  │  │  │  │ for each trajectory: │   │   │ │   │
│  │  │  │  • Backward pass       │  │  │  │  │   age = step - stored│   │   │ │   │
│  │  │  │  • Optimizer step      │  │  │  │  │   w = exp(-age/decay)│   │   │ │   │
│  │  │  │                        │  │  │  │  │   tree[i] = (p*w)^α  │   │   │ │   │
│  │  │  └────────────────────────┘  │  │  │  └──────────────────────┘   │   │ │   │
│  │  └──────────────────────────────┘  │  └──────────────────────────────────┘ │   │
│  │            ▲ 并行执行 ▲                                                    │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                      │
│                                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │  Phase 5: Store Fresh Data + Wait Async Refresh                             │   │
│  │                                                                              │   │
│  │  ① store_fresh_data_to_replay_buffer(batch, global_step)                    │   │
│  │     • 计算初始优先级: priority = (|reward| + ε) × exp(-0/decay) = |reward|+ε│   │
│  │     • Segment Tree 写入: tree[slot] = max_priority^α                        │   │
│  │                                                                              │   │
│  │  ② _wait_age_decay_refresh()  ← 同步点: 等待后台刷新完成                     │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                      │
│                                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │  Phase 6: Replay Training Loop (train_steps_per_env_step 次)                │   │
│  │                                                                              │   │
│  │  for step_idx in range(train_steps_per_env_step):                           │   │
│  │    ┌─────────────────┐    ┌──────────────────┐    ┌───────────────────────┐ │   │
│  │    │ PER Sampling     │    │ Train on Batch    │    │ Priority Update       │ │   │
│  │    │                  │    │                   │    │                       │ │   │
│  │    │ Segment Tree     │───►│ Compute advantage │───►│ update_priorities()   │ │   │
│  │    │ Stratified Sample│    │ PPO/GRPO train    │    │ new_p = |advantage|+ε │ │   │
│  │    │                  │    │                   │    │ w = exp(-age/decay)   │ │   │
│  │    │ Returns:         │    │ Collects:         │    │ tree[i] = (new_p*w)^α│ │   │
│  │    │ • DataProto      │    │ • advantages      │    │                       │ │   │
│  │    │ • sampled_indices│    │ • td_errors       │    │ max_p = max(max_p, p) │ │   │
│  │    └─────────────────┘    └──────────────────┘    └───────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 优先级函数详解

### 3.1 可用优先级函数一览

代码位置：`roll/agentic/replay_buffer/priority_functions.py`

| 函数名 | 公式 | 适用场景 | 需要训练后更新? |
|--------|------|---------|:---:|
| `uniform` | `1.0` | 均匀采样基准线 | ❌ |
| `reward` | `\|reward\| + ε` | 高奖励样本优先 | ✅ (reward) |
| `recency` | `exp(-α × age)` | 新鲜样本优先 | ❌ (自动衰减) |
| `combined` | 加权 reward + recency | 平衡奖励和新鲜度 | ✅ (reward) |
| **`reward_fresh`** | **`(\|reward\| + ε) × exp(-age/decay)`** | **off-policy LLM RL推荐** | **✅ (reward)** |
| `advantage` | `\|advantage\| + ε` | 基于训练信号 | ✅ (advantage) |
| `td_error` | `\|TD-error\| + ε` | 标准PER | ✅ (td_error) |
| `length` | 序列长度 | 长/短序列偏好 | ❌ |
| `lifo` | 确定性LIFO | Echo模式 | ❌ |
| `fifo` | 确定性FIFO | 保证全覆盖 | ❌ |

### 3.2 Reward-Fresh优先级函数（核心创新）

```python
# priority_functions.py: reward_fresh_priority (L275-336)
def reward_fresh_priority(trajectory, global_step, epsilon=1e-6, age_decay=500.0, **kwargs):
    """
    priority = (|reward| + epsilon) * exp(-age / age_decay)

    两个信号的乘积:
    1. |reward| + epsilon: 高奖励样本更值得学习
    2. exp(-age / age_decay): 新鲜样本与当前策略更接近
    """
    reward = abs(float(trajectory.scores.sum()))
    age = max(0, global_step - trajectory.global_step)
    freshness = math.exp(-age / age_decay)
    return (reward + epsilon) * freshness
```

**Age Decay参数对比**：

| `age_decay` | 半衰期(steps) | Age=400时权重 | 适用场景 |
|:-----------:|:------------:|:------------:|---------|
| 500 | ~346 | 45% | 激进衰减，强调新鲜度 |
| 1000 | ~693 | 67% | 平衡（FrozenLake最佳） |
| 1500 | ~1040 | 77% | 保守衰减，保留老样本 |

### 3.3 优先级更新指标映射

```python
# priority_functions.py: PRIORITY_UPDATE_METRIC (L384-400)
PRIORITY_UPDATE_METRIC = {
    "uniform": None,        # 无需更新
    "lifo": None,           # 确定性采样
    "fifo": None,           # 确定性采样
    "recency": None,        # 自动age衰减
    "length": None,         # 长度不变
    "reward": "reward",     # 用|reward|更新
    "advantage": "advantage", # 用|advantage|更新
    "td_error": "td_error",  # 用|TD-error|更新
    "combined": "reward",   # 更新reward分量
    "reward_fresh": "reward", # 更新reward分量（freshness自动衰减）
}
```

---

## 4. Segment Tree 数据结构

代码位置：`roll/agentic/replay_buffer/segment_tree.py`

### 4.1 为什么使用Segment Tree

PER需要两个核心操作：
1. **按概率采样**：`P(sample_i) = priority_i^α / Σ priority_j^α`
2. **计算最小权重**：用于importance sampling校正

Segment Tree提供 O(log n) 的更新和查询效率：

```
                     SumSegmentTree                        MinSegmentTree
                    (用于采样概率)                          (用于IS权重)

                       [sum]                                  [min]
                      /     \                                /     \
                 [sum_L]  [sum_R]                       [min_L]  [min_R]
                 /    \   /    \                         /    \   /    \
               [p1] [p2] [p3] [p4]                   [p1] [p2] [p3] [p4]
```

### 4.2 分层采样算法

```python
# trajectory_buffer.py: _sample_proportional_slots (L513-558)
def _sample_proportional_slots(self, batch_size):
    p_total = sum(tree[i] for i in valid_slots)
    every_range_len = p_total / batch_size

    indices = []
    for i in range(batch_size):
        # 在第i个分层区间内随机采样
        mass = random() * every_range_len + i * every_range_len
        idx = find_slot_where_cumulative_sum >= mass
        indices.append(idx)

    return indices
```

**效果**：高优先级样本被采样的概率更高，但通过分层保证了采样的多样性。

---

## 5. 异步Age Decay刷新机制

### 5.1 问题：CPU-GPU并行利用

在标准训练流程中，Actor Training阶段GPU忙碌而CPU空闲：

```
时间 ──────────────────────────────────────────────────►

GPU:  [Rollout] [Ref/Old LogProbs] [Actor Train ████] [Replay Train]
CPU:  [Env Mgr] [Advantage Comp  ] [IDLE ⭐⭐⭐⭐] [Sample Buffer]
                                     │
                                     └─ CPU完全空闲！
                                        可以刷新age decay
```

### 5.2 解决方案：ThreadPoolExecutor异步刷新

代码位置：`roll/pipeline/agentic/agentic_pipeline.py`

#### 5.2.1 初始化

```python
# agentic_pipeline.py (L223-238)
from concurrent.futures import ThreadPoolExecutor

self._age_decay_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="age_decay_refresh"
)
self._age_decay_future = None
```

#### 5.2.2 异步提交刷新任务

```python
# agentic_pipeline.py (L240-270)
def _async_refresh_age_decay(self, global_step):
    """在Actor Training期间提交后台刷新任务"""
    # 等待上一次刷新完成（防止累积）
    self._wait_age_decay_refresh()

    # 提交新的刷新任务到后台线程
    self._age_decay_future = self._age_decay_executor.submit(
        self.replay_buffer.refresh_all_age_decay,
        global_step
    )
```

#### 5.2.3 同步等待（采样前）

```python
# agentic_pipeline.py (L272-287)
def _wait_age_decay_refresh(self):
    """采样前等待刷新完成，确保优先级是最新的"""
    if self._age_decay_future is None:
        return
    try:
        result = self._age_decay_future.result(timeout=30.0)
    except Exception as e:
        logger.warning(f"Age decay refresh failed: {e}")
    finally:
        self._age_decay_future = None
```

### 5.3 时序保证

```
Step N:
  ┌─ Actor Train (GPU) ─────────────────────────────┐
  │  _async_refresh_age_decay(N)                     │
  │       │                                          │
  │       ▼ (后台线程)                                │
  │  ┌─ refresh_all_age_decay ──┐                    │
  │  │  遍历buffer所有轨迹       │                    │
  │  │  重新计算 priority × w   │                    │
  │  │  更新Segment Tree        │                    │
  │  └─────────────────────────┘                     │
  └──────────────────────────────────────────────────┘
                                │
                                ▼
  _wait_age_decay_refresh()  ← 同步屏障
                                │
                                ▼
  sample_for_training()  ← 使用刷新后的优先级采样
```

**关键设计决策**：

| 设计点 | 选择 | 理由 |
|--------|------|------|
| 线程数 | 1 | 避免竞争条件，单线程足够 |
| 超时时间 | 30秒 | 防止死锁，优雅降级 |
| 刷新时机 | Actor Train期间 | CPU空闲，不增加关键路径延迟 |
| 同步点 | 采样前 | 确保采样使用最新优先级 |

---

## 6. 完整优先级生命周期

### 6.1 阶段概览

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Phase 1     │     │  Phase 2     │     │  Phase 3     │     │  Phase 4     │
│  初始存储     │────►│  异步衰减     │────►│  PER采样     │────►│  训练后更新   │
│              │     │              │     │              │     │              │
│ 新轨迹入库   │     │ 后台刷新age  │     │ 分层采样     │     │ 基于advantage│
│ 初始priority │     │ 更新Seg Tree │     │ 返回indices  │     │ 更新priority │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
   max_p^α            p × exp(-age/τ)^α     P(i) ∝ tree[i]     new_p × w^α
```

### 6.2 Phase 1：初始存储

代码位置：`trajectory_buffer.py: push_from_dataproto` (L119-225)

触发时机：`agentic_pipeline.py: store_fresh_data_to_replay_buffer`

```python
# 1. 计算初始优先级
priority = priority_fn(trajectory, global_step, age_decay=self.age_decay)
# 对于reward_fresh: priority = (|reward| + ε) × exp(-0/decay) = |reward| + ε
# （age=0，刚存入，freshness=1.0）

# 2. 存入Segment Tree（使用max_priority保证至少被采样一次）
priority_alpha = self._max_priority ** self.priority_exponent
self._it_sum[slot_idx] = priority_alpha   # 用于采样概率
self._it_min[slot_idx] = priority_alpha   # 用于IS权重
```

**设计理由**：新样本使用`max_priority`而非计算值，这是标准PER做法——确保每个样本至少有一次被采样的机会。

### 6.3 Phase 2：异步Age Decay刷新

代码位置：`trajectory_buffer.py: refresh_all_age_decay` (L691-763)

触发时机：`agentic_pipeline.py: _async_refresh_age_decay` (Actor Train期间)

```python
# 遍历buffer中所有有效轨迹
for idx in range(self.capacity):
    if not self.valid_mask[idx]:
        continue

    trajectory = self.trajectories[idx]
    age = current_global_step - trajectory.global_step
    freshness_weight = math.exp(-age / self.age_decay)

    # 使用存储的priority乘以freshness权重
    effective_priority = max(trajectory.priority * freshness_weight, 1e-8)

    # 更新Segment Tree
    priority_alpha = effective_priority ** self.priority_exponent
    self._it_sum[idx] = priority_alpha
    self._it_min[idx] = priority_alpha
```

**复杂度**：O(capacity)，典型耗时 50-100ms（capacity=50k-100k）

### 6.4 Phase 3：PER采样

代码位置：`trajectory_buffer.py: sample_for_training` (L306-511)

采样概率：`P(sample_i) = tree[i] / Σ tree[j]`

```python
# 返回值包含sampled_indices，用于Phase 4的优先级更新
data_proto, sampled_indices = buffer.sample_for_training(batch_size)
```

### 6.5 Phase 4：训练后优先级更新

代码位置：`agentic_pipeline.py: _update_replay_priorities` (L1578-1660)

触发时机：Replay Training完成后

```python
# 1. 根据priority_function确定更新指标
priority_metric = get_update_metric(self._priority_function)
# reward_fresh → "reward"
# advantage   → "advantage"
# td_error    → "td_error"

# 2. 从训练batch中提取新优先级值
if priority_metric == 'advantage':
    priorities = |advantages|.mean(dim=1)
elif priority_metric == 'td_error':
    priorities = |td_errors|.mean(dim=1)
elif priority_metric == 'reward':
    priorities = |scores.sum(dim=1)|

# 3. 调用buffer更新
self.replay_buffer.update_priorities(
    indices=sampled_indices,
    priorities=priorities,
    current_global_step=global_step
)
```

在`update_priorities`内部：

```python
# trajectory_buffer.py: update_priorities (L610-660)
for slot_idx, priority in zip(indices, priorities):
    # 更新存储的intrinsic priority
    self.trajectories[slot_idx].priority = float(priority)

    # 重新计算含age decay的有效优先级
    if self.enable_age_decay:
        age = global_step - self.trajectories[slot_idx].global_step
        freshness = math.exp(-age / self.age_decay)
        effective = priority * freshness
    else:
        effective = priority

    # 更新Segment Tree
    self._it_sum[slot_idx] = effective ** self.priority_exponent
    self._it_min[slot_idx] = effective ** self.priority_exponent

    # 更新max_priority（用于新样本初始化）
    self._max_priority = max(self._max_priority, priority)
```

---

## 7. 配置参数

### 7.1 核心配置

```yaml
replay:
  enabled: true
  capacity: 50000                    # Buffer容量

  # 采样策略
  priority_function: "reward_fresh"  # 优先级函数
  priority_exponent: 0.6             # PER α参数 (0=均匀, 1=完全按优先级)
  sampling_mode: "trajectory"        # 轨迹级采样

  # Age Decay配置
  enable_age_decay: true             # 启用时间衰减
  age_decay: 1000.0                  # 衰减常数 (τ)

  # 训练配置
  train_steps_per_env_step: 2        # 每次环境交互后训练2次
  use_rollout_batch_size: true       # 使用rollout batch size

  # Importance Sampling校正
  importance_sampling_correction: false
  importance_beta: 0.4               # IS β参数

  # 存储配置
  storage_mode: tokens_only          # 只存储token
  eviction_strategy: "fifo"          # FIFO淘汰策略
```

### 7.2 参数调优建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `priority_function` | `reward_fresh` | off-policy LLM RL最佳选择 |
| `priority_exponent` | 0.6 | 适中的优先级区分度 |
| `age_decay` | 1000 | FrozenLake验证最佳；CliffWalking待验证 |
| `train_steps_per_env_step` | 2 | 2倍样本利用率 |
| `importance_sampling_correction` | false | 实验表明IS校正对LLM RL帮助有限 |

---

## 8. 与现有文档的关系

| 文档 | 内容 | 与本文档关系 |
|------|------|-------------|
| `replay_buffer.md` | Replay Buffer基础架构、Padding重构 | 本文档是其PER部分的深入扩展 |
| `age_decay_issue_analysis.md` | Age Decay问题分析和时序图 | 本文档中的异步方案是该分析的最终解决方案 |
| `replay_buffer_config_guide.md` | 所有配置参数详解 | 本文档补充了PER相关参数的设计理由 |
| `revisited_replay_buffer.md` | 经验回放的理论基础 | 本文档的实现基于该理论框架 |

---

## 9. 代码位置索引

| 组件 | 文件 | 关键行号 |
|------|------|---------|
| 优先级函数定义 | `roll/agentic/replay_buffer/priority_functions.py` | L24-400 |
| Reward-Fresh函数 | `priority_functions.py` | L275-336 |
| 更新指标映射 | `priority_functions.py` | L384-400 |
| Segment Tree | `roll/agentic/replay_buffer/segment_tree.py` | 全文件 |
| 轨迹Buffer | `roll/agentic/replay_buffer/trajectory_buffer.py` | 全文件 |
| 初始存储 | `trajectory_buffer.py: push_from_dataproto` | L119-225 |
| PER采样 | `trajectory_buffer.py: sample_for_training` | L306-511 |
| 优先级更新 | `trajectory_buffer.py: update_priorities` | L610-660 |
| Age Decay刷新 | `trajectory_buffer.py: refresh_all_age_decay` | L691-763 |
| ThreadPool初始化 | `agentic_pipeline.py` | L223-238 |
| 异步提交 | `agentic_pipeline.py: _async_refresh_age_decay` | L240-270 |
| 同步等待 | `agentic_pipeline.py: _wait_age_decay_refresh` | L272-287 |
| 训练后更新 | `agentic_pipeline.py: _update_replay_priorities` | L1578-1660 |
