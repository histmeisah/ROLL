# Old Prob 完整实现文档

## 概述

Old Prob（旧策略概率）是ROLL框架中用于计算off-policy ratio的关键组件。本文档详细说明了old prob的完整实现，包括两个维度的配置选项。

## 配置选项

### 1. old_prob_mode（计算范围）

控制计算old log prob时覆盖的范围：

- **`trajectory`**（默认）：计算整个轨迹中所有assistant回复的log prob
- **`turn`**：仅计算最后一个assistant回复（当前轮次）的log prob

### 2. old_prob_compute（计算位置）

控制在哪里计算old log prob：

- **`trainer`**（默认）：在Actor-Train侧重新计算，更准确
- **`engine`**：使用推理引擎返回的log probs，更高效但可能不够准确

## 实现架构

### 1. Trainer模式实现

#### 1.1 数据流

```
生成阶段 → EnvManager → ReplayBuffer → 训练阶段
                           ↓
                    compute_log_probs
                    (Actor-Train)
```

#### 1.2 核心代码路径

**agentic_pipeline.py**：
```python
# 存储到replay buffer时计算behavior log probs
if old_prob_compute == "trainer":
    behavior_refs = self.actor_train.compute_log_probs(fresh_batch, blocking=False)
    behavior = DataProto.materialize_concat(data_refs=behavior_refs)
    fresh_batch.batch["behavior_log_probs"] = behavior.batch["log_probs"]
```

**base_worker.py**：
```python
def forward_func_log_probs(self, data: DataProto, output_tensor: torch.Tensor):
    # 检查是否需要使用turn模式
    old_prob_mode = data.meta_info.get("old_prob_mode", "trajectory")
    response_mask = data.batch["response_mask"]
    
    if old_prob_mode in ["step", "turn"] and "prompt_mask" in data.batch:
        # Turn模式：只计算最后一个assistant turn的log probs
        from roll.utils.turn_mode_utils import create_turn_mode_response_mask
        turn_response_mask, debug_info = create_turn_mode_response_mask(
            response_mask=response_mask,
            prompt_mask=data.batch.get("prompt_mask", None),
            messages_list=data.non_tensor_batch.get("messages_list", None)
        )
        response_mask_for_log_probs = turn_response_mask
    else:
        # Trajectory模式：使用原始response mask
        response_mask_for_log_probs = response_mask
```

### 2. Engine模式实现

#### 2.1 数据流

```
生成阶段 → VLLM/HF Engine → postprocess_generate → ReplayBuffer
            ↓                      ↓
         logprobs=1        generation_log_probs
```

#### 2.2 核心代码路径

**vllm_strategy.py**：
```python
def create_sampling_params_for_vllm(gen_kwargs):
    # 当选择engine计算路径时启用engine logprobs
    want_engine_logprobs = gen_kwargs.get("old_prob_compute", "trainer") == "engine"
    logprobs_flag = 1 if want_engine_logprobs else 0
    
    return SamplingParams(
        # ... 其他参数
        logprobs=logprobs_flag,
    )
```

**generate_scheduler.py**（推测实现）：
```python
# 在postprocess_output_ids中附加engine logprobs
if "output_logprobs" in response_data.meta_info:
    engine_logprobs = response_data.meta_info["output_logprobs"]
    # 处理并附加到output.batch["generation_log_probs"]
```

**agentic_pipeline.py**：
```python
# 使用engine提供的log probs
if old_prob_compute == "engine" and "generation_log_probs" in fresh_batch.batch:
    if old_prob_mode == "turn" and "prompt_mask" in fresh_batch.batch:
        # Apply turn mask to engine log probs
        turn_mask, _ = create_turn_mode_response_mask(
            response_mask=response_mask,
            prompt_mask=prompt_mask,
            messages_list=messages_list
        )
        # Apply mask to engine log probs
        masked_log_probs = engine_log_probs * turn_mask.float()
        fresh_batch.batch["behavior_log_probs"] = masked_log_probs
    else:
        # Trajectory mode: use engine log probs directly
        fresh_batch.batch["behavior_log_probs"] = fresh_batch.batch["generation_log_probs"]
```

## Turn模式详解

### 1. 动机

在多轮对话场景中，每个动作都是基于当时的context生成的，而不是基于完整轨迹。Turn模式旨在更准确地反映这种情况。

### 2. 实现原理

**turn_mode_utils.py**：
```python
def create_turn_mode_response_mask(response_mask, prompt_mask, messages_list):
    """
    创建一个只标记最后一个assistant turn的response mask。
    
    原理：
    1. 扫描原始response_mask，找到所有assistant回复段
    2. 定位最后一个连续的1序列（最后一个assistant turn）
    3. 创建新mask，只在最后一个turn的位置标记为1
    """
```

### 3. 适用场景

- **TrajEnvManager**：需要动态创建turn mask
- **StepEnvManager**：天然支持，因为每个样本只包含当前轮次

## 配置示例

### 1. Trajectory + Trainer（默认）
```yaml
old_prob_mode: trajectory
old_prob_compute: trainer
```
最准确但计算开销较大。

### 2. Turn + Trainer
```yaml
old_prob_mode: turn
old_prob_compute: trainer
```
更准确地反映off-policy情况，适合多轮对话场景。

### 3. Trajectory + Engine
```yaml
old_prob_mode: trajectory
old_prob_compute: engine
```
使用引擎返回的log probs，效率高但可能不够准确。

### 4. Turn + Engine
```yaml
old_prob_mode: turn
old_prob_compute: engine
```
结合了turn模式的准确性和engine模式的效率。在存储到replay buffer时会应用turn mask，只保留最后一轮的log probs。

## 注意事项

1. **Engine模式的限制**：
   - 需要推理引擎支持返回log probs
   - VLLM通过设置`logprobs=1`参数启用
   - HF策略可能需要额外实现

2. **Turn模式的要求**：
   - 需要`prompt_mask`来识别轮次边界
   - TrajEnvManager和StepEnvManager都生成了必要的masks

3. **性能考虑**：
   - Trainer模式需要额外的前向传播
   - Engine模式直接使用生成时的log probs，更高效
   - Turn模式减少了需要计算的token数量

4. **准确性权衡**：
   - Trainer模式使用相同的模型权重重新计算，最准确
   - Engine模式可能因为量化、批处理等因素导致细微差异

## 当前实现状态

### 已实现功能

1. **Trainer模式**：
   - ✅ Trajectory模式：完整实现
   - ✅ Turn模式：完整实现（通过`turn_mode_utils.py`）

2. **Engine模式**：
   - ✅ VLLM策略：已实现提取log probs的代码
   - ✅ generate_scheduler：将engine返回的log probs附加到`generation_log_probs`
   - ✅ Turn模式兼容性：已支持，在存储到replay buffer时应用turn mask
   - ⚠️  需要验证：VLLM实际输出格式可能需要调整

3. **环境管理器支持**：
   - ✅ TrajEnvManager：生成`prompt_mask`，支持turn模式
   - ✅ StepEnvManager：天然支持turn模式，每个样本只包含当前轮次

### 待完善功能

1. **HF策略的Engine模式**：
   - 需要确认HF策略是否支持返回log probs
   - 可能需要额外实现

## 未来改进方向

1. **Engine模式的完整实现**：
   - 确保HF策略也支持返回generation_log_probs
   - 处理不同推理引擎的log probs格式差异
   - 修复Engine + Turn模式的兼容性问题

2. **Turn模式的优化**：
   - 缓存turn mask的计算结果
   - 支持更复杂的turn识别逻辑

3. **监控和验证**：
   - 添加metrics来比较不同模式的off-policy ratio
   - 验证engine和trainer模式计算结果的一致性

4. **文档和测试**：
   - 添加单元测试覆盖所有模式组合
   - 提供性能基准测试结果
