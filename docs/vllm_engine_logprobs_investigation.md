# VLLM 推理引擎 Log Probabilities 调研报告

## 1. 背景：behavior_log_probs 存储 bug

在当前 `agentic_pipeline.py` 中，存入 replay buffer 的 `behavior_log_probs` 并非真正的 π_μ（行为策略，即生成数据时的推理引擎策略），而是训练后的 π_θ：

```
Line 749:  actor_train.train_step(batch)          # 策略参数更新: π_μ → π_θ
Line 823:  batch.batch["behavior_log_probs"] = actor_train_metrics.batch["log_probs"]  # 这是 π_θ!
Line 827:  store_fresh_data_to_replay_buffer(batch, global_step)  # 存入的是 π_θ 而非 π_μ
```

要获得真正的 π_μ，最直接的方式是从 VLLM 推理引擎在生成时直接获取 log probabilities。

---

## 2. 调研结论：VLLM 引擎已支持返回 logprobs

**结论：基础设施已完整实现，但数据流在 agentic 管线中断裂。**

ROLL 框架中已有完整的 VLLM logprobs 获取链路，只是 agentic pipeline 的 env_manager 没有消费这个数据。

---

## 3. 现有实现链路分析

### 3.1 SamplingParams 配置

**文件**: `roll/distributed/strategy/vllm_strategy.py:358-381`

```python
def create_sampling_params_for_vllm(gen_kwargs):
    want_engine_logprobs = gen_kwargs.get("old_prob_compute", "trainer") == "engine"
    logprobs_flag = 1 if want_engine_logprobs else 0

    return SamplingParams(
        max_tokens=gen_kwargs["max_new_tokens"],
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        top_k=gen_kwargs["top_k"],
        stop_token_ids=gen_kwargs["eos_token_id"],
        repetition_penalty=gen_kwargs["repetition_penalty"],
        n=gen_kwargs["num_return_sequences"],
        logprobs=logprobs_flag,       # <-- 已支持！
    )
```

- 当 `old_prob_compute="engine"` 时，`logprobs=1`，VLLM 会在推理时计算并返回每个 token 的 log probability
- 默认 `old_prob_compute="trainer"`，`logprobs=0`，不返回

### 3.2 VLLM 输出提取

**文件**: `roll/distributed/strategy/vllm_strategy.py:176-207`

```python
def process_vllm_output(self, vllm_outputs, request_complete_callback):
    for request_output in vllm_outputs:
        output_logprobs = []
        for i, completion_output in enumerate(request_output.outputs):
            output_token_ids.append(completion_output.token_ids)
            if hasattr(completion_output, 'logprobs') and completion_output.logprobs is not None:
                token_logprobs = []
                for logprob_obj in completion_output.logprobs:
                    if logprob_obj is not None:
                        token_logprobs.append(logprob_obj.logprob if hasattr(logprob_obj, 'logprob') else 0.0)
                    else:
                        token_logprobs.append(0.0)
                output_logprobs.append(token_logprobs)
        output_data = DataProto(meta_info=self.request_metas[request_id])
        output_data.meta_info["output_token_ids"] = output_token_ids
        if output_logprobs:
            output_data.meta_info["output_logprobs"] = output_logprobs   # <-- 存入 meta_info
        request_complete_callback(data=output_data)
```

VLLM 的 `CompletionOutput.logprobs` 是一个 `List[Logprob]`，每个 `Logprob` 对象有 `.logprob` 属性（float）。提取后存入 `meta_info["output_logprobs"]`。

### 3.3 GenerateScheduler 转换为 Tensor

**文件**: `roll/distributed/scheduler/generate_scheduler.py`

有两个处理位置：

**DynamicSamplingScheduler.postprocess_output_ids()** (line 692-730):
```python
if "output_logprobs" in data.meta_info:
    output_logprobs = data.meta_info["output_logprobs"]
    batch_size = len(output_logprobs)
    seq_len = self.pipeline_config.sequence_length
    generation_log_probs = torch.zeros(batch_size, seq_len - 1, dtype=torch.float32)

    for i, seq_logprobs in enumerate(output_logprobs):
        start_idx = input_length - 1   # next-token indexing
        end_idx = min(start_idx + len(seq_logprobs), seq_len - 1)
        generation_log_probs[i, start_idx:end_idx] = torch.tensor(
            seq_logprobs[:end_idx - start_idx], dtype=torch.float32
        )

    output.batch["generation_log_probs"] = generation_log_probs   # <-- 附加到 batch
```

**RequestScheduler.generate_one_request()** (line 856-886):
类似处理，将 `output_logprobs` 转换为与 `response_mask` 对齐的 tensor。

### 3.4 数据流断裂点

**关键问题：`generation_log_probs` 在 generate_scheduler 中创建后，没有被 agentic 管线消费。**

搜索验证：
```
grep -rn "generation_log_probs" roll/agentic/     → 无结果
grep -rn "generation_log_probs" roll/pipeline/    → 无结果
```

原因：
1. **TrajEnvManager.formulate_rollouts()** (`traj_env_manager.py:308-419`) 会**重新 tokenize** 整个对话历史来构建训练数据，完全不使用 generate_scheduler 的输出 batch
2. generate_scheduler 的输出 batch 是**单步推理结果**，而 TrajEnvManager 在多步交互后才构建**整条轨迹**的训练数据
3. 因此即使 `generation_log_probs` 存在于中间 batch 中，也会在 env_manager 重新构建时丢失

---

## 4. VLLM logprobs 的语义与局限性

### 4.1 VLLM logprobs 代表什么

VLLM 返回的 logprobs 是**推理引擎在 autoregressive 采样过程中**，每个 token 被选中时的 log probability。这正是真正的 π_μ(a_t | s_t)。

### 4.2 与 actor_train.compute_log_probs 的区别

| 特性 | VLLM engine logprobs | actor_train.compute_log_probs |
|------|---------------------|-------------------------------|
| 计算时机 | 生成时（推理引擎） | 训练前（训练模型） |
| 代表的策略 | π_μ（行为策略） | π_θ（可能已更新） |
| 前向传播 | 无额外开销（生成时顺带） | 需要额外一次完整前向传播 |
| 计算精度 | 推理精度（通常 fp16/bf16） | 训练精度 |
| 覆盖范围 | 仅 response tokens | 可覆盖 prompt + response |

### 4.3 多轮对话场景的复杂性

在 agentic 多轮对话中，一条轨迹包含多个 turn，每个 turn 由**不同时刻**的推理引擎策略生成：

```
Turn 1: π_μ^(t=0) 生成 response_1    → logprobs_1
Turn 2: π_μ^(t=0) 生成 response_2    → logprobs_2  (同一 global_step 内)
Turn 3: π_μ^(t=0) 生成 response_3    → logprobs_3
```

在同一个 global_step 内的多轮交互中，推理引擎参数不变（模型权重在 rollout 结束后才更新），所以所有 turn 的 logprobs 都来自同一个 π_μ^(t=0)。

但要注意：每个 turn 的 logprobs 只覆盖**该 turn 自己生成的 tokens**，不包括之前 turn 的 response tokens。如果需要整条轨迹所有 response tokens 的 logprobs，就需要在生成完所有 turn 后，用 actor_train 做一次完整的前向传播。

### 4.4 temperature 与 logprobs 的关系（重要警告）

**VLLM 默认返回的是 raw logprobs（未经 temperature 缩放），而非采样分布的 logprobs。**

这是 VLLM 社区的已知问题：
- [Issue #9453](https://github.com/vllm-project/vllm/issues/9453): Logprob values are affected by sampling parameters
- [TRL Issue #4159](https://github.com/huggingface/trl/issues/4159): vLLM not computing correct log probs when using GRPO with !=1 temperature

#### 问题本质

```
Raw logprobs:    log(softmax(logits))           # temperature=1 等价
Scaled logprobs: log(softmax(logits / T))       # 实际采样分布
```

当 temperature != 1.0 时（ROLL 默认 temperature=0.95），两者不同：
- **Raw logprobs** = 模型原始概率，不含 temperature 影响
- **Scaled logprobs** = 实际采样分布 π_μ(a|s)，包含 temperature 缩放

对于 importance sampling，我们需要的是**实际采样分布**的 logprobs（scaled），因为：
```
ratio = π_θ(a|s) / π_μ(a|s)
```
其中 π_μ 是包含 temperature 的实际采样分布。

#### 解决方案

**VLLM >= 0.10.2** 引入了 `logprobs_mode` 参数：

```python
# 在 SamplingParams 中设置
SamplingParams(
    logprobs=1,
    logprobs_mode="processed_logprobs",  # 返回经过 temperature/top_p/top_k 处理后的 logprobs
)
```

ROLL 目前支持的 VLLM 版本：
- `vllm_0_7_3` — 不支持 `logprobs_mode`
- `vllm_0_8_4` — 不支持 `logprobs_mode`
- `vllm_0_10_0` — **需确认**是否支持（0.10.0 vs 0.10.2）

#### 对我们方案的影响

| 场景 | 是否需要 `logprobs_mode` |
|------|------------------------|
| temperature=1.0 | 不需要，raw = scaled |
| temperature!=1.0，trainer 也用 temperature | 需要，否则 ratio 计算有偏 |
| temperature!=1.0，但只用于采样不用于 ratio | 不需要 |

**建议**：
1. 如果 VLLM >= 0.10.2，在 `create_sampling_params_for_vllm()` 中添加 `logprobs_mode="processed_logprobs"`
2. 如果 VLLM < 0.10.2，需要在获取 raw logprobs 后手动做 temperature 缩放：`scaled_logprob = raw_logprob / T`（近似，不完全等价于 softmax(logits/T)）
3. 或者强制 temperature=1.0 来规避此问题

### 4.5 单 turn logprobs vs 整轨迹 logprobs

这里有两种用法：

1. **Engine logprobs (per-turn)**：每个 turn 生成时 VLLM 顺便返回的 logprobs。优点是零额外开销、是真正的 π_μ。缺点是只有当前 turn 的 logprobs，不含历史 turn 的（因为那些 tokens 当时还没生成）。

2. **Trainer logprobs (full-trajectory)**：生成完所有 turn 后，把整条轨迹喂给 actor_train 做一次前向传播。优点是覆盖所有 response tokens。缺点是需要额外前向传播、且如果模型在此期间更新过，就不再是 π_μ。

> 对于 replay buffer 的 importance sampling 来说：
> - 如果只关心 **last turn**（`old_prob_mode=turn`）：engine logprobs 完全足够
> - 如果需要 **整条轨迹**（`old_prob_mode=trajectory`）：需要拼接所有 turn 的 engine logprobs，或做一次完整前向传播

---

## 5. 修复方案

### 方案 A：使用 Engine Logprobs（推荐）

**原理**：在生成时设置 `logprobs=1`，收集每个 turn 的 engine logprobs，拼接成完整轨迹的 behavior_log_probs。

**需要修改的代码**：

1. **配置修改**：设置 `old_prob_compute: engine`（或新增独立配置项 `save_engine_logprobs: true`）

2. **TrajEnvManager 修改**：在多轮交互过程中，收集每个 turn 的 VLLM logprobs 并保存到 `rollout_cache`

3. **formulate_rollouts() 修改**：将收集到的 per-turn logprobs 拼接成与完整 response_mask 对齐的 tensor

**优点**：
- 零额外前向传播开销
- 是真正的 π_μ
- 推理时顺便获取，不影响吞吐量（VLLM logprobs=1 的开销极小）

**缺点**：
- 需要在 env_manager 中做额外的 logprobs 收集和对齐逻辑
- per-turn logprobs 需要与重新 tokenize 后的 position 正确对齐（可能有 tokenization 不一致的风险）

### 方案 B：训练前保存 behavior_log_probs（简单修复）

**原理**：在 `actor_train.train_step()` 之前保存 `behavior_log_probs`，训练后恢复。

```python
# agentic_pipeline.py，训练前
true_behavior_log_probs = batch.batch.get("behavior_log_probs", None)

# ... train_step ...

# 存储到 replay buffer 前恢复
if true_behavior_log_probs is not None:
    batch.batch["behavior_log_probs"] = true_behavior_log_probs
self.store_fresh_data_to_replay_buffer(batch, global_step)
```

**优点**：改动极小，一行代码修复
**缺点**：仍然依赖 `_compute_and_attach_behavior_log_probs()` 的额外前向传播

### 方案 C：混合方案

- 使用 Engine Logprobs 作为 per-turn 的真实 π_μ
- 在需要 full-trajectory logprobs 时，用 actor_train 补充计算（但在 train_step 之前而非之后）

---

## 6. 数据流对比

### 当前错误流程
```
Rollout (π_μ) → [VLLM logprobs 被丢弃]
            → _compute_and_attach_behavior_log_probs (π_μ, 额外前向传播)
            → train_step (π_μ → π_θ)
            → batch["behavior_log_probs"] = train_metrics["log_probs"]  (π_θ!)
            → store_to_replay_buffer  (存的是 π_θ)
```

### 方案 A 修复后
```
Rollout (π_μ) → [VLLM logprobs 收集到 rollout_cache]
            → formulate_rollouts → batch["behavior_log_probs"] = 拼接的 engine logprobs (π_μ)
            → train_step
            → store_to_replay_buffer  (存的是 π_μ)
```

### 方案 B 修复后
```
Rollout (π_μ) → _compute_and_attach_behavior_log_probs (π_μ)
            → true_behavior = batch["behavior_log_probs"]  (保存 π_μ)
            → train_step
            → batch["behavior_log_probs"] = true_behavior  (恢复 π_μ)
            → store_to_replay_buffer  (存的是 π_μ)
```

---

## 7. 关键代码位置索引

| 组件 | 文件 | 行号 | 说明 |
|------|------|------|------|
| SamplingParams 配置 | `roll/distributed/strategy/vllm_strategy.py` | 358-381 | `logprobs` 参数设置 |
| VLLM 输出提取 | `roll/distributed/strategy/vllm_strategy.py` | 176-207 | 从 CompletionOutput 提取 logprobs |
| Tensor 转换 | `roll/distributed/scheduler/generate_scheduler.py` | 692-730 | 转为 `generation_log_probs` tensor |
| Tensor 转换 (async) | `roll/distributed/scheduler/generate_scheduler.py` | 856-886 | RequestScheduler 版本 |
| 断裂点 | `roll/pipeline/agentic/env_manager/traj_env_manager.py` | 308-419 | formulate_rollouts 不使用 generation_log_probs |
| Bug 位置 | `roll/pipeline/agentic/agentic_pipeline.py` | 823 | 用 π_θ 覆盖 π_μ |
| behavior_log_probs 计算 | `roll/pipeline/agentic/agentic_pipeline.py` | 1625-1658 | 额外前向传播获取 |

---

## 8. VLLM GitHub 调研结论

### 8.1 可行性：完全可行

VLLM 原生支持在生成时返回 logprobs，通过 `SamplingParams(logprobs=1)` 即可启用。
ROLL 框架中已有完整的提取链路（vllm_strategy → generate_scheduler → `generation_log_probs` tensor），
只是 agentic env_manager 没有消费这个数据。

### 8.2 关键注意事项

1. **temperature != 1.0 时 logprobs 语义问题**（[Issue #9453](https://github.com/vllm-project/vllm/issues/9453)）
   - VLLM 默认返回 raw logprobs（未经 temperature 缩放）
   - 对于 importance sampling 需要实际采样分布的 logprobs
   - VLLM >= 0.10.2 可用 `logprobs_mode="processed_logprobs"` 解决
   - VLLM < 0.10.2 需手动缩放或强制 temperature=1.0

2. **CompletionOutput.logprobs 数据结构**
   - `logprobs` 字段是 `List[Logprob]`，每个 `Logprob` 有 `.logprob` 属性（float）
   - 当 `SamplingParams(logprobs=N)` 时，返回 top-N 候选 token 的 logprobs + 被选中 token 的 logprob
   - `logprobs=1` 足以获取被选中 token 的 log probability

3. **性能开销极小**
   - `logprobs=1` 只要求返回被选中 token 的 logprob，不需要额外的 softmax 计算（推理时本就需要）
   - 相比额外的 `actor_train.compute_log_probs()` 前向传播，引擎 logprobs 几乎零开销

### 8.3 实现方案 A 的具体步骤

1. 在 `create_sampling_params_for_vllm()` 中始终设置 `logprobs=1`（不依赖 `old_prob_compute` 配置）
2. 在 agentic 多轮交互过程中，将每个 turn 的 VLLM logprobs 保存到 `rollout_cache`
3. 在 `TrajEnvManager.formulate_rollouts()` 中，将 per-turn logprobs 拼接为完整轨迹的 `behavior_log_probs`
4. 如果 temperature != 1.0，确保使用 `logprobs_mode="processed_logprobs"`（VLLM >= 0.10.2）或手动缩放

### 8.4 参考资源

- [VLLM SamplingParams 文档](https://docs.vllm.ai/en/latest/api/vllm/sampling_params.html)
- [Issue #9453: Logprobs affected by sampling parameters](https://github.com/vllm-project/vllm/issues/9453)
- [TRL Issue #4159: GRPO logprobs with temperature != 1](https://github.com/huggingface/trl/issues/4159)
- [Discussion #1203: How to use logprobs](https://github.com/vllm-project/vllm/discussions/1203)
- [Issue #29280: Selective Token Logprobs Tracking](https://github.com/vllm-project/vllm/issues/29280)
