# Engine Logprobs 实现方案：从 VLLM 获取真正的 pi_mu

## 1. 问题回顾

当前 `agentic_pipeline.py:823` 将训练后的 log_probs（pi_theta）覆盖了 behavior_log_probs，
导致存入 replay buffer 的不是真正的行为策略 pi_mu。

## 2. 数据流现状

```
VLLM generate
  -> vllm_strategy.process_vllm_output()    # 提取 output_logprobs -> meta_info
  -> generate_scheduler.generate_one_request()  # 转换为 batch["generation_log_probs"] tensor
  -> PolicyProxy.generate()                 # 返回给 TrajEnvManager
  -> TrajEnvManager.make_decision()         # lm_output 包含 generation_log_probs
  -> TrajEnvManager.step(lm_output)         # 只提取 responses 文本，logprobs 被丢弃！
  -> TrajEnvManager.formulate_rollouts()    # 重新 tokenize，完全没有 logprobs
```

## 3. 修改清单

### 修改 1: vllm_strategy.py — 始终启用 logprobs

**文件**: `roll/distributed/strategy/vllm_strategy.py`
**行**: 358-381

**当前**:
```python
want_engine_logprobs = gen_kwargs.get("old_prob_compute", "trainer") == "engine"
logprobs_flag = 1 if want_engine_logprobs else 0
```

**改为**:
```python
# 始终从引擎获取 logprobs，用于 replay buffer 存储真正的 pi_mu
logprobs_flag = 1
```

另外需要处理 temperature 问题。当前 ROLL 支持 vllm_0_10_0，
需要确认是否支持 `logprobs_mode="processed_logprobs"`（VLLM >= 0.10.2 才有）。

如果升级到 >= 0.10.2：
```python
return SamplingParams(
    ...
    logprobs=1,
    logprobs_mode="processed_logprobs",  # 返回 temperature-scaled logprobs
)
```

如果暂时不升级，temperature=1.0 时无影响；temperature!=1.0 时需要在下游手动缩放。

---

### 修改 2: traj_env_manager.py — step() 中保存 per-turn logprobs

**文件**: `roll/pipeline/agentic/env_manager/traj_env_manager.py`
**行**: 202-239

在 `step()` 方法中，从 `llm_output` 提取 response token 的 logprobs 并存入 history。

`lm_output` 的 `batch["generation_log_probs"]` 是 shape `[1, full_seq_len]` 的 tensor，
其中 response token 位置有真实的 logprob 值，prompt 位置为 0。

```python
def step(self, llm_output: DataProto):
    responses = self.tokenizer.batch_decode(
        llm_output.batch['responses'],
        skip_special_tokens=True
    )

    # --- 新增：提取 engine logprobs ---
    if "generation_log_probs" in llm_output.batch:
        gen_lp = llm_output.batch["generation_log_probs"][0]       # [full_seq_len]
        resp_mask = llm_output.batch.get("response_mask", None)
        if resp_mask is not None:
            resp_mask = resp_mask[0].bool()                        # [full_seq_len]
            resp_logprobs = gen_lp[resp_mask].tolist()              # 只取 response tokens
        else:
            resp_logprobs = gen_lp[gen_lp != 0].tolist()           # fallback
    else:
        resp_logprobs = None
    # --- 新增结束 ---

    next_state, reward, terminated, truncated, info = self.env.step(action=responses[0])

    # ... 保持现有逻辑不变 ...

    self.rollout_cache.history[-1]['llm_response'] = responses[0]

    # --- 新增：保存 logprobs 到 history ---
    if resp_logprobs is not None:
        self.rollout_cache.history[-1]['response_logprobs'] = resp_logprobs
    # --- 新增结束 ---

    # ... 其余逻辑不变 ...
```

**存储的数据**: `history[turn_idx]['response_logprobs']` = `List[float]`，
长度等于该 turn 生成的 response token 数量。

---

### 修改 3: traj_env_manager.py — formulate_rollouts() 中组装 behavior_log_probs

**文件**: `roll/pipeline/agentic/env_manager/traj_env_manager.py`
**行**: 308-419

在 `formulate_rollouts()` 中，利用已有的 `response_masks_list`（per-turn 的 mask）
和存储的 `response_logprobs`，组装完整轨迹的 `behavior_log_probs`。

关键逻辑：

```python
def formulate_rollouts(self, rollout_cache: RolloutCache) -> Optional[DataProto]:
    # ... 现有代码到 response_mask 计算之后 ...

    # --- 新增：组装 behavior_log_probs ---
    has_engine_logprobs = any(
        'response_logprobs' in h for h in self.rollout_cache.history
        if 'reward' in h  # 只看有 reward 的 step（即完成了交互的 step）
    )

    if has_engine_logprobs:
        # behavior_log_probs: 与 response_mask 对齐的 tensor
        # shape: [1, seq_len]（后续在 pipeline 中会转为 [1, seq_len-1] 的 next-token 格式）
        behavior_log_probs = torch.zeros(1, len(token_ids), dtype=torch.float32)

        # response_masks_list 是 per-segment 的 mask 列表
        # 我们需要找到每个 assistant turn 对应的 segment，并将 logprobs 放入
        turn_idx = 0
        completed_steps = [
            h for h in self.rollout_cache.history if 'reward' in h
        ]

        # 遍历 response_masks_list，每组 segments 对应一个 turn
        pos = 0  # 当前在 token_ids 中的位置
        for seg_masks in response_masks_list:
            seg_len = len(seg_masks)
            # 检查这个 segment 是否包含 response tokens
            resp_positions = [i for i, m in enumerate(seg_masks) if m == 1]
            if resp_positions and turn_idx < len(completed_steps):
                step_data = completed_steps[turn_idx]
                stored_lp = step_data.get('response_logprobs', None)
                if stored_lp is not None:
                    # 对齐长度：re-tokenize 可能产生不同数量的 response tokens
                    n_resp = len(resp_positions)
                    n_stored = len(stored_lp)
                    n_use = min(n_resp, n_stored)
                    if n_resp != n_stored:
                        self.logger.debug(
                            f"[ENGINE_LP] Turn {turn_idx}: token mismatch: "
                            f"re-tokenized={n_resp}, stored={n_stored}, using={n_use}"
                        )
                    for j in range(n_use):
                        behavior_log_probs[0, pos + resp_positions[j]] = stored_lp[j]
                turn_idx += 1
            pos += seg_len
    # --- 新增结束 ---

    # 截断到 last_response_idx+1（与现有逻辑一致）
    # ...

    lm_input.batch = TensorDict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "penalty": torch.Tensor([episode_penalty]),
        "response_mask": response_mask,
        "prompt_mask": prompt_mask,
        "scores": score_tensor,
    }, batch_size=input_ids.shape[0])

    # --- 新增：附加 behavior_log_probs ---
    if has_engine_logprobs:
        # 截断到与其他 tensor 相同长度
        behavior_log_probs = behavior_log_probs[:, :last_response_idx+1]
        lm_input.batch["behavior_log_probs"] = behavior_log_probs
    # --- 新增结束 ---
```

> **注意**: `behavior_log_probs` 在这里的 shape 是 `[1, seq_len]`（与 input_ids 对齐）。
> 但 PPO 训练中 old_log_probs 通常是 `[1, seq_len-1]`（next-token prediction 格式）。
> 这个对齐在 replay buffer 存储时处理（`trajectory_buffer.py` 已有对应逻辑）。

实际上这里需要注意 behavior_log_probs 和现有代码对log_probs的约定格式：
- 现有 `actor_train.compute_log_probs()` 返回的是 `[batch, seq_len-1]`（next-token）
- VLLM engine 返回的 logprobs 是 per-generated-token
- 需要统一为 next-token 格式

**更简单的做法**：在 formulate_rollouts 中直接输出 `[1, seq_len-1]` 的 next-token 格式：

```python
# next-token 格式：position i 的 logprob 对应 token i+1 的概率
# response token 在位置 [resp_start, resp_end]
# 其 next-token logprob 在 [resp_start-1, resp_end-1]
behavior_log_probs_nt = torch.zeros(1, len(token_ids) - 1, dtype=torch.float32)
# ... 放置 logprobs 到 next-token 位置 ...
behavior_log_probs_nt = behavior_log_probs_nt[:, :last_response_idx]  # 截断
lm_input.batch["behavior_log_probs"] = behavior_log_probs_nt
```

---

### 修改 4: agentic_pipeline.py — 停止覆盖 behavior_log_probs

**文件**: `roll/pipeline/agentic/agentic_pipeline.py`
**行**: 817-827

**当前（错误）**:
```python
if self.pipeline_config.replay.enabled and self.pipeline_config.critic_warmup <= global_step:
    if actor_train_metrics.batch is not None and "log_probs" in actor_train_metrics.batch:
        batch.batch["behavior_log_probs"] = actor_train_metrics.batch["log_probs"]  # 用 pi_theta 覆盖!
    self.store_fresh_data_to_replay_buffer(batch, global_step)
```

**改为**:
```python
if self.pipeline_config.replay.enabled and self.pipeline_config.critic_warmup <= global_step:
    # behavior_log_probs 已在 TrajEnvManager.formulate_rollouts() 中
    # 从 VLLM engine logprobs 设置，是真正的 pi_mu，不再覆盖
    self.store_fresh_data_to_replay_buffer(batch, global_step)
```

---

### 修改 5: agentic_pipeline.py — 简化 behavior_log_probs 计算逻辑

**行**: 326-331, 415-420

**当前**: 有条件地调用 `_compute_and_attach_behavior_log_probs()`（额外前向传播）

**改为**: 如果 engine logprobs 已经附加在 batch 中（来自 formulate_rollouts），
直接跳过额外前向传播。

```python
# Line 326-331: 移除或改为 fallback
if "behavior_log_probs" not in batch.batch:
    # Fallback: engine logprobs 不可用时，用 actor_train 计算
    if self.pipeline_config.offpolicy_monitor.enabled:
        batch = self._compute_and_attach_behavior_log_probs(batch)
```

---

### 修改 6: store_fresh_data_to_replay_buffer() — fallback 逻辑

**行**: 1675-1679

当前 fallback 会调用 `_compute_and_attach_behavior_log_probs()`。
保留此 fallback 用于 engine logprobs 不可用的情况。

---

## 4. 关于 tokenization 不一致的处理

`formulate_rollouts()` 中有一个已知问题（line 329 TODO）：
re-tokenize 可能产生与原始 generation 不同的 token 数量。

对于 logprobs 对齐：
- **大多数情况**：token 数量完全一致（正常的 encode → decode → encode 循环）
- **少数情况**：相差 1-2 个 token（如 leading space、special token 处理）
- **处理策略**：取 min(re_tokenized, stored) 长度对齐，多余的填 0.0
- **影响评估**：即使有少量 mismatch，engine logprobs 仍远优于使用 pi_theta

## 5. behavior_log_probs 格式约定

需确认 replay buffer 存储时的格式：

| 来源 | 格式 | shape |
|------|------|-------|
| actor_train.compute_log_probs() | next-token | [batch, seq_len-1] |
| VLLM engine (generate_one_request) | full-sequence, 0-padded | [batch, full_seq_len] |
| formulate_rollouts 输出 | 需统一为 next-token | [1, truncated_len-1] |
| trajectory_buffer 存储 | next-token (numpy) | [truncated_len-1] |

**结论**：在 `formulate_rollouts()` 中输出 `[1, seq_len-1]` 的 next-token 格式，
与现有 `actor_train.compute_log_probs()` 的输出保持一致。

## 6. VLLM 升级要求

- **最低要求**: VLLM 支持 `SamplingParams(logprobs=1)` — 所有版本都支持
- **推荐版本**: VLLM >= 0.10.2，支持 `logprobs_mode="processed_logprobs"`
- **影响**: 当 temperature != 1.0 时，返回 temperature-scaled logprobs（真正的采样分布）

## 7. 实施顺序

1. 升级 VLLM 到 >= 0.10.2（或确认 0.10.0 是否支持 logprobs_mode）
2. `vllm_strategy.py`: 始终 logprobs=1 + logprobs_mode
3. `traj_env_manager.py`: step() 保存 + formulate_rollouts() 组装
4. `agentic_pipeline.py`: 移除覆盖 + 简化 fallback
5. 测试：对比 engine logprobs 与 actor_train.compute_log_probs() 的一致性
