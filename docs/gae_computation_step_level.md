# ROLL框架Step-Level GAE计算完整分析

## 文档目的

本文档详细分析ROLL框架在**step-level replay buffer**环境下，GAE（Generalized Advantage Estimation）的计算方式，包括：
- 有replay buffer和无replay buffer两种情况的完整数据流
- 不同adv_estimator（gae/reinforce/grpo/gigpo）的处理逻辑
- Token-level rewards的生成机制
- 与replay buffer n-step returns的关系

---

## 1. 核心流程概览

### 1.1 Pipeline中的调用顺序

**AgenticPipeline.run()主循环** (`agentic_pipeline.py:195-600`):

```python
for global_step in range(max_steps):
    # === Step 1: Rollout ===
    batch = self.train_rollout_scheduler.get_batch(...)

    # === Step 2: 🔑 Compute discounted returns (GIGPO特有) ===
    batch = compute_discounted_returns(
        batch,
        self.pipeline_config.adv_estimator,
        self.pipeline_config.step_reward_gamma
    )

    # === Step 3: Replay Buffer Integration ===
    if self.pipeline_config.replay.enabled:
        batch = self.integrate_replay_buffer_data(batch, global_step)
        # 内部会: push -> sample -> 返回sampled batch

    # === Step 4: 🔑 Compute response-level rewards ===
    batch = compute_response_level_rewards(
        batch=batch,
        pipeline_config=self.pipeline_config
    )

    # === Step 5: Compute ref log probs ===
    ref_log_probs = self.reference.compute_log_probs(batch)
    batch = batch.union(ref_log_probs)

    # === Step 6: Compute old log probs ===
    if not from_replay_buffer:
        old_log_probs = self.actor_train.compute_log_probs(batch)
        batch.batch["old_log_probs"] = old_log_probs

    # === Step 7: 🔑 Apply KL penalty (生成token_level_rewards) ===
    batch, kl_metrics = apply_kl_penalty(
        data=batch,
        kl_ctrl=self.kl_ctrl,
        kl_penalty=self.pipeline_config.kl_penalty
    )

    # === Step 8: 🔑 Compute advantages (GAE在这里!) ===
    batch = compute_advantage(
        data=batch,
        gamma=self.pipeline_config.gamma,
        lambd=self.pipeline_config.lambd,
        adv_estimator=self.pipeline_config.adv_estimator,
        advantage_clip=self.pipeline_config.advantage_clip,
        whiten_advantages=self.pipeline_config.whiten_advantages,
        whiten_rewards=self.pipeline_config.whiten_rewards,
    )

    # === Step 9: Training ===
    if self.pipeline_config.adv_estimator == "gae":
        self.critic.train_step(batch)
    self.actor_train.train_step(batch)
```

### 1.2 两种路径对比

| 特性 | 无Replay Buffer | 有Replay Buffer |
|------|----------------|----------------|
| **数据来源** | Fresh rollout batch | Sampled from buffer |
| **discounted returns** | 仅GIGPO计算 | 仅GIGPO计算 |
| **response_level_rewards** | 当前batch计算 | 当前batch计算 |
| **old_log_probs** | 实时计算 | 从buffer读取或fallback计算 |
| **token_level_rewards** | apply_kl_penalty生成 | apply_kl_penalty生成 |
| **GAE计算** | compute_advantage | compute_advantage |
| **N-Step Returns** | ❌ 不支持 | ✅ 在sample时计算（可选） |

**关键观察**: GAE计算逻辑在两种情况下**完全相同**，区别只在于数据来源。

---

## 2. 无Replay Buffer的GAE计算

### 2.1 完整数据流

```python
# === Step 1: Rollout产生fresh batch ===
# StepEnvManager.formulate_rollouts() 返回:
batch.batch = {
    "input_ids": [10, 2048],           # batch中10个steps
    "attention_mask": [10, 2048],
    "response_mask": [10, 2048],
    "scores": [10, 1],                 # 每个step的reward scalar
    "penalty": [10],                   # length penalty等
    # ...
}
batch.non_tensor_batch = {
    "traj_id": ["ep1", "ep1", "ep1", "ep1", "ep2", ...],  # 4 steps from ep1, ...
    "step": [0, 1, 2, 3, 0, ...],
    "step_scores": [[2.5], [2.5], [2.5], [2.5], [3.0], ...],
    "episode_scores": [[10.0], [10.0], [10.0], [10.0], [12.0], ...],
    # ...
}

# === Step 2: compute_discounted_returns (仅GIGPO) ===
# utils.py:73-107
if adv_estimator == "gigpo":
    # 按traj_id分组
    batch_group_by_traj = batch.group_by(keys="traj_id")

    for traj_id, traj_batch in batch_group_by_traj.items():
        # 按step排序
        indices = torch.argsort(traj_batch.non_tensor_batch["step"])
        traj_batch.reorder(indices)

        # 计算discounted returns
        step_scores = traj_batch.non_tensor_batch["step_scores"]  # [2.5, 2.5, 2.5, 2.5]
        rewards = torch.as_tensor(step_scores).float()
        discounts = torch.empty_like(rewards)
        running_return = 0.0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            discounts[t] = running_return

        # 示例: gamma=0.99
        # t=3: running_return = 2.5 + 0 = 2.5
        # t=2: running_return = 2.5 + 0.99*2.5 = 4.975
        # t=1: running_return = 2.5 + 0.99*4.975 = 7.42525
        # t=0: running_return = 2.5 + 0.99*7.42525 = 9.8509975

        traj_batch.batch["step_rewards"] = discounts  # [9.85, 7.43, 4.98, 2.50]

    batch = DataProto.concat(list(batch_group_by_traj.values()))
    batch.reorder(...)  # 恢复原始顺序
else:
    # gae/reinforce/grpo: 不计算,直接返回
    pass

# === Step 3: compute_response_level_rewards ===
# utils.py:147-176
if adv_estimator == "gigpo":
    # GIGPO: episode reward + step reward
    # 3.1 归一化episode rewards
    episode_scores = batch.non_tensor_batch["episode_scores"]  # [10.0, 10.0, ...]
    scores_with_penalty = episode_scores + batch.batch["penalty"]
    episode_rewards = grouped_reward_norm(scores_with_penalty, ...)  # 按traj_id归一化

    # 3.2 构建state groups (多轨迹共享state)
    batch = build_state_group(batch)

    # 3.3 归一化step rewards
    step_rewards = grouped_reward_norm(batch.batch["step_rewards"], grouping="state_group_id")

    # 3.4 加权组合
    batch.batch["response_level_rewards"] = (
        pipeline_config.episode_reward_weight * episode_rewards +
        pipeline_config.step_reward_weight * step_rewards
    )
else:
    # gae/reinforce/grpo: 只用scores
    scores_with_penalty = batch.batch["scores"].sum(dim=-1) + batch.batch["penalty"]
    batch.batch["response_level_rewards"] = grouped_reward_norm(scores_with_penalty, ...)

# 输出:
# batch.batch["response_level_rewards"]: [10] - 归一化后的reward scalar per step

# === Step 4: apply_kl_penalty ===
# functionals.py:644-677
response_mask = batch.batch["response_mask"][:, 1:]  # [10, 2047] - 去掉第一个token

# 4.1 Expand response_level_rewards to token-level
token_level_rewards = expand_to_token_level(batch)
# functionals.py:435-455
# 将response_level_rewards放到EOS token位置,其他位置为0
# token_level_rewards: [10, 2047] - 只有EOS位置有值

# 4.2 Compute KL divergence
if "ref_log_probs" in batch.batch:
    kld = compute_approx_kl(
        log_probs=batch.batch["old_log_probs"],      # [10, 2047]
        log_probs_base=batch.batch["ref_log_probs"],  # [10, 2047]
        action_mask=response_mask,                     # [10, 2047]
        kl_penalty="kl"
    )  # [10, 2047] - per-token KL
    beta = kl_ctrl.value  # 例如: 0.01
else:
    beta = 0
    kld = torch.zeros_like(response_mask, dtype=torch.float32)

# 4.3 Apply KL penalty
token_level_rewards = token_level_rewards - beta * kld  # [10, 2047]

# 4.4 Update KL controller
current_kl = masked_mean(kld, mask=response_mask, dim=-1).mean().item()
kl_ctrl.update(current=current_kl, n_steps=batch_size)

batch.batch["token_level_rewards"] = token_level_rewards  # [10, 2047]

# === Step 5: compute_advantage ===
# functionals.py:680-736
response_mask = batch.batch["response_mask"][:, 1:]  # [10, 2047]
token_level_rewards = batch.batch["token_level_rewards"].float()  # [10, 2047]

# 5.1 Whiten rewards (可选)
if whiten_rewards:
    token_level_rewards = masked_whiten(values=token_level_rewards, mask=response_mask)

token_level_rewards = token_level_rewards * response_mask  # 应用mask

# 5.2 根据adv_estimator选择算法
if adv_estimator == "gae":
    # 🔑 GAE算法
    values = batch.batch["values"].float()  # [10, 2047] - 从critic获得
    values = values * response_mask

    advantages, returns = compute_gae_advantage_return(
        token_level_rewards=token_level_rewards,  # [10, 2047]
        values=values,                             # [10, 2047]
        gamma=gamma,                               # 0.99
        lambd=lambd                                # 0.95
    )
    # functionals.py:396-432
    # 逐token计算GAE:
    # δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
    # A_t = δ_t + γ*λ*A_{t+1}

elif adv_estimator in ["reinforce", "grpo", "gigpo"]:
    # REINFORCE算法 (Monte Carlo returns)
    advantages, returns = compute_reinforce_return(
        token_level_rewards=token_level_rewards,  # [10, 2047]
        gamma=gamma,                               # 0.99
        lambd=lambd                                # (未使用)
    )
    # functionals.py:382-393
    # G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...

# 5.3 Whiten advantages (可选)
if whiten_advantages:
    advantages = masked_whiten(values=advantages, mask=response_mask)

advantages = advantages * response_mask

# 5.4 Clip advantages (可选)
if advantage_clip is not None:
    advantages = torch.clamp(advantages, min=-advantage_clip, max=advantage_clip)

batch.batch["advantages"] = advantages  # [10, 2047]
batch.batch["returns"] = returns        # [10, 2047]
```

### 2.2 GAE的详细计算

**公式** (`functionals.py:396-432`):

```python
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,  # [bs, seq_len]
    values: torch.Tensor,                # [bs, seq_len]
    gamma: float,                        # 0.99
    lambd: float                         # 0.95
):
    """
    GAE公式:
    δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
    A_t = δ_t + γ*λ*A_{t+1}

    Returns = Advantages + Values
    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]  # 2047

        for t in reversed(range(gen_len)):
            # 下一个token的value
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0

            # TD error
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]

            # GAE递归
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values

    return advantages, returns
```

**具体示例** (单个step, seq_len=5):

```python
# 输入
token_level_rewards = [0, 0, 0, 0, 5.0]  # 只有EOS有reward
values = [2.0, 2.1, 2.2, 2.3, 2.4]       # critic估计的values
gamma = 0.99
lambd = 0.95

# 反向计算
# t=4 (EOS):
nextvalues = 0.0
delta_4 = 5.0 + 0.99*0.0 - 2.4 = 2.6
A_4 = 2.6 + 0.99*0.95*0 = 2.6

# t=3:
nextvalues = 2.4
delta_3 = 0.0 + 0.99*2.4 - 2.3 = 0.076
A_3 = 0.076 + 0.99*0.95*2.6 = 2.52425

# t=2:
nextvalues = 2.3
delta_2 = 0.0 + 0.99*2.3 - 2.2 = 0.077
A_2 = 0.077 + 0.99*0.95*2.52425 = 2.44448

# t=1:
nextvalues = 2.2
delta_1 = 0.0 + 0.99*2.2 - 2.1 = 0.078
A_1 = 0.078 + 0.99*0.95*2.44448 = 2.37509

# t=0:
nextvalues = 2.1
delta_0 = 0.0 + 0.99*2.1 - 2.0 = 0.079
A_0 = 0.079 + 0.99*0.95*2.37509 = 2.31509

# 输出
advantages = [2.315, 2.375, 2.444, 2.524, 2.6]
returns = advantages + values = [4.315, 4.475, 4.644, 4.824, 5.0]
```

**关键特点**:
- GAE通过λ参数平衡bias和variance
- λ=0: 单步TD (高bias, 低variance)
- λ=1: Monte Carlo (低bias, 高variance)
- λ=0.95: 折中 (常用)

### 2.3 REINFORCE的计算

**公式** (`functionals.py:382-393`):

```python
def compute_reinforce_return(
    token_level_rewards: torch.Tensor,  # [bs, seq_len]
    gamma: float,
    lambd: float  # 未使用
):
    """
    REINFORCE (Monte Carlo):
    G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
    """
    with torch.no_grad():
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]
        cumulative_reward = 0

        for t in reversed(range(gen_len)):
            local_reward = token_level_rewards[:, t]
            cumulative_reward = local_reward + gamma * cumulative_reward
            advantages_reversed.append(cumulative_reward)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages  # 对于REINFORCE, advantages == returns

    return advantages, returns
```

**示例** (同样的输入):

```python
token_level_rewards = [0, 0, 0, 0, 5.0]
gamma = 0.99

# 反向计算
# t=4:
cumulative_reward = 5.0 + 0.99*0 = 5.0

# t=3:
cumulative_reward = 0.0 + 0.99*5.0 = 4.95

# t=2:
cumulative_reward = 0.0 + 0.99*4.95 = 4.9005

# t=1:
cumulative_reward = 0.0 + 0.99*4.9005 = 4.851495

# t=0:
cumulative_reward = 0.0 + 0.99*4.851495 = 4.80297951

# 输出
advantages = [4.803, 4.851, 4.901, 4.95, 5.0]
returns = advantages  # 相同
```

**对比**:
- GAE依赖critic的value估计,更stable
- REINFORCE只用actual returns,variance更高但无bias

---

## 3. 有Replay Buffer的GAE计算

### 3.1 完整数据流

```python
# === Step 1: Fresh Rollout ===
fresh_batch = self.train_rollout_scheduler.get_batch(...)

# === Step 2: Compute discounted returns (GIGPO) ===
fresh_batch = compute_discounted_returns(fresh_batch, ...)

# === Step 3: 🔑 Replay Buffer Integration ===
# agentic_pipeline.py:784-870
batch = self.integrate_replay_buffer_data(fresh_batch, global_step)

# integrate_replay_buffer_data内部流程:
def integrate_replay_buffer_data(fresh_batch, global_step):
    # 3.1 Push fresh data到buffer
    self.replay_buffer.push_from_dataproto(fresh_batch, global_step)
    # - 所有steps存入buffer
    # - Episode Index更新
    # - PER priorities初始化

    # 3.2 Sample from buffer (Echo模式)
    replay_batch, sampled_indices = self.replay_buffer.sample_for_training(
        batch_size=fresh_batch.batch_size[0],  # 相同大小
        device='cpu',
        tokenizer=self.tokenizer,
        sequence_length=self.pipeline_config.sequence_length,
        sampling_mode='step',  # step-level sampling
    )

    # 3.3 🔑 N-Step Returns计算 (如果启用)
    # step_buffer.py:252-468
    if self.enable_nstep:
        nstep_returns, completeness_mask = self.compute_nstep_returns(
            sampled_indices=sampled_indices,
            n_step=self.n_step,  # 例如: 5
            gamma=self.gamma,    # 0.99
            bootstrap_values=None
        )

        replay_batch.batch["nstep_returns"] = torch.from_numpy(nstep_returns)
        replay_batch.batch["nstep_completeness"] = torch.from_numpy(completeness_mask)

    # 3.4 Metadata
    replay_batch.meta_info["through_route"] = True  # 标记来自replay
    replay_batch.meta_info["sampled_indices"] = sampled_indices

    return replay_batch

# === Step 4: Compute response_level_rewards ===
# ⚠️ 注意: 对sampled batch重新计算!
batch = compute_response_level_rewards(batch, pipeline_config)

# === Step 5: Compute ref log probs ===
ref_log_probs = self.reference.compute_log_probs(batch)
batch = batch.union(ref_log_probs)

# === Step 6: Compute old log probs ===
# ⚠️ Replay buffer已存储old_log_probs
if "old_log_probs" not in batch.batch:
    # Fallback: 从actor_train计算
    old_log_probs = self.actor_train.compute_log_probs(batch)
    batch.batch["old_log_probs"] = old_log_probs

# === Step 7: Apply KL penalty ===
batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl, ...)
# 生成token_level_rewards: [batch_size, seq_len]

# === Step 8: Compute advantages (GAE) ===
batch = compute_advantage(
    data=batch,
    gamma=gamma,
    lambd=lambd,
    adv_estimator=adv_estimator,
    ...
)
# ⚠️ 逻辑与无replay buffer完全相同!
```

### 3.2 关键区别

| 步骤 | 无Replay Buffer | 有Replay Buffer |
|------|----------------|----------------|
| **数据来源** | Fresh rollout | Buffer sample |
| **Episode完整性** | 完整episode | 可能不完整 |
| **N-Step Returns** | ❌ | ✅ (可选,在sample时计算) |
| **old_log_probs** | 实时计算 | 从buffer读取 |
| **GAE计算** | 标准GAE | **标准GAE (相同!)** |
| **token_level_rewards** | 当前计算 | 当前计算 |

**重要**: GAE的`compute_advantage`函数**不关心**数据来自哪里,只要有:
- `token_level_rewards`: token-level奖励
- `values`: critic估计的values (仅GAE需要)
- `response_mask`: 有效token的mask

### 3.3 N-Step Returns的角色

**Question**: N-Step Returns和GAE有什么关系?

**Answer**: **没有直接关系,它们是两个独立的特性**。

**N-Step Returns** (`step_buffer.py:749-840`):
```python
# 在sample时计算,存储在batch中
nstep_returns = compute_nstep_returns(sampled_indices, n_step=5, gamma=0.99)
batch.batch["nstep_returns"] = nstep_returns  # [batch_size]
```

**GAE** (`functionals.py:396-432`):
```python
# 在compute_advantage中计算
advantages, returns = compute_gae_advantage_return(
    token_level_rewards=token_level_rewards,  # [batch_size, seq_len]
    values=values,                             # [batch_size, seq_len]
    gamma=gamma,
    lambd=lambd
)
```

**两者的区别**:

| 特性 | N-Step Returns | GAE |
|------|---------------|-----|
| **计算时机** | Buffer sample时 | Advantage computation时 |
| **维度** | Step-level scalar | Token-level sequence |
| **输入** | Episode内连续steps的rewards | Token-level rewards + values |
| **公式** | R_t^(n) = Σ(γ^i * r_{t+i}) | A_t = Σ((γλ)^i * δ_{t+i}) |
| **依赖critic** | ❌ | ✅ (需要values) |
| **用途** | 多步bootstrap | Advantage estimation |

**潜在的集成方式** (当前未实现):

```python
# 方案1: N-Step Returns作为GAE的bootstrap target
if "nstep_returns" in batch.batch:
    # 用n-step returns替代单步TD target
    target_values = batch.batch["nstep_returns"]  # [batch_size]
    # 需要expand到token-level
    # target_values_expanded = expand_to_token_level(target_values)
    # 然后用于GAE计算...
else:
    # 标准GAE
    advantages, returns = compute_gae_advantage_return(...)

# 方案2: N-Step Returns直接作为advantages (跳过GAE)
if "nstep_returns" in batch.batch:
    # 直接使用n-step returns
    advantages = batch.batch["nstep_returns"] - values
    returns = batch.batch["nstep_returns"]
else:
    # 标准GAE
    advantages, returns = compute_gae_advantage_return(...)
```

**当前实现**: N-Step Returns存在于batch中但**未被使用**,仅用于监控和debug。

---

## 4. Replay Buffer额外训练步的处理

### 4.1 完整流程

```python
# agentic_pipeline.py:453-598
if self.pipeline_config.replay.enabled:
    rb_cfg = self.pipeline_config.replay

    # Check if buffer ready
    if self.replay_buffer.can_sample(batch_size=training_batch_size):

        # 执行多次replay training
        for step_idx in range(rb_cfg.train_steps_per_env_step):
            # === Step 1: Sample from buffer ===
            mb, sampled_indices = self.replay_buffer.sample_for_training(
                batch_size=training_batch_size,
                device='cpu',
                tokenizer=self.tokenizer,
                sequence_length=self.pipeline_config.sequence_length,
                sampling_mode='step',
            )

            # === Step 2: Compute ref log probs ===
            ref_log_probs = self.reference.compute_log_probs(mb)
            mb = mb.union(ref_log_probs)

            # === Step 3: Compute old log probs (fallback) ===
            if "old_log_probs" not in mb.batch:
                behavior_old = self.actor_train.compute_log_probs(mb)
                mb.batch["old_log_probs"] = behavior_old.batch["log_probs"]

            # === Step 4: 🔑 Compute discounted returns (GIGPO) ===
            mb = compute_discounted_returns(
                mb,
                self.pipeline_config.adv_estimator,
                self.pipeline_config.step_reward_gamma
            )

            # === Step 5: 🔑 Compute response_level_rewards ===
            mb = compute_response_level_rewards(
                batch=mb,
                pipeline_config=self.pipeline_config
            )

            # === Step 6: 🔑 Apply KL penalty ===
            mb, _ = apply_kl_penalty(
                data=mb,
                kl_ctrl=self.kl_ctrl,
                kl_penalty=self.pipeline_config.kl_penalty
            )

            # === Step 7: 🔑 Compute advantages (GAE) ===
            mb = compute_advantage(
                data=mb,
                gamma=self.pipeline_config.gamma,
                lambd=self.pipeline_config.lambd,
                adv_estimator=self.pipeline_config.adv_estimator,
                advantage_clip=self.pipeline_config.advantage_clip,
                whiten_advantages=self.pipeline_config.whiten_advantages,
                whiten_rewards=self.pipeline_config.whiten_rewards,
            )

            # === Step 8: Training ===
            if self.pipeline_config.adv_estimator == "gae":
                self.critic.train_step(mb)
            self.actor_train.train_step(mb)
```

**关键点**:
1. **每次sample都重新计算**: discounted returns → response_level_rewards → kl_penalty → advantages
2. **不复用之前的advantages**: 因为actor已经更新,old policy改变
3. **critic values是最新的**: 每次training后critic更新,下次sample会用新的values

### 4.2 Off-Policy考虑

**问题**: Replay buffer中的数据是old policy产生的,当前policy已经更新多步,如何处理?

**PPO的解决方案** (Importance Sampling):

```python
# 在loss计算中
ratio = torch.exp(new_log_probs - old_log_probs)  # π_new/π_old
clipped_ratio = torch.clamp(ratio, 1-clip, 1+clip)
loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
```

**ROLL的实现**:
- `old_log_probs`: 从buffer读取 (behavior policy)
- `new_log_probs`: 实时计算 (current policy)
- `ratio`: 在actor training中计算
- `pg_clip`: PPO clip参数 (例如: 0.2)

**Off-Policy监控** (`agentic_pipeline.py:528-548`):

```python
if self.pipeline_config.offpolicy_monitor.enabled:
    replay_offpolicy_metrics = compute_offpolicy_metrics(
        current_batch=mb,
        actor_train_cluster=self.actor_train,
        old_prob_mode=self.pipeline_config.offpolicy_monitor.behavior_scope,
        pg_clip=self.pipeline_config.pg_clip
    )
    # 计算:
    # - offpolicy/kl_divergence
    # - offpolicy/importance_ratio
    # - offpolicy/clipped_ratio
```

---

## 5. StepReplayBuffer中GAE的简化实现

### 5.1 Buffer中的GAE方法

**代码**: `step_buffer.py:842-920`

```python
def compute_gae(
    self,
    sampled_indices: List[int],
    values: Optional[np.ndarray] = None,
    gamma: Optional[float] = None,
    gae_lambda: Optional[float] = None,
    gae_horizon: Optional[int] = None
) -> np.ndarray:
    """
    ⚠️ 简化实现: 仅供参考,不推荐在生产环境使用

    当前限制:
    1. 只使用第一个step的value,其他为0
    2. 不支持multi-step value estimation
    3. 需要Pipeline提供完整的values array
    """
    gamma = gamma or self.gamma
    gae_lambda = gae_lambda or 0.95
    gae_horizon = gae_horizon or 20

    batch_size = len(sampled_indices)
    advantages = np.zeros(batch_size, dtype=np.float32)
    buffer_list = list(self.steps)

    for i, start_idx in enumerate(sampled_indices):
        # 获取连续的n步
        indices, _ = self.get_nstep_indices(start_idx, min(gae_horizon, self.n_step))

        if not indices:
            continue

        # ⚠️ 简化: 只用第一个value
        v_t = values[i] if values is not None and len(values) > i else 0.0

        # GAE计算
        gae = 0.0
        for j, idx in enumerate(indices):
            step_entry = buffer_list[idx]
            response_mask_bool = step_entry.response_mask.astype(bool)
            reward = float(step_entry.scores[response_mask_bool].sum())

            # ⚠️ 简化: v_next始终为0
            v_next = 0.0

            # TD error
            delta = reward + gamma * v_next - (v_t if j == 0 else 0.0)

            # GAE递归
            gae = delta + gamma * gae_lambda * gae

        advantages[i] = gae

    return advantages
```

**为什么是简化版?**

1. **Values维度不匹配**:
   - GAE需要: `values[batch_size, seq_len]` - 每个token的value
   - Buffer提供: `values[batch_size]` - 只有一个scalar

2. **正确的实现需要**:
   ```python
   # 在Pipeline中:
   # 1. 提取所有sampled steps的states
   all_indices = buffer.build_stacked_indices(sampled_indices, n_step)
   all_states = extract_states_from_buffer(all_indices)

   # 2. 批量计算values
   all_values = critic.compute_values(all_states)  # [batch*n_step, seq_len]

   # 3. Reshape并传递给buffer
   values_per_sample = all_values.reshape(batch_size, n_step, seq_len)
   advantages = buffer.compute_gae(sampled_indices, values=values_per_sample, ...)
   ```

3. **当前状态**:
   - Buffer中的GAE是**占位实现**
   - 实际使用的是`functionals.compute_advantage`
   - 在Pipeline层面计算,有完整的token-level values

---

## 6. 不同Adv Estimator的对比

### 6.1 四种方法总结

| Method | 公式 | 依赖Critic | Bias | Variance | 适用场景 |
|--------|------|-----------|------|----------|---------|
| **reinforce** | G_t = Σγ^i*r_{t+i} | ❌ | Low | High | 简单任务,episode短 |
| **gae** | A_t = Σ(γλ)^i*δ_{t+i} | ✅ | Medium | Medium | 通用,需要stable training |
| **grpo** | G_t = Σγ^i*r_{t+i} | ❌ | Low | High | Group-normalized rewards |
| **gigpo** | A_t = α*R_ep + β*R_step | ❌ | Low | Medium | 分层奖励,multi-step tasks |

### 6.2 计算成本对比

**假设**: batch_size=128, seq_len=2048

| Method | Critic Forward | 额外计算 | 总耗时 (相对) |
|--------|---------------|---------|--------------|
| **reinforce** | ❌ | Monte Carlo sum | 1x |
| **gae** | ✅ | GAE recursion | 3x |
| **grpo** | ❌ | Group norm | 1.2x |
| **gigpo** | ❌ | Discounted returns + group norm | 1.5x |

### 6.3 代码路径对比

```python
# reinforce/grpo
compute_advantage(..., adv_estimator="reinforce"):
    advantages, returns = compute_reinforce_return(
        token_level_rewards=token_level_rewards,
        gamma=gamma,
        lambd=lambd
    )

# gae
compute_advantage(..., adv_estimator="gae"):
    values = batch.batch["values"]  # 从critic获得
    advantages, returns = compute_gae_advantage_return(
        token_level_rewards=token_level_rewards,
        values=values,
        gamma=gamma,
        lambd=lambd
    )

# gigpo
# Step 1: compute_discounted_returns
compute_discounted_returns(batch, adv_estimator="gigpo", gamma=0.99):
    # 按trajectory计算discounted step rewards
    batch.batch["step_rewards"] = ...

# Step 2: compute_response_level_rewards
compute_response_level_rewards(batch, pipeline_config):
    episode_rewards = grouped_reward_norm(episode_scores, ...)
    step_rewards = grouped_reward_norm(batch.batch["step_rewards"], ...)
    response_level_rewards = α*episode_rewards + β*step_rewards

# Step 3: compute_advantage
compute_advantage(..., adv_estimator="gigpo"):
    # 使用组合的response_level_rewards
    advantages, returns = compute_reinforce_return(
        token_level_rewards=token_level_rewards,  # 基于组合rewards
        gamma=gamma,
        lambd=lambd
    )
```

---

## 7. 总结和建议

### 7.1 关键发现

1. **GAE计算与Replay Buffer解耦**:
   - GAE在`compute_advantage`中统一计算
   - 不关心数据来自fresh rollout还是buffer
   - Buffer只负责数据存储和采样

2. **N-Step Returns与GAE独立**:
   - N-Step Returns: Step-level多步bootstrap
   - GAE: Token-level advantage estimation
   - 当前未集成,可作为future work

3. **Token-level vs Step-level**:
   - **Rewards**: Step-level scalar → Expand to token-level
   - **GAE**: Token-level计算,逐token递归
   - **Training**: Token-level loss,但masked

4. **Off-Policy处理**:
   - PPO的importance sampling ratio处理policy shift
   - Replay buffer存储old_log_probs作为behavior policy
   - 每次sample重新计算advantages (不缓存)

### 7.2 优化建议

#### 建议1: 集成N-Step Returns和GAE

```python
# 在compute_advantage中:
if "nstep_returns" in batch.batch and enable_nstep_gae:
    # 用n-step returns作为bootstrap
    # 需要实现: expand_nstep_to_token_level
    nstep_values = expand_nstep_to_token_level(
        batch.batch["nstep_returns"],
        batch.batch["response_mask"]
    )

    # 修改GAE公式,用n-step target
    advantages, returns = compute_gae_with_nstep(
        token_level_rewards=token_level_rewards,
        values=values,
        nstep_values=nstep_values,
        gamma=gamma,
        lambd=lambd,
        n_step=n_step
    )
```

#### 建议2: 缓存Advantages (可选)

**问题**: 每次replay sample都重新计算advantages,成本高

**方案**: 在push时预计算并缓存

```python
# 在StepReplayBuffer.push_from_dataproto中:
if self.cache_advantages:
    # 预计算advantages (需要critic)
    advantages = self.critic.compute_advantages(batch)

    # 存储到StepEntry
    for i, step_entry in enumerate(batch):
        step_entry.cached_advantages = advantages[i]
```

**权衡**:
- ✅ 减少重复计算
- ❌ 增加存储开销
- ❌ Advantages会过时 (critic更新后)

**建议**: 不缓存,保持当前设计 (实时计算)

#### 建议3: 监控Off-Policy程度

```yaml
# 配置
offpolicy_monitor:
  enabled: true
  monitor_interval: 10
  monitor_replay_batch: true
  kl_threshold: 0.1  # 超过则warning

# 监控指标
offpolicy/kl_divergence: 0.05  # behavior vs current policy
offpolicy/importance_ratio: 1.12  # exp(KL)
offpolicy/clipped_ratio: 0.85  # 被PPO clip的比例
```

#### 建议4: 完善Buffer中的GAE实现

**当前问题**: `step_buffer.compute_gae`是简化版

**完整实现需要**:
1. Pipeline提供完整的`values[batch_size, n_step, seq_len]`
2. Buffer内实现完整的token-level GAE
3. 支持multi-step value bootstrapping

**是否值得?** ❌
- 当前Pipeline层面的GAE已经工作良好
- 在Buffer内实现会增加复杂度
- 好处有限 (只是移动计算位置)

**建议**: 保持当前设计,在Pipeline计算GAE

### 7.3 配置建议

**基础配置** (推荐):
```yaml
# Advantage estimation
adv_estimator: gae
gamma: 0.99
lambd: 0.95
advantage_clip: 10.0
whiten_advantages: true
whiten_rewards: false

# Replay buffer
replay:
  enabled: true
  capacity: 100000
  enable_nstep: false  # 暂不启用,待集成

# Critic
critic_warmup: 100  # warm up critic before actor training
```

**高级配置** (GIGPO):
```yaml
adv_estimator: gigpo
step_reward_gamma: 0.99
episode_reward_weight: 0.5
step_reward_weight: 0.5
reward_normalization:
  grouping: traj_id
  method: mean_std
```

### 7.4 调试技巧

**验证GAE计算**:
```python
# 在Pipeline中添加debug logging
logger.debug(f"token_level_rewards: {token_level_rewards.mean():.4f}")
logger.debug(f"values: {values.mean():.4f}")
logger.debug(f"advantages: {advantages.mean():.4f}, std: {advantages.std():.4f}")
logger.debug(f"returns: {returns.mean():.4f}")

# 检查response_mask
logger.debug(f"response_mask ratio: {response_mask.float().mean():.2%}")

# 监控off-policy
logger.debug(f"kl_divergence: {kl_divergence:.4f}")
logger.debug(f"importance_ratio: {importance_ratio.mean():.4f}")
```

**单元测试**:
```python
def test_gae_computation():
    # 构造简单case
    token_level_rewards = torch.tensor([[0, 0, 5.0]])
    values = torch.tensor([[2.0, 2.1, 2.4]])
    gamma = 0.99
    lambd = 0.95

    advantages, returns = compute_gae_advantage_return(
        token_level_rewards, values, gamma, lambd
    )

    # 验证
    assert advantages.shape == (1, 3)
    assert returns.shape == (1, 3)
    # 手算expected values并比较
```

---

**文档版本**: 1.0.0
**最后更新**: 2025-10-30
**文档类型**: 完整技术分析
**面向读者**: ROLL框架开发者和用户
