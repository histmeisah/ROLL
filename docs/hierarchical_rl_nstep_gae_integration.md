# 分层强化学习视角：N-Step Returns与Token-Level GAE的正确融合

## 1. 问题的本质：两层RL结构

### 1.1 当前架构的分层特性

ROLL框架实际上是一个**两层强化学习系统**：

```
外层 (Episode/Step Level):
┌─────────────────────────────────────────────────┐
│  Agent与环境交互                                  │
│  State_t → Action (LLM Response) → State_{t+1}   │
│  获得 Step Reward: r_t                           │
│  Episode = [Step_0, Step_1, ..., Step_T]        │
└─────────────────────────────────────────────────┘
                    ↓
内层 (Token Level):
┌─────────────────────────────────────────────────┐
│  每个Step的Action生成过程                         │
│  Token_1, Token_2, ..., Token_N                 │
│  Token-level policy gradient                    │
│  最后一个token (EOS) 获得整个step的reward        │
└─────────────────────────────────────────────────┘
```

### 1.2 当前实现的问题

**问题1: N-Step Returns被计算但未使用**

```python
# step_buffer.py:sample_for_training
if self.enable_nstep:
    nstep_returns, completeness_mask = self.compute_nstep_returns(
        sampled_indices=sampled_indices,
        n_step=self.n_step,  # 外层的step
        gamma=self.gamma,
        bootstrap_values=None
    )
    # ✅ 计算了
    batch.batch["nstep_returns"] = nstep_returns  # [batch_size] - Step-level

# functionals.py:compute_advantage
# ❌ 但从未使用！GAE只用token_level_rewards
advantages, returns = compute_gae_advantage_return(
    token_level_rewards=token_level_rewards,  # [batch_size, seq_len] - Token-level
    values=values,
    gamma=gamma,  # 这个gamma用于token-level折扣！
    lambd=lambd
)
```

**问题2: 两个Gamma混淆**

```yaml
# 配置中的gamma
gamma: 0.99  # 用于什么？Token-level还是Step-level?

# Replay buffer中的gamma
replay:
  nstep_gamma: 0.99  # 用于Step-level n-step returns

# GIGPO中的gamma
step_reward_gamma: 1.0  # 用于Step-level discounted returns
```

**问题3: Token-level和Step-level的奖励没有正确连接**

当前流程：
```
Step Rewards → expand_to_token_level (放到EOS位置)
             ↓
        Token-level GAE (用token gamma折扣)
```

缺失的连接：
```
多个Steps的累积奖励 (N-Step Returns)
         ↓
    如何影响Token-level的advantage?
```

---

## 2. 正确的分层RL formulation

### 2.1 数学框架

**外层MDP (Step-level)**:
- State: s_t (环境状态)
- Action: a_t (整个LLM response)
- Reward: r_t (step reward)
- Transition: s_{t+1} = f(s_t, a_t)

**内层POMDP (Token-level)**:
- State: 隐式 (LLM内部状态)
- Action: token_i (第i个token)
- Reward: 0 for i < N, r_t for i = N (EOS)
- Policy: π_θ(token_i | token_{<i}, s_t)

**N-Step Returns (外层)**:
```
R_t^(n) = r_t + γ_step * r_{t+1} + ... + γ_step^{n-1} * r_{t+n-1} + γ_step^n * V(s_{t+n})
```

**Token-level Returns (内层)**:
```
G_i = 0 + γ_token * 0 + ... + γ_token^{N-i} * r_t
    = γ_token^{N-i} * r_t
```

**关键洞察**: r_t应该用R_t^(n)替代！

### 2.2 正确的融合方式

```
Step 1: 计算N-Step Returns (外层)
R_t^(n) = r_t + γ_step * r_{t+1} + ... + γ_step^{n-1} * r_{t+n-1}

Step 2: 将N-Step Returns作为Token-level的最终奖励
token_level_rewards[EOS位置] = R_t^(n)  # 而不是单步r_t

Step 3: Token-level GAE使用γ_token
δ_i = token_reward_i + γ_token * V(token_{i+1}) - V(token_i)
A_i = Σ (γ_token * λ)^k * δ_{i+k}
```

---

## 3. 实现方案

### 3.1 方案A: 分层独立计算 (推荐，正确)

**核心思想**:
- 内层token-level GAE保持不变 (用单步reward)
- 外层step-level用n-step returns计算额外的step advantage
- 两者相加得到最终的token-level advantage

**关键**: **不修改token_level_rewards**，在advantage层面进行组合。

#### 实现步骤

##### Step 1: 计算Step-level Advantage

```python
# 新增函数: functionals.py
def compute_step_level_advantage_simple(
    data: "DataProto",
    gamma_step: float = 0.99,
) -> torch.Tensor:
    """
    Compute step-level advantage using n-step returns.

    This provides long-term credit assignment at the step level,
    without affecting token-level GAE.

    Args:
        data: DataProto with "nstep_returns" and "response_level_rewards"
        gamma_step: Step-level discount factor

    Returns:
        step_advantages: [batch_size] - Step-level advantage for each sample
    """
    if "nstep_returns" not in data.batch:
        # No n-step returns available, return zeros
        batch_size = data.batch.batch_size[0]
        return torch.zeros(batch_size, device=data.batch.device)

    # N-step returns: R_t^(n) = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1}
    nstep_returns = data.batch["nstep_returns"].clone().detach()

    # Single-step rewards (baseline)
    single_step_rewards = data.batch["response_level_rewards"].clone().detach()

    # Step-level advantage: A_step = R_t^(n) - r_t
    # 这表示"使用多步信息比单步好多少"
    step_advantages = nstep_returns - single_step_rewards

    # 可选: 归一化step advantages
    if step_advantages.numel() > 1:
        step_advantages = (step_advantages - step_advantages.mean()) / (step_advantages.std() + 1e-8)

    return step_advantages


def expand_step_advantage_to_token_level(
    step_advantages: torch.Tensor,
    data: "DataProto"
) -> torch.Tensor:
    """
    Expand step-level advantages to token-level by placing at EOS position.

    Args:
        step_advantages: [batch_size] - Step-level advantage
        data: DataProto with attention_mask and position_ids

    Returns:
        token_level_step_advantages: [batch_size, seq_len-1]
    """
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    position_ids = data.batch["position_ids"]

    if position_ids.dim() == 3:
        position_ids = position_ids[:, 0]

    # Find EOS positions
    eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)

    # Create token-level tensor
    token_level_step_adv = torch.zeros_like(attention_mask, dtype=step_advantages.dtype)
    token_level_step_adv[torch.arange(batch_size), eos_mask_idx] = step_advantages
    token_level_step_adv = token_level_step_adv[:, 1:]  # Remove first token

    return token_level_step_adv
```

##### Step 2: 修改 `compute_advantage`

```python
# 修改: functionals.py:compute_advantage
@torch.no_grad()
def compute_advantage(
    data: "DataProto",
    gamma,
    lambd,
    adv_estimator,
    advantage_clip=None,
    whiten_advantages=False,
    whiten_rewards=False,
    response_mask=None,
    use_nstep_advantage=False,  # 🔑 新增参数
    nstep_advantage_weight=0.5,  # 🔑 新增参数
    nstep_gamma=0.99,  # 🔑 新增参数
):
    """
    Compute advantages with optional step-level n-step advantage integration.

    When use_nstep_advantage=True:
    1. Compute token-level advantages (inner layer) - unchanged
    2. Compute step-level advantages (outer layer) using n-step returns
    3. Combine: final_adv = token_adv + weight * step_adv

    This preserves the independence of inner and outer layers.
    """
    if response_mask is None:
        response_mask = data.batch["response_mask"][:, 1:]

    if response_mask.sum() == 0:
        whiten_rewards = False
        whiten_advantages = False
        logger.info("Warning: response_mask.sum() == 0! All masked_whiten will be skipped.")

    # === 内层: Token-level advantages (保持不变) ===
    token_level_rewards = data.batch["token_level_rewards"].float()

    if whiten_rewards:
        token_level_rewards = masked_whiten(values=token_level_rewards, mask=response_mask)

    token_level_rewards = token_level_rewards * response_mask
    data.batch["token_level_rewards"] = token_level_rewards

    # Compute token-level advantages
    if adv_estimator == "gae":
        values = data.batch["values"].float()
        data.batch["values"] = values * response_mask
        token_advantages, token_returns = compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            gamma=gamma,  # γ_token
            lambd=lambd
        )
    elif adv_estimator in ["reinforce", "grpo", "gigpo"]:
        token_advantages, token_returns = compute_reinforce_return(
            token_level_rewards=token_level_rewards,
            gamma=gamma,
            lambd=lambd
        )
    else:
        raise NotImplementedError

    # === 外层: Step-level advantages (新增) ===
    if use_nstep_advantage and "nstep_returns" in data.batch:
        # 计算step-level advantage
        step_advantages = compute_step_level_advantage_simple(
            data=data,
            gamma_step=nstep_gamma
        )  # [batch_size]

        # Expand到token-level
        token_level_step_adv = expand_step_advantage_to_token_level(
            step_advantages=step_advantages,
            data=data
        )  # [batch_size, seq_len-1]

        # 应用mask
        token_level_step_adv = token_level_step_adv * response_mask

        # 🔑 组合: 内层 + 外层
        combined_advantages = token_advantages + nstep_advantage_weight * token_level_step_adv

        # 存储用于监控
        data.batch["token_advantages"] = token_advantages  # 内层
        data.batch["step_advantages_expanded"] = token_level_step_adv  # 外层

        logger.debug(
            f"Hierarchical advantages: token_mean={token_advantages.mean():.4f}, "
            f"step_mean={token_level_step_adv.mean():.4f}, "
            f"combined_mean={combined_advantages.mean():.4f}"
        )
    else:
        # 没有n-step,只用token-level
        combined_advantages = token_advantages

    data.batch["raw_advantages"] = combined_advantages

    # Whiten combined advantages
    if whiten_advantages:
        combined_advantages = masked_whiten(values=combined_advantages, mask=response_mask)

    combined_advantages = combined_advantages * response_mask

    # Clip
    if advantage_clip is not None:
        adv_clip_frac = compute_clip_fraction(
            values=combined_advantages,
            clip_min=-advantage_clip,
            clip_max=advantage_clip
        )
        data.meta_info["metrics"] = {"critic/advantage_clip_frac": adv_clip_frac}
        combined_advantages = torch.clamp(
            combined_advantages,
            min=-advantage_clip,
            max=advantage_clip
        )

    data.batch["advantages"] = combined_advantages
    data.batch["returns"] = token_returns  # 保持token-level returns

    return data
```

#### 修改 `functionals.py:apply_kl_penalty`

```python
@torch.no_grad()
def apply_kl_penalty(data: "DataProto", kl_ctrl: AdaptiveKLController, kl_penalty="kl", use_nstep_returns: bool = False):
    response_mask = data.batch["response_mask"][:, 1:]

    # 🔑 使用修改后的expand函数
    token_level_rewards = expand_to_token_level(data, use_nstep_returns=use_nstep_returns)

    # ... (其余逻辑不变)
    if "ref_log_probs" in data.batch.keys():
        kld = compute_approx_kl(...)
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_rewards - beta * kld

    # ... (更新KL controller等)
    data.batch["token_level_rewards"] = token_level_rewards

    return data, metrics
```

#### 修改 `agentic_pipeline.py:run`

```python
# 在主循环中
for global_step in range(max_steps):
    # ... rollout, replay buffer integration ...

    # 🔑 传递use_nstep_returns标志
    use_nstep = (
        self.pipeline_config.replay.enabled and
        self.pipeline_config.replay.enable_nstep and
        "nstep_returns" in batch.batch
    )

    batch, kl_metrics = apply_kl_penalty(
        data=batch,
        kl_ctrl=self.kl_ctrl,
        kl_penalty=self.pipeline_config.kl_penalty,
        use_nstep_returns=use_nstep  # 🔑 新增参数
    )

    batch = compute_advantage(
        data=batch,
        gamma=self.pipeline_config.gamma,  # 这是γ_token
        lambd=self.pipeline_config.lambd,
        adv_estimator=self.pipeline_config.adv_estimator,
        # ...
    )
```

#### 配置修改

```yaml
# agentic_config.py
replay:
  enabled: true

  # Step-level n-step returns
  enable_nstep: true
  n_step: 5
  nstep_gamma: 0.99  # γ_step - 外层step之间的折扣
  use_nstep_in_advantage: true  # 🔑 新增: 是否在advantage计算中使用

# Token-level GAE
gamma: 0.995  # γ_token - 内层token之间的折扣 (通常接近1)
lambd: 0.95   # GAE λ参数
adv_estimator: gae
```

### 3.2 方案B: 混合使用 (高级，灵活)

**核心思想**: 既用N-Step Returns又用单步rewards，加权组合

#### 实现

```python
def expand_to_token_level_hybrid(
    data: "DataProto",
    nstep_weight: float = 0.5,
    single_step_weight: float = 0.5
):
    """
    Hybrid approach: combine n-step returns and single-step rewards

    Args:
        nstep_weight: Weight for n-step returns
        single_step_weight: Weight for single-step rewards
    """
    # Single-step rewards
    single_step_rewards = data.batch["response_level_rewards"].clone().detach()

    # N-step returns (if available)
    if "nstep_returns" in data.batch:
        nstep_returns = data.batch["nstep_returns"].clone().detach()

        # 加权组合
        combined_rewards = (
            nstep_weight * nstep_returns +
            single_step_weight * single_step_rewards
        )

        # Logging
        logger.debug(f"Hybrid rewards: {nstep_weight:.2f}*nstep + {single_step_weight:.2f}*single")
        logger.debug(f"  nstep_returns mean: {nstep_returns.mean():.4f}")
        logger.debug(f"  single_step mean: {single_step_rewards.mean():.4f}")
        logger.debug(f"  combined mean: {combined_rewards.mean():.4f}")
    else:
        combined_rewards = single_step_rewards

    # Expand to token-level
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    position_ids = data.batch["position_ids"]

    if position_ids.dim() == 3:
        position_ids = position_ids[:, 0]

    eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)
    token_level_rewards = torch.zeros_like(attention_mask, dtype=combined_rewards.dtype)
    token_level_rewards[torch.arange(batch_size), eos_mask_idx] = combined_rewards
    token_level_rewards = token_level_rewards[:, 1:]

    return token_level_rewards
```

#### 配置

```yaml
replay:
  enable_nstep: true
  n_step: 5
  nstep_gamma: 0.99

  # 混合模式配置
  nstep_usage_mode: hybrid  # "replace" / "hybrid" / "separate"
  nstep_weight: 0.7  # N-step的权重
  single_step_weight: 0.3  # 单步的权重
```

### 3.3 方案C: 双路GAE (最复杂，理论最优)

**核心思想**: 分别计算Step-level GAE和Token-level GAE，然后结合

#### Step-level GAE

```python
def compute_step_level_gae(
    batch: DataProto,
    replay_buffer: StepReplayBuffer,
    sampled_indices: List[int],
    critic: CriticWorker,
    gamma_step: float = 0.99,
    lambda_step: float = 0.95,
    n_step: int = 5
) -> torch.Tensor:
    """
    Compute GAE at step level using n-step sequences.

    Returns:
        step_level_advantages: [batch_size] - Advantage for each sampled step
    """
    batch_size = len(sampled_indices)
    advantages = torch.zeros(batch_size, dtype=torch.float32)

    # 1. 获取所有steps的values (需要critic forward)
    # 这需要提取step的state并用critic估计value
    step_values = torch.zeros(batch_size, dtype=torch.float32)
    for i, idx in enumerate(sampled_indices):
        step_entry = replay_buffer.steps[idx]
        # 用critic计算value (simplified - 实际需要完整的forward)
        state = extract_state(step_entry)  # 提取state
        step_values[i] = critic.compute_value(state).item()

    # 2. 对每个step计算GAE
    for i, start_idx in enumerate(sampled_indices):
        # 获取n-step sequence
        indices, complete = replay_buffer.get_nstep_indices(start_idx, n_step)

        if not indices:
            continue

        # 提取rewards
        rewards = []
        for idx in indices:
            step_entry = list(replay_buffer.steps)[idx]
            response_mask_bool = step_entry.response_mask.astype(bool)
            reward = float(step_entry.scores[response_mask_bool].sum())
            rewards.append(reward)

        # 提取next values (需要critic forward)
        next_values = torch.zeros(len(indices) + 1, dtype=torch.float32)
        next_values[0] = step_values[i]  # 当前step的value
        for j, idx in enumerate(indices[1:]):
            step_entry = list(replay_buffer.steps)[idx]
            state = extract_state(step_entry)
            next_values[j+1] = critic.compute_value(state).item()

        # Bootstrap (如果incomplete)
        if not complete and len(indices) < n_step:
            next_values[-1] = 0.0  # 或者用其他方式估计

        # GAE计算 (Step-level)
        gae = 0.0
        for j in range(len(rewards)):
            delta = rewards[j] + gamma_step * next_values[j+1] - next_values[j]
            gae = delta + gamma_step * lambda_step * gae

        advantages[i] = gae

    return advantages
```

#### Token-level GAE (保持不变)

```python
def compute_token_level_gae(
    token_level_rewards: torch.Tensor,
    token_values: torch.Tensor,
    gamma_token: float = 0.995,
    lambda_token: float = 0.95
) -> torch.Tensor:
    """
    Standard token-level GAE.
    """
    return compute_gae_advantage_return(
        token_level_rewards=token_level_rewards,
        values=token_values,
        gamma=gamma_token,
        lambd=lambda_token
    )
```

#### 结合两者

```python
def compute_hierarchical_advantage(
    batch: DataProto,
    replay_buffer: StepReplayBuffer,
    sampled_indices: List[int],
    critic: CriticWorker,
    pipeline_config: AgenticConfig
) -> DataProto:
    """
    Compute advantages using hierarchical approach:
    1. Step-level GAE for long-term credit
    2. Token-level GAE for within-step credit
    3. Combine them
    """
    # 1. Step-level GAE
    step_advantages = compute_step_level_gae(
        batch=batch,
        replay_buffer=replay_buffer,
        sampled_indices=sampled_indices,
        critic=critic,
        gamma_step=pipeline_config.replay.nstep_gamma,
        lambda_step=pipeline_config.replay.nstep_lambda,
        n_step=pipeline_config.replay.n_step
    )  # [batch_size]

    # 2. Expand step advantages to token-level
    step_advantages_expanded = expand_to_token_level(
        DataProto.from_dict({"response_level_rewards": step_advantages})
    )  # [batch_size, seq_len]

    # 3. Token-level GAE
    token_advantages = compute_token_level_gae(
        token_level_rewards=batch.batch["token_level_rewards"],
        token_values=batch.batch["values"],
        gamma_token=pipeline_config.gamma,
        lambda_token=pipeline_config.lambd
    )  # [batch_size, seq_len]

    # 4. 组合 (加权)
    alpha = pipeline_config.step_advantage_weight  # 例如: 0.5
    beta = pipeline_config.token_advantage_weight  # 例如: 0.5

    combined_advantages = alpha * step_advantages_expanded + beta * token_advantages

    # 5. 存储
    batch.batch["step_advantages"] = step_advantages  # 用于监控
    batch.batch["token_advantages"] = token_advantages  # 用于监控
    batch.batch["advantages"] = combined_advantages  # 用于训练

    return batch
```

---

## 4. 推荐方案对比

| 方案 | 实现复杂度 | 计算成本 | 理论正确性 | 灵活性 | 推荐度 |
|------|-----------|---------|-----------|--------|--------|
| **A: 直接替换** | ⭐ 简单 | ⭐ 低 (无额外成本) | ⭐⭐⭐ 较好 | ⭐⭐ 中等 | ⭐⭐⭐⭐⭐ |
| **B: 混合使用** | ⭐⭐ 中等 | ⭐ 低 | ⭐⭐⭐⭐ 好 | ⭐⭐⭐⭐ 高 | ⭐⭐⭐⭐ |
| **C: 双路GAE** | ⭐⭐⭐⭐ 复杂 | ⭐⭐⭐ 高 (需要额外critic forward) | ⭐⭐⭐⭐⭐ 最优 | ⭐⭐⭐⭐⭐ 最高 | ⭐⭐⭐ |

### 推荐策略

1. **立即实现**: 方案A (直接替换)
   - 修改最少 (~20行代码)
   - 无额外计算成本
   - 能够正确利用N-Step Returns
   - 足够满足大多数场景

2. **未来优化**: 方案B (混合使用)
   - 如果发现方案A效果不稳定
   - 可以通过超参数调整权重
   - 平滑过渡

3. **理论探索**: 方案C (双路GAE)
   - 研究项目可尝试
   - 需要大量工程实现
   - 计算成本显著增加

---

## 5. 实现细节和注意事项

### 5.1 Gamma参数的区分

**必须区分两个gamma**:

```yaml
# Token-level折扣 (内层)
gamma: 0.995  # 接近1，因为token之间时间跨度小
lambd: 0.95

# Step-level折扣 (外层)
replay:
  nstep_gamma: 0.99  # 标准RL折扣，环境交互
  nstep_lambda: 0.95  # (如果实现方案C)
```

**为什么Token-level gamma应该接近1?**

```
假设一个response有100个tokens，时间跨度约1秒
如果用γ=0.99:
  第1个token的折扣: 0.99^99 ≈ 0.37 (衰减太快！)

如果用γ=0.995:
  第1个token的折扣: 0.995^99 ≈ 0.61 (更合理)

如果用γ=0.999:
  第1个token的折扣: 0.999^99 ≈ 0.91 (可能更好)
```

**推荐配置**:
```python
# 根据平均response长度调整
avg_response_length = 100
target_discount_ratio = 0.5  # 期望第1个token获得最后token奖励的50%

gamma_token = target_discount_ratio ** (1 / avg_response_length)
# gamma_token ≈ 0.993
```

### 5.2 Completeness处理

**问题**: N-Step Returns可能不完整 (episode被截断)

```python
# step_buffer.py:compute_nstep_returns返回:
nstep_returns: [batch_size]
completeness_mask: [batch_size] - bool array

# 在使用时需要考虑
if use_nstep_returns:
    if "nstep_completeness" in batch.batch:
        completeness = batch.batch["nstep_completeness"]  # [batch_size]
        complete_ratio = completeness.float().mean()

        if complete_ratio < 0.5:
            logger.warning(
                f"Low n-step completeness: {complete_ratio:.2%}. "
                f"Consider reducing n_step or increasing buffer capacity."
            )

        # 策略1: 只对complete samples使用n-step
        use_nstep_mask = completeness.bool()
        response_level_rewards = torch.where(
            use_nstep_mask,
            batch.batch["nstep_returns"],
            batch.batch["response_level_rewards"]  # fallback
        )

        # 策略2: 对incomplete samples降权
        weights = completeness.float() * 1.0 + 0.5  # complete: 1.0, incomplete: 0.5
        batch.batch["sample_weights"] = weights
```

### 5.3 GIGPO的特殊处理

**问题**: GIGPO已经有step-level discounted returns

```python
# GIGPO的流程:
# 1. compute_discounted_returns -> batch.batch["step_rewards"]
# 2. compute_response_level_rewards -> 组合episode + step rewards

# 与N-Step Returns的关系?
# - compute_discounted_returns: 单个episode内的Monte Carlo returns
# - N-Step Returns: 跨多个steps的bootstrap

# 建议: GIGPO不使用N-Step Returns
if adv_estimator == "gigpo":
    use_nstep_returns = False
    logger.info("GIGPO uses its own discounted returns, skipping n-step")
```

### 5.4 监控指标

**必须添加的metrics**:

```python
# 在apply_kl_penalty中
metrics = {
    "critic/kl": current_kl,
    "critic/kl_coef": beta,
}

if use_nstep_returns:
    metrics.update({
        "nstep/enabled": 1.0,
        "nstep/n_step": self.pipeline_config.replay.n_step,
        "nstep/completeness_ratio": batch.batch["nstep_completeness"].float().mean().item(),
        "nstep/returns_mean": batch.batch["nstep_returns"].mean().item(),
        "nstep/returns_std": batch.batch["nstep_returns"].std().item(),
        "nstep/single_step_mean": batch.batch["response_level_rewards"].mean().item(),
        "nstep/returns_vs_single_ratio": (
            batch.batch["nstep_returns"].mean() /
            (batch.batch["response_level_rewards"].mean() + 1e-8)
        ).item(),
    })
```

---

## 6. 完整实现示例 (方案A)

### 6.1 修改 `functionals.py`

```python
# 在文件顶部添加参数
def expand_to_token_level(data: "DataProto", use_nstep_returns: bool = False):
    """
    Expand response-level rewards to token-level, placing the reward at EOS token.

    Args:
        data: DataProto containing rewards
        use_nstep_returns: If True and "nstep_returns" exists in batch, use it instead
                          of "response_level_rewards". This allows using multi-step
                          bootstrapped returns from replay buffer.

    Returns:
        token_level_rewards: [batch_size, seq_len-1] tensor with rewards at EOS positions
    """
    # Select reward source
    if use_nstep_returns and "nstep_returns" in data.batch:
        response_level_rewards = data.batch["nstep_returns"].clone().detach()
        logger.debug(
            f"Using n-step returns: n={data.meta_info.get('nstep_n', 'unknown')}, "
            f"mean={response_level_rewards.mean():.4f}"
        )
    else:
        response_level_rewards = data.batch["response_level_rewards"].clone().detach()

    batch_size = data.batch.batch_size[0]

    # Expand as token_level_rewards
    attention_mask = data.batch["attention_mask"]
    position_ids = data.batch["position_ids"]

    if position_ids.dim() == 3:
        # qwen2vl, (bsz, 3, seqlen)
        position_ids = position_ids[:, 0]

    eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
    token_level_rewards = torch.zeros_like(attention_mask, dtype=response_level_rewards.dtype)
    token_level_rewards[torch.arange(batch_size), eos_mask_idx] = response_level_rewards

    # Select the response part (remove first token)
    token_level_rewards = token_level_rewards[:, 1:]

    return token_level_rewards


@torch.no_grad()
def apply_kl_penalty(
    data: "DataProto",
    kl_ctrl: AdaptiveKLController,
    kl_penalty="kl",
    use_nstep_returns: bool = False
):
    """
    Apply KL penalty to rewards and generate token-level rewards.

    Args:
        data: DataProto with batch data
        kl_ctrl: Adaptive KL controller
        kl_penalty: Type of KL penalty ("kl", "abs", "mse", etc.)
        use_nstep_returns: Whether to use n-step returns from replay buffer

    Returns:
        data: Updated DataProto with "token_level_rewards"
        metrics: Dictionary of metrics
    """
    response_mask = data.batch["response_mask"][:, 1:]

    # Generate token-level rewards (may use n-step returns)
    token_level_rewards = expand_to_token_level(data, use_nstep_returns=use_nstep_returns)

    if "token_level_rewards" in data.batch.keys():
        data.rename(old_keys="token_level_rewards", new_keys="token_level_scores")

    batch_size = data.batch.batch_size[0]

    # Compute KL divergence
    if "ref_log_probs" in data.batch.keys():
        kld = compute_approx_kl(
            log_probs=data.batch["old_log_probs"],
            log_probs_base=data.batch["ref_log_probs"],
            action_mask=response_mask,
            kl_penalty=kl_penalty,
        )  # (batch_size, seq_len-1)
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    # Apply KL penalty
    token_level_rewards = token_level_rewards - beta * kld

    # Update KL controller
    current_kl = masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    kl_ctrl.update(current=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {
        "critic/kl": current_kl,
        "critic/kl_coef": beta
    }

    # Add n-step metrics if used
    if use_nstep_returns and "nstep_returns" in data.batch:
        completeness = data.batch.get("nstep_completeness", None)
        metrics.update({
            "nstep/enabled": 1.0,
            "nstep/returns_mean": data.batch["nstep_returns"].mean().item(),
            "nstep/returns_std": data.batch["nstep_returns"].std().item(),
        })
        if completeness is not None:
            metrics["nstep/completeness_ratio"] = completeness.float().mean().item()

    return data, metrics
```

### 6.2 修改 `agentic_pipeline.py`

```python
# 在run()主循环中，大约在line 380-390附近

# ... (前面的代码: rollout, replay integration, etc.)

# 🔑 检查是否使用n-step returns
use_nstep = False
if self.pipeline_config.replay.enabled and hasattr(self.pipeline_config.replay, 'enable_nstep'):
    use_nstep = (
        self.pipeline_config.replay.enable_nstep and
        self.pipeline_config.replay.get('use_nstep_in_advantage', False) and
        "nstep_returns" in batch.batch
    )

    if use_nstep:
        logger.debug(
            f"Using n-step returns in advantage computation: "
            f"n={self.pipeline_config.replay.n_step}, "
            f"gamma_step={self.pipeline_config.replay.nstep_gamma}"
        )

# Apply KL penalty with n-step support
batch, kl_metrics = apply_kl_penalty(
    data=batch,
    kl_ctrl=self.kl_ctrl,
    kl_penalty=self.pipeline_config.kl_penalty,
    use_nstep_returns=use_nstep  # 🔑 传递标志
)

# ... (后续的compute_advantage等)
```

### 6.3 修改配置 `agentic_config.py`

```python
@dataclass
class ReplayConfig:
    """Replay buffer configuration"""
    enabled: bool = field(default=False, metadata={"help": "Enable replay buffer"})
    capacity: int = field(default=100000, metadata={"help": "Replay buffer capacity"})

    # ... (其他现有字段)

    # N-Step Returns Configuration
    enable_nstep: bool = field(
        default=False,
        metadata={"help": "Enable n-step returns computation for replay buffer."}
    )
    n_step: int = field(
        default=5,
        metadata={"help": "Number of steps for n-step returns (step-level, not token-level)."}
    )
    nstep_gamma: float = field(
        default=0.99,
        metadata={"help": "Discount factor for step-level n-step returns (γ_step)."}
    )
    use_nstep_in_advantage: bool = field(
        default=False,
        metadata={
            "help": "Whether to use n-step returns in advantage computation. "
                    "If True, n-step returns replace single-step rewards in token-level GAE."
        }
    )

    # ... (其他字段)


@dataclass
class AgenticConfig:
    """Main agentic pipeline configuration"""

    # ... (其他字段)

    # Token-level GAE
    gamma: float = field(
        default=0.995,
        metadata={
            "help": "Token-level discount factor (γ_token). Should be close to 1.0 "
                    "since tokens within a response have small time intervals."
        }
    )
    lambd: float = field(
        default=0.95,
        metadata={"help": "GAE lambda parameter for token-level advantage estimation."}
    )

    # ... (replay字段)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
```

### 6.4 配置文件示例

```yaml
# experiments/config.yaml

# Token-level GAE配置
adv_estimator: gae
gamma: 0.995  # γ_token - 内层token折扣，接近1
lambd: 0.95
advantage_clip: 10.0
whiten_advantages: true

# Replay buffer配置
replay:
  enabled: true
  capacity: 100000
  batch_size: 128

  # Step-level N-Step Returns
  enable_nstep: true
  n_step: 5  # 外层step数量
  nstep_gamma: 0.99  # γ_step - 外层step折扣，标准RL
  use_nstep_in_advantage: true  # 🔑 启用在advantage中使用

  # Priority (可选)
  priority:
    function: combined
    alpha: 0.6

# Critic warmup
critic_warmup: 100

# Logging
logging_steps: 10
```

---

## 7. 实验验证方案

### 7.1 对比实验

**实验组**:
1. **Baseline**: `use_nstep_in_advantage=false` (当前行为)
2. **N-Step-3**: `n_step=3, use_nstep_in_advantage=true`
3. **N-Step-5**: `n_step=5, use_nstep_in_advantage=true`
4. **N-Step-10**: `n_step=10, use_nstep_in_advantage=true`

**评估指标**:
- Episode reward mean/std
- Success rate (任务相关)
- Training stability (reward variance)
- Sample efficiency (达到目标性能的steps)
- Completeness ratio (n-step完整度)

### 7.2 消融实验

**变量**:
1. `nstep_gamma`: [0.95, 0.99, 1.0]
2. `gamma` (token-level): [0.99, 0.995, 0.999]
3. `n_step`: [3, 5, 7, 10]

**分析**:
- Gamma对学习稳定性的影响
- N-step对长期credit assignment的改善
- Completeness ratio vs performance关系

### 7.3 监控Dashboard

```python
# Wandb/TensorBoard logging
metrics = {
    # Performance
    "train/episode_reward_mean": ...,
    "train/episode_reward_std": ...,
    "train/success_rate": ...,

    # N-Step specific
    "nstep/enabled": ...,
    "nstep/completeness_ratio": ...,
    "nstep/returns_mean": ...,
    "nstep/returns_vs_single_ratio": ...,

    # GAE
    "critic/advantages_mean": ...,
    "critic/advantages_std": ...,
    "critic/values_mean": ...,

    # Off-policy
    "offpolicy/kl_divergence": ...,
    "offpolicy/importance_ratio": ...,
}
```

---

## 8. 总结

### 8.1 关键要点

1. **两层RL结构**:
   - 外层: Step-level环境交互 (用γ_step≈0.99)
   - 内层: Token-level策略优化 (用γ_token≈0.995)

2. **N-Step Returns的正确使用**:
   - 当前: 计算但未使用 ❌
   - 正确: 替换单步reward作为token-level的最终奖励 ✅

3. **推荐实现**: 方案A (直接替换)
   - 修改 `expand_to_token_level` 和 `apply_kl_penalty`
   - 添加 `use_nstep_returns` 参数
   - 配置 `use_nstep_in_advantage: true`

4. **Gamma参数区分**:
   - `nstep_gamma`: Step-level折扣 (0.99)
   - `gamma`: Token-level折扣 (0.995-0.999)

### 8.2 预期效果

**优势**:
- ✅ 更好的长期credit assignment (跨多个steps)
- ✅ 减少variance (bootstrap而非Monte Carlo)
- ✅ 充分利用replay buffer数据
- ✅ 理论上更正确的分层RL

**潜在风险**:
- ⚠️ 如果completeness低,效果可能不明显
- ⚠️ 需要调整gamma参数
- ⚠️ 与GIGPO可能冲突

### 8.3 Next Steps

1. **立即实现**: 方案A (~50 lines code)
2. **测试**: 在简单任务上验证
3. **调优**: 根据completeness ratio调整n_step
4. **监控**: 添加充分的logging
5. **消融**: 对比不同配置的效果

---

**文档版本**: 1.0.0
**作者**: Claude
**日期**: 2025-10-30
**状态**: 实现方案完整
