# N-Step Returns使用指南

## 概述

N-Step Returns功能为ROLL框架的StepReplayBuffer提供了更高效的样本效率和训练稳定性。本功能完全基于Tianshou的设计理念实现,通过Episode Index维护episode结构,支持动态n-step returns计算。

## 核心特性

### ✅ 已实现的功能

1. **Episode Index** - 高效的episode结构跟踪
   - 自动维护 (traj_id, step) -> buffer_idx 映射
   - O(1)时间复杂度的step查找
   - 自动处理buffer wrap-around时的清理

2. **N-Step Returns** - 多步TD学习
   - 支持任意n-step配置(默认5-step)
   - 可选bootstrap values支持
   - 自动检测并处理不完整episodes

3. **GAE Support** - 广义优势估计(基础版)
   - 支持truncated GAE计算
   - 可配置lambda参数
   - 当前为简化实现,完整GAE需要critic values存储

4. **完整的监控和统计**
   - Episode index统计(episode数量,平均长度)
   - N-step completeness ratio
   - 与PER无缝集成

---

## 快速开始

### 1. 配置文件设置

在你的YAML配置文件中启用n-step:

```yaml
replay:
  enabled: true
  capacity: 100000

  # N-Step Returns配置
  enable_nstep: true    # 启用n-step returns
  n_step: 5             # 5-step TD
  nstep_gamma: 0.99     # 折扣因子
  use_bootstrap: false  # 是否使用critic bootstrap

  # GAE配置(可选,高级功能)
  enable_gae: false
  gae_lambda: 0.95
  gae_horizon: 20
```

### 2. 训练脚本

无需修改训练脚本!N-step returns会自动在replay buffer采样时计算:

```python
# experiments/start_agentic_pipeline.py
python experiments/start_agentic_pipeline.py \
    --config_name your_config \
    replay.enable_nstep=true \
    replay.n_step=5
```

### 3. 验证功能

检查日志输出,应该看到:

```
[INFO] Creating StepReplayBuffer: ..., enable_nstep=True, n_step=5, gamma=0.99
[DEBUG] Computed n-step returns: n_step=5, gamma=0.99, complete_ratio=0.85
```

---

## 配置详解

### N-Step Returns参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_nstep` | bool | false | 启用n-step returns |
| `n_step` | int | 5 | n-step TD的步数 |
| `nstep_gamma` | float | 0.99 | 折扣因子γ |
| `use_bootstrap` | bool | false | 使用critic values进行bootstrap |

### GAE参数(高级)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_gae` | bool | false | 启用GAE |
| `gae_lambda` | float | 0.95 | GAE lambda参数 |
| `gae_horizon` | int | 20 | GAE截断horizon |

---

## 使用场景

### 场景1: 基础n-step TD (推荐入门)

**适用情况**: 稳定环境,episode较长

```yaml
replay:
  enable_nstep: true
  n_step: 5
  nstep_gamma: 0.99
  use_bootstrap: false  # 不使用bootstrap,纯Monte Carlo
```

**优点**:
- 简单高效
- 无需critic
- 训练稳定

**缺点**:
- 不完整episodes的returns精度较低

### 场景2: Bootstrap N-Step TD (推荐生产)

**适用情况**: 需要更高样本效率

```yaml
replay:
  enable_nstep: true
  n_step: 5
  nstep_gamma: 0.99
  use_bootstrap: true  # 使用critic bootstrap
```

**优点**:
- 提高不完整episodes的returns精度
- 更好的样本效率
- 适合长episode

**缺点**:
- 需要额外的critic forward pass
- 计算开销略高

### 场景3: GAE (高级用户)

**适用情况**: 需要精细的优势估计

```yaml
replay:
  enable_nstep: false  # GAE替代n-step
  enable_gae: true
  gae_lambda: 0.95
  gae_horizon: 20
```

**注意**: 当前GAE实现为简化版本,完整功能需要在buffer中存储critic values。

---

## 性能优化建议

### 内存开销

Episode Index内存开销估算:

```python
# 假设配置
capacity = 100000 steps
avg_episode_length = 5 steps
num_episodes = capacity / avg_episode_length = 20000

# 内存开销
episode_index_size = 20000 * 5 * 50 bytes ≈ 5 MB
buffer_to_episode_size = 100000 * 50 bytes ≈ 5 MB
total_overhead ≈ 10 MB  # 可忽略不计
```

**结论**: 对于100K capacity的buffer,Episode Index仅增加~10MB内存,可忽略不计。

### 计算开销

N-Step Returns计算复杂度:

```
时间复杂度: O(batch_size * n_step)
示例: batch_size=128, n_step=5
     总计算量: 128 * 5 = 640次step查找和reward累加
     开销: 可忽略 (<1ms)
```

### 推荐配置

根据buffer容量和episode长度选择n_step:

| Buffer Capacity | Avg Episode Length | 推荐n_step | 原因 |
|-----------------|--------------------|-----------|----|
| 10K - 50K | 3-5 steps | 3 | 减少不完整episodes |
| 50K - 100K | 5-10 steps | 5 | 平衡效率和完整性 |
| 100K+ | 10+ steps | 7-10 | 充分利用长episodes |

---

## 监控和调试

### 重要指标

在训练日志中关注以下指标:

```python
# Episode Index统计
episode_index/num_episodes: 15234        # Buffer中的episode数量
episode_index/avg_episode_length: 4.8   # 平均episode长度
episode_index/total_indexed_steps: 73123 # 已索引的step总数

# N-Step Completeness
nstep_complete_ratio: 0.85               # 85%的samples有完整n-step序列
```

### Completeness Ratio解读

**Completeness Ratio** = 能找到完整n-step序列的样本比例

- **> 0.9**: 优秀,大部分samples有完整n-step序列
- **0.7 - 0.9**: 良好,可接受
- **< 0.7**: 较差,考虑:
  - 减小n_step
  - 增加buffer capacity
  - 启用bootstrap

### 调试技巧

**问题1: Completeness ratio过低**

```yaml
# 解决方案1: 减小n_step
replay:
  n_step: 3  # 从5降到3

# 解决方案2: 启用bootstrap
replay:
  use_bootstrap: true

# 解决方案3: 增加capacity
replay:
  capacity: 200000  # 从100K增加到200K
```

**问题2: Episode index占用过多内存**

```python
# 检查stats
buffer.get_stats()
# 如果 episode_index/num_episodes > 50K,考虑减小capacity
```

**问题3: N-step计算速度慢**

```yaml
# 减小n_step
replay:
  n_step: 3  # 从更大的值降低

# 或考虑禁用n-step
replay:
  enable_nstep: false
```

---

## API文档

### StepReplayBuffer N-Step方法

#### `get_nstep_indices(start_idx, n_step)`

获取从起始index开始的n个连续steps(同一episode)。

**参数**:
- `start_idx`: int - 起始buffer index
- `n_step`: int - 需要的step数量

**返回**:
- `indices`: List[int] - 找到的buffer indices
- `complete`: bool - 是否找到完整的n步序列

**示例**:
```python
indices, complete = buffer.get_nstep_indices(start_idx=42, n_step=5)
if complete:
    print(f"找到完整5-step序列: {indices}")
else:
    print(f"仅找到{len(indices)}步,遇到episode边界")
```

#### `compute_nstep_returns(sampled_indices, n_step, gamma, bootstrap_values)`

计算n-step returns。

**参数**:
- `sampled_indices`: List[int] - 采样的起始indices
- `n_step`: int - n-step数量(可选,默认self.n_step)
- `gamma`: float - 折扣因子(可选,默认self.gamma)
- `bootstrap_values`: np.ndarray - Bootstrap values(可选)

**返回**:
- `returns`: np.ndarray - 计算的n-step returns
- `completeness_mask`: np.ndarray - 完整性mask

**示例**:
```python
returns, completeness = buffer.compute_nstep_returns(
    sampled_indices=[0, 10, 20],
    n_step=5,
    gamma=0.99,
    bootstrap_values=critic_values  # 可选
)
print(f"Returns: {returns}")
print(f"Complete samples: {completeness.sum()} / {len(completeness)}")
```

#### `compute_gae(sampled_indices, values, gamma, lambda_, n_step)`

计算广义优势估计(GAE)。

**参数**:
- `sampled_indices`: List[int] - 采样的起始indices
- `values`: np.ndarray - State values V(s_t)
- `gamma`: float - 折扣因子(可选)
- `lambda_`: float - GAE lambda参数
- `n_step`: int - 截断horizon

**返回**:
- `advantages`: np.ndarray - GAE优势
- `completeness_mask`: np.ndarray - 完整性mask

**示例**:
```python
advantages, completeness = buffer.compute_gae(
    sampled_indices=[0, 10, 20],
    values=critic_values,
    gamma=0.99,
    lambda_=0.95,
    n_step=20
)
```

---

## 实现原理

### Episode Index设计

```python
# 两个核心数据结构
self._episode_index = {
    "ep_001": {0: 42, 1: 43, 2: 44},  # traj_id -> {step -> buffer_idx}
}

self._buffer_to_episode = {
    42: ("ep_001", 0),  # buffer_idx -> (traj_id, step)
}
```

**关键操作**:
1. **Push**: 更新两个映射,O(1)
2. **Eviction**: 清理被覆盖的step,O(1)
3. **Lookup**: 查找next step,O(1)

### N-Step Traversal

模仿tianshou的`next()`方法:

```python
def get_nstep_indices(start_idx, n_step):
    indices = []
    for offset in range(n_step):
        target_step = start_step + offset
        if target_step not in episode:
            return indices, False  # 遇到边界
        indices.append(episode[target_step])
    return indices, True  # 完整序列
```

### N-Step Returns公式

```
R_t^(n) = r_t + γ*r_{t+1} + ... + γ^(n-1)*r_{t+n-1} + γ^n*V(s_{t+n})
                                                        ^^^^^^^^^^^^^^^^
                                                        可选bootstrap
```

---

## 与Tianshou的对比

| 维度 | Tianshou | ROLL StepReplayBuffer |
|------|----------|----------------------|
| **Episode跟踪** | 隐式(done标志) | 显式(Episode Index) |
| **`next()`实现** | `(i+1) % size if not done[i]` | `episode_index[traj_id][step+1]` |
| **适用场景** | 标准RL(Atari, MuJoCo) | LLM多轮对话RL |
| **时间复杂度** | O(1) | O(1) |
| **内存开销** | 0 (隐式) | ~10MB for 100K buffer |

**结论**: ROLL的实现保持了Tianshou的所有优点,仅通过显式Episode Index适配LLM场景。

---

## FAQ

### Q1: 为什么需要Episode Index?

**A**: 因为StepReplayBuffer存储独立的conversation steps,经过buffer wrap-around后,同一episode的steps会分散在buffer中。Tianshou使用`done`标志隐式维护episode结构,而ROLL需要显式的Episode Index来维护`(traj_id, step) -> buffer_idx`映射。

### Q2: N-Step和GAE有什么区别?

**A**:
- **N-Step Returns**: 简单的多步TD,计算R_t = Σ γ^k * r_{t+k} + γ^n * V_{t+n}
- **GAE**: 更复杂的优势估计,使用指数加权的TD errors: A_t = Σ (γλ)^k * δ_{t+k}
- **推荐**: 初学者使用N-Step,高级用户使用GAE

### Q3: Bootstrap values从哪里来?

**A**: 当`use_bootstrap=true`时,需要在AgenticPipeline中添加critic forward pass来计算values。当前默认为`false`,使用纯Monte Carlo returns。

### Q4: 内存开销大吗?

**A**: 非常小。对于100K capacity的buffer,Episode Index仅占用~10MB(相比tokenized data的数GB,可忽略不计)。

### Q5: 是否影响训练速度?

**A**: 几乎不影响。N-step计算的时间复杂度为O(batch_size * n_step),对于典型配置(batch=128, n_step=5),计算量可忽略(<1ms)。

### Q6: Completeness ratio多少算正常?

**A**:
- **> 0.9**: 优秀
- **0.7 - 0.9**: 良好
- **< 0.7**: 需要调整(减小n_step或增加capacity)

### Q7: 可以和PER一起使用吗?

**A**: 完全可以!N-Step和PER是正交的功能,可以同时启用:

```yaml
replay:
  enable_nstep: true
  n_step: 5

  priority:
    function: reward  # PER
    alpha: 0.6
    use_importance_weights: true
```

---

## 最佳实践总结

### ✅ 推荐做法

1. **从小开始**: 初次使用时,设置`n_step=3`,逐步增加
2. **监控completeness**: 保持ratio > 0.8
3. **根据episode长度调整**: 长episodes可以用更大的n_step
4. **结合PER**: N-Step + PER = 更好的样本效率
5. **先禁用bootstrap**: 等稳定后再启用`use_bootstrap=true`

### ❌ 避免的错误

1. **n_step过大**: 不要超过平均episode长度
2. **忽略completeness ratio**: 低ratio会导致训练不稳定
3. **盲目启用GAE**: GAE需要更多配置,建议先用N-Step
4. **过小的buffer**: 确保capacity >> n_step * batch_size

---

## 示例配置

### 配置1: 保守型(推荐新手)

```yaml
replay:
  enabled: true
  capacity: 50000

  enable_nstep: true
  n_step: 3
  nstep_gamma: 0.99
  use_bootstrap: false
```

### 配置2: 平衡型(推荐生产)

```yaml
replay:
  enabled: true
  capacity: 100000

  enable_nstep: true
  n_step: 5
  nstep_gamma: 0.99
  use_bootstrap: true

  priority:
    function: combined
    alpha: 0.6
    kwargs:
      reward_weight: 0.7
      recency_weight: 0.3
```

### 配置3: 激进型(高级用户)

```yaml
replay:
  enabled: true
  capacity: 200000

  enable_nstep: true
  n_step: 10
  nstep_gamma: 0.995
  use_bootstrap: true

  enable_gae: true
  gae_lambda: 0.95
  gae_horizon: 20

  priority:
    function: td_error
    alpha: 0.7
    use_importance_weights: true
    importance_beta: 0.4
```

---

## 下一步

1. **阅读测试代码**: `tests/test_nstep_replay_buffer.py` 包含详细的使用示例
2. **查看实现细节**: `roll/agentic/replay_buffer/step_buffer.py` 查看完整实现
3. **参考Tianshou文档**: https://github.com/thu-ml/tianshou 了解n-step TD的理论基础
4. **运行实验**: 使用提供的配置模板开始你的第一次n-step训练!

---

## 参考文献

1. **Tianshou**: https://github.com/thu-ml/tianshou
2. **Schulman et al., 2016**: High-Dimensional Continuous Control Using Generalized Advantage Estimation
3. **Mnih et al., 2016**: Asynchronous Methods for Deep Reinforcement Learning (A3C n-step returns)
4. **ROLL Framework**: `roll_dev/ROLL/docs/`

---

**版本**: 1.0.0
**最后更新**: 2025-10-30
**作者**: Claude + ROLL Dev Team
