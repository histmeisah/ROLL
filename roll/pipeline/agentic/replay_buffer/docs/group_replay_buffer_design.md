# GroupReplayBuffer 设计文档

## 1. 背景与动机

### 问题

原有的 `TrajectoryReplayBuffer` 以单条轨迹为存取粒度。当 GRPO 使用 `group_size > 1`（同一个 prompt 生成 K 条 response）时：

1. **采样打破 group 结构**：K 条同组轨迹被独立存储和随机采样，导致组内 reward 归一化失效
2. **单样本归一化**：采样出的 batch 中每个 `traj_group_id` 可能只有 1 条轨迹，组内 `mean=自身, std=0`，advantage 全为 0
3. **Pipeline 缺失集成**：`run()` 中没有 push/sample 调用

### 解决方案

引入 `GroupReplayBuffer`，以 **trajectory group** 为原子存取单位：

```
存储单位: TrajectoryGroup = {traj_group_id, [traj_0, traj_1, ..., traj_{K-1}]}
采样单位: N 个 group → 展开为 N×K 条轨迹的 DataProto
```

## 2. 架构设计

### 类层次

```
BaseReplayBuffer (base_buffer.py)
├── TrajectoryReplayBuffer (trajectory_buffer.py)  # 单条轨迹存取
├── StepReplayBuffer (step_buffer.py)              # step-level 存取
└── GroupReplayBuffer (group_buffer.py)             # group-level 存取 ← 新增
```

### 数据结构

```python
@dataclass
class TrajectoryGroup:
    traj_group_id: str              # 唯一标识（同 prompt/state 的 K 条轨迹）
    trajectories: List[TrajectoryEntry]  # K 条轨迹
    group_size: int                 # K
    tag: str                        # 环境标签
    mean_episode_score: float       # 组内平均 episode score
    stored_at_step: int             # 存入时的 global_step
    priority: float                 # group-level priority
    sample_count: int               # 被采样次数
    global_step: int                # 用于 age 计算
```

### 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 存储粒度 | group | 保证 GRPO 组内归一化的完整性 |
| capacity 含义 | group 数量 | 直观，与 group_size 无关 |
| push 时机 | rollout 后、reward norm 前 | 存储原始数据，采样时重新计算 reward/advantage |
| sample 后处理 | 走完整 pipeline | reward norm → token reward → advantage → train |
| priority 粒度 | group-level | 用首条轨迹的 priority_fn 结果代表整组 |
| PER weights 广播 | group weight → 组内每条轨迹 | 同组轨迹共享采样权重 |

## 3. 数据流

### On-policy 训练（不变）

```
Rollout → get_batch → adjust_batch → ref_log_probs → old_log_probs
→ reward_norm → token_reward → advantage → actor.train_step
```

### Replay 训练（新增）

```
Rollout → get_batch → [PUSH to buffer] → ... on-policy train ...
→ [SAMPLE from buffer] → adjust_batch → old_log_probs (recompute)
→ ref_log_probs → reward_norm → token_reward → advantage
→ actor.train_step (replay)
```

### Pipeline 集成位置

```
PHASE 7:  get_batch (rollout)
PHASE 10: push_from_dataproto(batch, global_step)     ← 新增
PHASE 11-14: on-policy training (不变)
PHASE 15: replay training loop                          ← 新增
  for replay_step in range(train_steps_per_env_step):
    sample → adjust → old_log_probs → ref_log_probs
    → reward_norm → advantage → train
```

## 4. GRPO 兼容性

### 为什么 GroupReplayBuffer 与 GRPO 兼容

1. **组内完整性**：sample 返回完整的 group，每个 `traj_group_id` 包含 K 条轨迹
2. **reward norm 正常工作**：`agentic_reward_norm()` 按 `traj_group_id` 分组，组内 K 条轨迹的 mean/std 有意义
3. **advantage 正确计算**：GRPO 的 reinforce advantage 在组内归一化后的 reward 上计算

### group_size=1 时的退化

当 `group_size=1` 时，每个 group 只有 1 条轨迹，行为等价于原 `TrajectoryReplayBuffer`。
Reward norm 退化为 batch-level（因为组内只有 1 个样本），这与 reinforce/PPO 的默认行为一致。

## 5. 配置

### ReplayConfig 新增字段

```yaml
replay:
  enabled: true
  group_level: true              # 使用 GroupReplayBuffer（GRPO 必须为 true）
  capacity: 10000                # 最多存储的 group 数量
  train_steps_per_env_step: 1    # 每次 rollout 后额外训练几步
  sample_method: "uniform"       # 采样方法
  priority_function: "uniform"   # Priority 函数
  # ... 其他 PER 配置不变
```

### 配置示例

```yaml
# GRPO + Replay Buffer
train_env_manager:
  group_size: 4

replay:
  enabled: true
  group_level: true
  capacity: 5000          # 5000 groups × 4 trajectories = 20000 trajectories
  train_steps_per_env_step: 2
  sample_method: uniform
  priority_function: reward
  priority_exponent: 0.6

adv_estimator: grpo
reward_normalization:
  grouping: traj_group_id
  method: mean_std
```

## 6. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `replay_buffer/group_buffer.py` | 新增 | GroupReplayBuffer + TrajectoryGroup |
| `replay_buffer/buffer_factory.py` | 修改 | 添加 `group_level` 参数和 GroupReplayBuffer 创建 |
| `replay_buffer/__init__.py` | 修改 | 导出 GroupReplayBuffer, TrajectoryGroup |
| `agentic_config.py` | 修改 | ReplayConfig 添加 `group_level` 字段 |
| `agentic_pipeline.py` | 修改 | 初始化传入 group_level；run() 中集成 push/sample |

## 7. 后续优化方向

1. **Async replay training**：replay 训练与 rollout 并行执行
2. **Stale group 检测**：当 group 的 behavior policy 与当前 policy 差距过大时自动丢弃
3. **Group-level PER**：基于组内 advantage 方差或 mean reward 的优先级函数
4. **Mixed batch**：on-policy 和 replay 数据混合在同一个 batch 中训练
