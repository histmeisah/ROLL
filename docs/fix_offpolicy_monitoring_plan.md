# Fix Off-Policy Monitoring for Fresh Batch

## Problem Statement

当前fresh batch的off-policy监控显示ratio全部为1，原因是：
- 监控时机错误：在训练前而非训练后监控
- 比较的是同一时刻、同一模型的log_probs，没有参数更新的差异

## Goal

修复fresh batch的off-policy监控，使其能够正确反映PPO训练中的importance sampling ratio。

## Implementation Plan

### 1. 核心思路

监控PPO算法中实际使用的importance sampling ratio：
- **比较对象**：`old_log_probs`（训练前计算，第280行）vs 训练后的current log_probs
- **监控时机**：第一次actor训练后（能反映PPO的trust region约束效果）
- **关键指标**：ratio的mean、std、percentiles、KL divergence、ESS等

### 2. 代码修改计划

#### 2.1 移除错误的监控位置

**文件**：`roll/pipeline/agentic/agentic_pipeline.py`
**位置**：第380-395行
**操作**：删除训练前的fresh batch监控代码

```python
# 删除这段代码：
# === NEW: Unified off-policy monitoring for fresh/echo batch ===
if (self.pipeline_config.offpolicy_monitor.enabled and
    self.pipeline_config.offpolicy_monitor.monitor_fresh_batch and
    global_step % self.pipeline_config.offpolicy_monitor.monitor_interval == 0 and
    "old_log_probs" in batch.batch):
    ...
```

#### 2.2 在正确位置添加监控

**文件**：`roll/pipeline/agentic/agentic_pipeline.py`
**位置**：第一次actor训练后（约第402行后）
**操作**：添加训练后的off-policy监控

```python
# 在actor_train.train_step后添加：
if self.pipeline_config.critic_warmup <= global_step:
    actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
    actor_train_metrics = DataProto.materialize_concat(data_refs=actor_train_metrics_refs)
    metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))

    # NEW: Monitor off-policy ratio after first training step
    if (self.pipeline_config.offpolicy_monitor.enabled and
        self.pipeline_config.offpolicy_monitor.monitor_fresh_batch and
        global_step % self.pipeline_config.offpolicy_monitor.monitor_interval == 0 and
        "old_log_probs" in batch.batch and
        not self.pipeline_config.replay.enabled):  # Only for fresh batch without replay

        # At this point, actor has been updated, compute current log_probs
        fresh_offpolicy_metrics = compute_offpolicy_metrics(
            current_batch=batch,
            actor_train_cluster=self.actor_train,
            old_prob_mode=self.pipeline_config.offpolicy_monitor.behavior_scope,
            metric_prefix="fresh/offpolicy",
            pg_clip=self.pipeline_config.pg_clip
        )
        metrics.update(fresh_offpolicy_metrics)
```

#### 2.3 确保compute_offpolicy_metrics使用正确的字段

**文件**：`roll/pipeline/agentic/offpolicy_monitor.py`
**修改**：优先使用`old_log_probs`而不是`behavior_log_probs`

```python
# 第48-52行修改为：
behavior_field = None
if "old_log_probs" in current_batch.batch:
    behavior_field = "old_log_probs"  # 优先使用PPO的old_log_probs
elif "behavior_log_probs" in current_batch.batch:
    behavior_field = "behavior_log_probs"  # 仅在replay buffer场景使用
```

### 3. 验证要点

修改后需要验证：
1. **ratio不再是1**：应该看到mean在1.0-1.2之间，std > 0
2. **clip fraction合理**：通常< 0.1
3. **KL divergence适中**：通常< 0.01
4. **与actor/ratio_mean一致**：fresh/offpolicy/ratio/mean应该接近actor/ratio_mean

### 4. 配置说明

使用时的配置：
```yaml
offpolicy_monitor:
  enabled: true
  monitor_fresh_batch: true  # 监控fresh batch
  monitor_replay_batch: true  # 监控replay batch（如果有）
  monitor_interval: 1  # 每个step都监控
  save_behavior_log_probs: false  # fresh batch不需要额外保存
  behavior_compute: trainer  # 使用trainer模式
  behavior_scope: trajectory  # 或turn，取决于需求
```

### 5. 预期效果

修复后，fresh batch的监控指标应该显示：
- `fresh/offpolicy/ratio/mean`: 1.05-1.15（健康的更新）
- `fresh/offpolicy/ratio/std`: 0.1-0.5（适度的方差）
- `fresh/offpolicy/ratio/p99`: < 2.0（没有极端值）
- `fresh/offpolicy/kl_divergence`: 0.001-0.01（适度的策略变化）
- `fresh/offpolicy/ess_ratio`: > 0.5（有效样本率高）

### 6. 注意事项

1. **性能影响**：训练后监控需要额外一次forward pass，但只在monitor_interval触发
2. **与replay buffer的区别**：
   - Fresh batch：监控训练前后的变化
   - Replay batch：监控存储时vs当前的变化
3. **多次训练**：当前只监控第一次训练后，这最能反映PPO的核心机制

### 7. 后续优化

可选的后续改进：
1. 添加配置选项选择监控第几次训练
2. 支持监控累积的策略变化
3. 添加更详细的token-level分析