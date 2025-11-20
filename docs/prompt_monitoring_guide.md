# Prompt性能监控指南

## 概述

Bandit-REINFORCE++内置了prompt性能监控系统，自动追踪每个prompt的表现，帮助你：
- 实时查看哪个prompt效果最好
- 监控UCB算法的探索-利用平衡
- 分析prompt选择的收敛情况
- 所有数据自动同步到wandb

## 快速开始

### 1. 在训练脚本中启用监控

```python
from roll.algorithms.bandit import BanditReinforcePlusPlus

# 初始化时自动启用监控
bandit_rl = BanditReinforcePlusPlus(
    prompt_templates=prompts,
    enable_monitoring=True,  # 默认开启
    monitor_save_dir="./outputs/monitor",  # 保存监控数据
)
```

### 2. 监控数据自动输出到wandb

训练过程中，以下指标会自动记录到wandb：

**核心指标**:
- `bandit/episode_reward` - 每个episode的平均reward
- `bandit/episode_best_reward` - 每个episode最好的reward
- `bandit/recent_avg_reward` - 最近100个episode的平均reward

**UCB算法指标**:
- `bandit/ucb_value` - UCB值（预测reward + 探索bonus）
- `bandit/predicted_reward` - 神经网络预测的reward
- `bandit/confidence` - 置信界宽度（探索bonus）

**Per-Prompt指标** (每个prompt都有):
- `prompts/{prompt_name}/mean_reward` - 该prompt的平均reward
- `prompts/{prompt_name}/success_rate` - 成功率（对于二分类任务）
- `prompts/{prompt_name}/selections` - 被选择的次数

**选择分布**:
- `selection_dist/{prompt_name}` - 最近100个episode中该prompt的选择频率

**收敛指标**:
- `monitor/is_converged` - 是否收敛到某个prompt
- `monitor/best_prompt_reward` - 最佳prompt的reward

### 3. 查看实时监控

训练时在终端会定期输出summary：

```
================================================================================
Prompt Performance Monitor Summary
================================================================================
Total Episodes: 500
Global Mean Reward: 0.6540
Recent Mean Reward: 0.6821

--------------------------------------------------------------------------------
Best Performing Prompt:
--------------------------------------------------------------------------------
  zero_shot_cot_ape (idx=0)
  Mean Reward: 0.7456

--------------------------------------------------------------------------------
Prompt Rankings (by mean reward):
--------------------------------------------------------------------------------
  1. zero_shot_cot_ape              | Reward: 0.7456 | Selections:  201 | Success: 74.63%
  2. plan_and_solve_plus            | Reward: 0.6523 | Selections:  123 | Success: 65.04%
  3. solve_and_verify               | Reward: 0.6012 | Selections:   98 | Success: 60.20%
  4. detailed_explanation           | Reward: 0.5487 | Selections:   51 | Success: 54.90%
  5. direct_solve                   | Reward: 0.4023 | Selections:   27 | Success: 40.74%

--------------------------------------------------------------------------------
Recent Selection Distribution:
--------------------------------------------------------------------------------
  zero_shot_cot_ape              | 40.20% ████████████████████
  plan_and_solve_plus            | 24.60% ████████████
  solve_and_verify               | 19.60% █████████
  detailed_explanation           | 10.20% █████
  direct_solve                   |  5.40% ██

--------------------------------------------------------------------------------
Convergence Status:
--------------------------------------------------------------------------------
  ✗ Still exploring...
================================================================================
```

### 4. 在wandb中分析

登录wandb查看：

**主要看这些图表**:
1. `prompts/*/mean_reward` - 对比所有prompt的性能曲线
2. `selection_dist/*` - 看算法逐渐偏向哪个prompt
3. `bandit/confidence` - 探索bonus随时间下降说明算法越来越确信
4. `monitor/is_converged` - 何时收敛

**典型的好现象**:
- 最好的prompt的`mean_reward`持续最高
- `selection_dist`逐渐集中到1-2个prompt
- `confidence`逐渐下降
- `is_converged`最终变为1

## 高级用法

### 手动获取监控数据

```python
# 在训练过程中
if bandit_rl.enable_monitoring:
    # 获取完整summary
    summary = bandit_rl.monitor.get_summary()

    # 打印到终端
    bandit_rl.monitor.print_summary()

    # 获取wandb格式的metrics
    from roll.algorithms.bandit import get_wandb_metrics
    metrics = get_wandb_metrics(bandit_rl.monitor)

    # 保存到文件
    bandit_rl.monitor.save_data("monitor_checkpoint.json")
```

### 分析特定prompt

```python
# 获取某个prompt的详细统计
prompt_idx = 0  # zero_shot_cot_ape
stats = bandit_rl.monitor.prompt_stats[prompt_idx]

print(f"Prompt: {stats.prompt_name}")
print(f"Total selections: {stats.total_selections}")
print(f"Mean reward: {stats.mean_reward:.4f}")
print(f"Success rate: {stats.success_rate:.2%}")
print(f"Recent mean reward: {stats.recent_mean_reward:.4f}")
```

### 查看prompt排名

```python
# 获取按reward排序的prompt列表
rankings = bandit_rl.monitor.get_prompt_rankings()

for rank, (idx, name, reward) in enumerate(rankings, 1):
    print(f"{rank}. {name}: {reward:.4f}")
```

### 检查收敛状态

```python
summary = bandit_rl.monitor.get_summary()

if summary['convergence']['is_converged']:
    print(f"✓ Converged to: {summary['convergence']['converged_name']}")
else:
    print("✗ Still exploring")

    # 查看当前选择分布
    dist = bandit_rl.monitor.get_selection_distribution(recent=True)
    for name, freq in sorted(dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {freq:.1%}")
```

## 监控数据格式

### Summary JSON结构

```json
{
  "total_episodes": 500,
  "n_prompts": 5,
  "best_prompt": {
    "idx": 0,
    "name": "zero_shot_cot_ape",
    "mean_reward": 0.7456
  },
  "convergence": {
    "is_converged": false,
    "converged_prompt": null,
    "converged_name": null
  },
  "prompt_rankings": [
    {
      "rank": 1,
      "idx": 0,
      "name": "zero_shot_cot_ape",
      "mean_reward": 0.7456
    }
  ],
  "selection_distribution": {
    "zero_shot_cot_ape": 0.402,
    "plan_and_solve_plus": 0.246
  },
  "prompt_stats": {
    "zero_shot_cot_ape": {
      "prompt_idx": 0,
      "prompt_name": "zero_shot_cot_ape",
      "total_selections": 201,
      "mean_reward": 0.7456,
      "std_reward": 0.1834,
      "success_rate": 0.7463
    }
  }
}
```

## 常见问题

### Q: 如何判断哪个prompt最好？

看3个指标：
1. **mean_reward最高** - 平均性能最好
2. **被选次数多** - 算法认为它好
3. **recent_mean_reward高且稳定** - 最近表现依然好

### Q: 为什么最好的prompt不总是被选？

这是UCB算法的探索机制：
- **前期**: 大量探索，所有prompt都会尝试
- **中期**: 逐渐偏向好的prompt，但仍保持探索
- **后期**: 主要使用最好的prompt，偶尔探索

通过`confidence`值可以看探索程度，该值越小表示探索越少。

### Q: 如何加快收敛？

降低探索参数：
```python
bandit_rl = BanditReinforcePlusPlus(
    exploration_param=0.5,  # 默认1.0，降低会更快利用
)
```

但注意：太快收敛可能错过更好的prompt！

### Q: 监控数据保存在哪？

- wandb云端: 自动同步
- 本地JSON: `{monitor_save_dir}/monitor_data_episode_{N}.json`
- Checkpoint: 训练checkpoint包含完整监控状态

## 配置选项

在`bandit_reinforce_config.yaml`中配置：

```yaml
bandit:
  exploration_param: 1.0  # UCB探索参数，越大探索越多

# 监控相关（可选）
monitoring:
  enabled: true
  save_dir: "./outputs/monitor"
  window_size: 100  # 计算recent metrics的窗口大小
  convergence_threshold: 0.1  # 收敛判断阈值
```

## 最佳实践

1. **前100个episode**: 主要看是否所有prompt都被尝试
2. **100-500个episode**: 关注rankings是否稳定，best prompt是否一致
3. **500+个episode**: 看是否收敛，selection_dist是否集中
4. **实验对比**: 在wandb中对比不同exploration_param的收敛速度

记住：**监控的目的是理解算法行为，不是过早干预！** UCB算法会自动平衡探索和利用。
