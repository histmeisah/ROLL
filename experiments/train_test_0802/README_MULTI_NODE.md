# ROLL 多机多卡双环境训练指南

## 概述

本指南介绍如何使用ROLL框架进行多机多卡的双环境训练（搜索环境 + 数学环境）。

## 系统架构

- **双环境训练**: 搜索环境 + 数学环境 (1:1比例)
- **异步训练**: async_generation_ratio=1 实现完全异步训练
- **奖励系统**: Outcome-based + xverify + OTC
- **多机多卡**: 支持多节点，每节点8个GPU
- **离线日志**: WandB离线模式 + 轨迹采样保存

## 快速开始

### 1. 启动Ray集群

在跳板机上运行：

```bash
# 启动整个Ray集群（包括head node和所有worker nodes）
bash setup_cluster.sh
```

这个脚本会：
- 在master node启动Ray head node
- 在所有worker nodes启动Ray worker
- 配置所有必要的环境变量
- 验证集群状态

### 2. 启动训练

在master node上运行：

```bash
# 启动双环境训练
bash run_multi_node_training.sh
```

### 3. 监控训练

- **Ray Dashboard**: http://172.26.81.88:8265
- **训练日志**: 查看终端输出
- **轨迹日志**: `./output/trajectories/`
- **WandB日志**: `./output/wandb/`

### 4. 同步WandB日志

在有网络的机器上运行：

```bash
# 启动WandB同步服务
bash start_wandb_sync.sh
```

### 5. 停止集群

训练完成后：

```bash
# 停止整个Ray集群
bash shutdown_cluster.sh
```

## 配置说明

### 集群配置

在 `setup_cluster.sh` 中修改：

```bash
# 集群节点配置
MASTER_IP="172.26.81.88"  # Head node IP
WORKER_NODES=(
    "172.26.85.167" # Worker node 1
    "172.26.87.246" # Worker node 2
    "172.26.91.117" # Worker node 3
)

# 环境配置
CONDA_ENV="roll"  # Conda环境名
```

### 训练配置

在 `agentic_search_train.yaml` 中的关键配置：

```yaml
# 多机多卡配置
num_gpus_per_node: 8
num_nodes: 4  # 根据实际节点数调整

# 训练配置
actor_train:
  device_mapping: list(range(0,32))  # 使用所有GPU
  world_size: 16  # 训练worker数量
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 2
      sequence_parallel: true

# 推理配置
actor_infer:
  device_mapping: list(range(16,32))  # 使用后半部分GPU
  world_size: 8  # 推理worker数量
  num_gpus_per_worker: 2
```

### 异步训练配置

```yaml
# 异步训练设置
async_generation_ratio: 1  # 完全异步训练

# 调度配置
schedule_config:
  generate_opt_level: 1
  max_running_requests: 256
  is_use_additional_prompts: true
  max_additional_running_prompts: 32

# 环境管理器优化
train_env_manager:
  max_env_num_per_worker: 16  # 增加并发
  max_traj_per_env: 4         # 异步轨迹数
```

### 奖励系统配置

```yaml
# Outcome-based奖励
use_outcome_reward_only: true
step_penalty: 0.0
invalid_action_penalty: 0.0

# 高级奖励计算
use_xverify: true  # 使用xverify验证
use_otc: false     # 可选的OTC奖励
```

## 故障排除

### 1. Ray集群问题

```bash
# 检查集群状态
ray status

# 重启集群
bash shutdown_cluster.sh
bash setup_cluster.sh
```

### 2. 网络连接问题

检查节点间网络连通性：

```bash
# 测试节点连通性
ping 172.26.81.88
nc -z 172.26.81.88 6379
```

### 3. GPU资源问题

```bash
# 检查GPU状态
nvidia-smi

# 检查Ray GPU资源
ray status
```

### 4. 环境变量问题

确保所有节点都有正确的环境变量：

```bash
# 检查关键环境变量
echo $PYTHONPATH
echo $WANDB_DIR
echo $ROLL_TRAJECTORY_LOG_DIR
```

## 监控和日志

### Ray Dashboard

访问 http://172.26.81.88:8265 查看：
- 集群资源使用情况
- 任务执行状态
- 节点健康状态

### 训练日志

- **标准输出**: 训练进度和指标
- **轨迹日志**: `./output/trajectories/` (10%采样)
- **WandB日志**: `./output/wandb/` (离线模式)

### 性能监控

```bash
# 查看GPU使用率
watch -n 1 nvidia-smi

# 查看网络流量
iftop -i eth0

# 查看系统资源
htop
```

## 最佳实践

1. **启动顺序**: 先启动Ray集群，再启动训练
2. **资源分配**: 合理分配GPU给训练和推理
3. **网络优化**: 确保节点间高速网络连接
4. **监控**: 定期检查Ray dashboard和系统资源
5. **日志管理**: 定期同步WandB日志和清理本地日志

## 常用命令

```bash
# 集群管理
bash setup_cluster.sh          # 启动集群
bash shutdown_cluster.sh       # 停止集群
ray status                     # 查看集群状态

# 训练管理
bash run_multi_node_training.sh  # 启动训练

# 日志管理
bash start_wandb_sync.sh       # 同步WandB日志
ls ./output/trajectories/      # 查看轨迹日志
ls ./output/wandb/            # 查看WandB日志

# 调试
tmux list-sessions            # 查看tmux会话
tmux attach -t head           # 连接到head node会话
tmux attach -t node0          # 连接到worker node会话
```

## 支持

如遇问题，请检查：
1. Ray集群状态
2. 网络连通性
3. GPU资源可用性
4. 环境变量配置
5. 数据集路径

更多详细信息请参考ROLL官方文档。
