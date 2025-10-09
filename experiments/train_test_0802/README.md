# Dual Environment Agentic Training

这个目录包含了**双环境训练**的配置和脚本：搜索环境 + 数学环境。

## 🎯 概述

本实验同时在两个不同环境上训练单个模型：
1. **搜索环境**: 基于网络搜索的问答任务 (xhpang_search 数据集)
2. **数学环境**: 带搜索工具的数学推理任务 (numinamath 数据集)

## 📁 文件结构

```
experiments/train_test_0802/
├── agentic_search_train.yaml    # 双环境训练配置文件
├── run_search_training.sh       # 训练启动脚本
├── test_dual_env_config.py      # 双环境配置验证脚本
├── test_config.py              # 传统配置测试脚本
└── README.md                   # 说明文档
```

## ⚙️ 配置详情

### 环境设置
- **训练比例**: 搜索:数学 = 1:1 (64:64 组)
- **验证比例**: 搜索:数学 = 1:1 (32:32 组)
- **速率限制**: 每环境 20,000 请求/分钟 (总计 40,000)
- **异步处理**: 完全异步环境执行

### 主要参数

- **模型**: Qwen/Qwen2.5-0.5B-Instruct
- **最大训练步数**: 512
- **批次大小**: 256 (训练), 128 (验证)
- **序列长度**: 4096
- **学习率**: 1e-6
- **最大轨迹长度**: 8步
- **最大搜索次数**: 5次

### 数据集配置

#### 搜索环境数据集
- **训练**: `/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet` (16.4 MB)
- **验证**: `/mnt/chensiheng/ai_researcher_data/xhpang_search/validation.parquet` (0.3 MB)

#### 数学环境数据集
- **训练**: `/mnt/chensiheng/ai_researcher_data/numinamath_new/train_subset.parquet` (12.0 MB)
- **验证**: `/mnt/chensiheng/ai_researcher_data/numinamath_new/validation_subset.parquet` (0.3 MB)

### 环境配置

#### 训练环境
- **SearchEnvTrain**: 64组 × 4大小 = 256个搜索环境
- **MathEnvTrain**: 64组 × 4大小 = 256个数学环境
- **总计**: 512个训练环境 (1:1比例)

#### 验证环境
- **SearchEnvVal**: 32组 × 1大小 = 32个搜索环境
- **MathEnvVal**: 32组 × 1大小 = 32个数学环境
- **总计**: 64个验证环境 (1:1比例)

### 奖励设置

- 正确答案奖励: +1.0
- 每步惩罚: -0.1
- 无效动作惩罚: -0.2
- 格式惩罚: -0.2

## 使用方法

### 1. 环境准备

确保所有数据集路径正确：
```bash
export XHPANG_TRAIN="/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet"
export NUMINA_TRAIN="/mnt/chensiheng/ai_researcher_data/numinamath_new/train_subset.parquet"
export XHPANG_VAL="/mnt/chensiheng/ai_researcher_data/xhpang_search/validation.parquet"
export NUMINA_VAL="/mnt/chensiheng/ai_researcher_data/numinamath_new/validation_subset.parquet"
```

确保工具服务器运行：
```bash
# 检查工具服务器是否可用
curl -X POST http://172.26.104.240:30002/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "print(\"Hello World\")"}'
```

### 2. 配置测试

运行双环境配置测试脚本：
```bash
cd /mnt/chensiheng/weiyu/xueban_v2/ROLL/experiments/train_test_0802
python test_dual_env_config.py
```

这将验证：
- ✅ 环境注册 (search + numina_math)
- ✅ 配置文件有效性
- ✅ 数据集可访问性
- ✅ 环境创建
- ✅ 1:1比例设置
- ✅ 速率限制配置

### 3. 启动训练

```bash
cd /mnt/chensiheng/weiyu/xueban_v2/ROLL/experiments/train_test_0802
chmod +x run_search_training.sh
./run_search_training.sh
```

或者直接使用Python命令：
```bash
cd /mnt/chensiheng/weiyu/xueban_v2/ROLL
python examples/start_agentic_pipeline.py \
    --config_path experiments/train_test_0802 \
    --config_name agentic_search_train
```

## 训练流程

### 1. 轨迹格式

训练使用以下轨迹格式：
```
观测0: 系统提示 + 用户问题
动作0: <think>推理过程</think> <code>搜索代码</code>
观测1: <execution_results>搜索结果</execution_results>
动作1: <think>分析结果</think> <answer>最终答案</answer>
观测2: 最终反馈
```

### 2. 终止条件

- **正常终止**: 模型输出 `<answer>` 标签
- **搜索限制终止**: 达到最大搜索次数 (5次)
- **步数截断**: 达到最大步数 (8步)

### 3. 动作模式

支持两种动作模式：
- **思考 + 搜索**: `<think>...</think> <code>web_search("query")</code>`
- **思考 + 回答**: `<think>...</think> <answer>答案</answer>`

## 监控和日志

### 输出目录

- 日志: `./output/logs/`
- 模型检查点: `/data/cpfs_0/rl_examples/models/search_agentic_pipeline/`
- 渲染文件: `./output/render/`

### 关键指标

- `success_rate`: 成功率
- `avg_reward`: 平均奖励
- `avg_steps`: 平均步数
- `search_calls`: 平均搜索次数

## 故障排除

### 常见问题

1. **Ray内存错误**
   - 确保 `disable_limiter: false` 在生产环境中
   - 调整 `num_env_groups` 和 `group_size`

2. **工具服务器连接失败**
   - 检查 `remote_service_url` 配置
   - 确认服务器状态

3. **数据集加载失败**
   - 验证数据集路径
   - 检查文件权限

### 调试模式

启用mock模式进行调试：
```yaml
env_config:
  use_mock_api: true
  use_remote_service: false
```

## 性能优化

### GPU使用

- 训练: 8 GPUs (Megatron并行)
- 推理: 8 GPUs (VLLM)
- 参考模型: 8 GPUs

### 批次大小调整

根据GPU内存调整：
```yaml
actor_train:
  training_args:
    per_device_train_batch_size: 1  # 减少以节省内存
    gradient_accumulation_steps: 32  # 增加以保持有效批次大小
```

## 扩展配置

### 添加新的搜索工具

在环境配置中添加新的工具函数，修改 `utils.py` 中的执行逻辑。

### 调整奖励函数

修改配置文件中的奖励参数：
```yaml
env_config:
  correct_answer_reward: 2.0    # 增加正确答案奖励
  step_penalty: -0.05           # 减少步数惩罚
```

### 多数据集训练

添加更多环境配置：
```yaml
custom_envs:
  SearchEnvTrain2:
    env_type: search
    env_config:
      dataset_path: "/path/to/another/dataset.parquet"
```
