# Numina Math Environment

基于 Numina Math 数据集的数学推理强化学习环境，支持模型通过搜索引擎进行数学问题求解。

## 功能特性

### 1. 数据集支持
- 读取 numinamath_new 数据集 (Parquet 格式)
- 支持数学推理任务，包含系统提示词和数学问题
- 自动解析数据集中的 prompt 结构

### 2. 搜索工具
继承自搜索环境，支持三种搜索工具：
- `web_search(keywords)`: 网络搜索数学相关信息
- `web_parse(link, query)`: 网页内容解析  
- `parse_img(link, query)`: 图片内容分析（数学图表等）

### 3. 数学专用奖励系统
- **传统数学评分**: 基于 prime_math 的符号数学比较
- **xverify 评估**: 使用 LLM 进行更智能的数学答案验证
- **OTC 奖励**: 优化工具调用效率（PPO/GRPO 模式）

### 4. 执行模式
- **Mock 模式**: 使用模拟搜索结果，适合测试和开发
- **HTTP 模式**: 通过远程 HTTP 服务执行搜索，适合生产环境
- **API 模式**: 直接调用搜索 API（可扩展）

## 使用方法

### 基本配置

```python
from roll.agentic.env.numina_math import NuminaMathEnv, NuminaMathEnvConfig

config = NuminaMathEnvConfig(
    dataset_path="/path/to/numinamath_new/train_subset.parquet",
    max_instances=1000,
    use_mock_api=True,  # 测试模式
    use_remote_service=False,  # 不使用远程服务
    max_steps=10,
    max_search_calls=5,
    use_xverify=True,  # 使用 xverify 评估
    use_otc=False  # 不使用 OTC 奖励
)

env = NuminaMathEnv(config)
```

### 数学专用配置

```python
config = NuminaMathEnvConfig(
    dataset_path="/path/to/numinamath_new/train_subset.parquet",
    use_xverify=True,  # 启用 xverify 数学评估
    use_otc=True,      # 启用 OTC 奖励
    otc_method="ppo",  # 使用 PPO 模式
    otc_alpha=1.0,     # OTC 权重
    otc_c=1.0,         # OTC 平滑常数
    xverify_config=None  # 使用默认 xverify 配置
)
```

### Trajectory 格式

环境遵循以下交互模式：

```
observation0: 数学系统提示词 + 数学问题
action0: <think>数学推理过程</think> + <code>搜索相关数学知识</code>
observation1: <execution_results>搜索结果</execution_results>
action1: <think>分析和计算</think> + <answer>数学答案</answer>
observation2: 最终反馈 + 奖励（基于数学评分）
```

### 示例代码

```python
# 重置环境
obs, info = env.reset(seed=42)

# 第一步：搜索数学知识
action = '''<think>
这是一个关于组合数学的问题，我需要搜索相关的理论知识。
</think>

<code>
result = web_search("combinatorics grid coloring constraints")
print(result)
</code>'''

obs, reward, terminated, truncated, info = env.step(action)

# 第二步：数学推理和计算
action = '''<think>
根据搜索结果和约束条件，我可以建立数学模型来求解。
每个单元格最多有两个相邻的黑色单元格，这限制了着色模式。
</think>

<code>
# 计算最优解
max_black_cells = 5 * 100 // 2 + 2  # 基于约束的理论上限
print(f"理论上限: {max_black_cells}")
</code>'''

obs, reward, terminated, truncated, info = env.step(action)

# 第三步：给出最终答案
action = '''<think>
通过分析和计算，最大可能的黑色单元格数量是 302。
</think>

<answer>302</answer>'''

obs, reward, terminated, truncated, info = env.step(action)
```

## 奖励机制

### 基础奖励
- 正确答案：+1.0
- 每步惩罚：-0.1
- 无效动作：-0.2
- 超时惩罚：额外 -0.2

### 数学专用评分
1. **快速字符串匹配**: 直接字符串比较
2. **数值比较**: 处理分数、百分比等数学格式
3. **符号数学比较**: 使用 SymPy 进行数学等价性检查
4. **xverify 评估**: LLM 辅助的智能数学验证

### OTC 奖励（可选）
- **PPO 模式**: 基于工具使用效率的奖励调整
- **GRPO 模式**: 需要提供正确轨迹作为参考

## 测试

### 运行基本测试
```bash
cd /mnt/chensiheng/weiyu/xueban_v2/ROLL
python -m roll.agentic.env.numina_math.env
```

### 创建测试文件
```bash
python -c "
from roll.agentic.env.numina_math import NuminaMathEnv, NuminaMathEnvConfig
config = NuminaMathEnvConfig(max_instances=5, disable_limiter=True)
env = NuminaMathEnv(config)
obs, info = env.reset(seed=42)
print('Environment created successfully!')
print(f'Sample question: {obs[:200]}...')
"
```

## 配置参数

### NuminaMathEnvConfig 主要参数

- `dataset_path`: 数学数据集路径
- `max_instances`: 最大实例数
- `max_steps`: 最大步数
- `max_search_calls`: 最大搜索次数
- `use_xverify`: 是否使用 xverify 数学评估
- `use_otc`: 是否使用 OTC 奖励
- `otc_method`: OTC 方法（"ppo" 或 "grpo"）
- `otc_alpha`: OTC 权重系数
- `otc_c`: OTC 平滑常数
- `xverify_config`: xverify 模型配置

## 与搜索环境的区别

1. **数据集**: 使用 Numina Math 而非搜索数据集
2. **奖励计算**: 专门的数学评分系统
3. **评估方法**: 支持 xverify 和 OTC 等高级评估
4. **问题类型**: 专注于数学推理而非一般问答

## 注意事项

1. 确保数学数据集路径正确
2. xverify 需要额外的模型配置
3. OTC-GRPO 模式需要提供正确轨迹
4. 数学评分比一般文本匹配更复杂，可能需要更多计算时间
5. 建议在开发阶段使用 Mock 模式以提高测试效率
