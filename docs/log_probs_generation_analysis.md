# ROLL框架中Log Probabilities的生成流程分析

## 核心发现

**关键结论**：在ROLL框架中，`old_log_probs`是在**输入和输出拼接后**重新计算的，而不是在推理过程中实时计算的。这意味着计算的log probabilities可能与生成时的实际概率存在差异。

## 详细分析

### 1. 推理引擎的生成流程

#### 1.1 vLLM/SGLang的职责
- **仅负责生成tokens**：推理引擎（vLLM、SGLang）只生成输出tokens
- **不计算log_probs**：虽然vLLM支持返回log probabilities（通过设置`logprobs`参数），但ROLL将其设置为0，不获取
- **返回token序列**：只返回生成的token IDs

```python
# vllm_strategy.py中的采样参数
SamplingParams(
    max_tokens=gen_kwargs["max_new_tokens"],
    temperature=gen_kwargs["temperature"],
    # ...
    logprobs=0,  # 关键：设置为0，不返回log probabilities
)
```

#### 1.2 生成后的拼接操作
生成完成后，ROLL会将输入和输出进行拼接：

```python
# roll/utils/functionals.py
def concatenate_input_and_output(input_ids, output_ids, num_return_sequences):
    batch_size, input_seq_len = input_ids.size()
    _, output_seq_len = output_ids.size()
    repeated_input_ids = (
        input_ids.unsqueeze(1)
        .repeat(1, num_return_sequences, 1)
        .view(batch_size * num_return_sequences, input_seq_len)
    )
    sequences = torch.cat((repeated_input_ids, output_ids), dim=1)
    return sequences
```

### 2. Log Probabilities的计算流程

#### 2.1 Actor-Train的重新计算
拼接后的完整序列会被发送给Actor-Train进行log probabilities计算：

```python
# base_worker.py - compute_log_probs方法
def compute_log_probs(self, data: DataProto):
    # 使用训练框架（DeepSpeed/FSDP）进行前向传播
    results = self.strategy.forward_step(
        batch=data, forward_func=self.forward_func_log_probs
    )
    
def forward_func_log_probs(self, data: DataProto, output_tensor: torch.Tensor):
    # 基于模型输出的logits计算log_probs
    log_probs = self.strategy.op_compute_log_probs(
        logits=output_tensor, 
        input_ids=data.batch["input_ids"], 
        attention_mask=data.batch["response_mask"]
    )
```

#### 2.2 具体计算方式
```python
# strategy.py - op_compute_log_probs
def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    # 使用next-token预测
    labels: torch.Tensor = input_ids[:, 1:].clone()
    labels[attention_mask[:, 1:] == 0] = 0
    log_probs = log_probs_from_logits(logits[:, :-1], labels)
    log_probs = log_probs * attention_mask[:, 1:]
    return log_probs

# functionals.py - log_probs_from_logits
def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)
```

### 3. 潜在的问题

#### 3.1 概率差异
由于是拼接后重新计算，可能存在以下差异：
1. **位置编码差异**：生成时是逐token的，重新计算时是整个序列
2. **注意力模式差异**：重新计算时可以"看到"完整序列
3. **数值精度差异**：不同框架（推理vs训练）的计算精度

#### 3.2 为什么这样设计？
1. **架构解耦**：推理引擎专注于高效生成，训练框架负责梯度计算
2. **内存效率**：推理引擎不需要保存中间激活值
3. **兼容性**：支持多种推理引擎，无需每个都实现log_probs计算

### 4. 对PPO训练的影响

在PPO中，`old_log_probs`用于计算重要性采样比率：
```python
ratio = (log_probs - old_log_probs).exp()
```

由于`old_log_probs`是重新计算的，这个比率反映的是：
- **分子**：当前策略对动作的评估
- **分母**：当前策略对之前生成的动作的重新评估（而非生成时的真实概率）

这种差异在大多数情况下可能很小，但在以下情况可能显著：
1. 模型权重更新较大时
2. 长序列生成时（累积误差）
3. 使用特殊的采样策略时

### 5. 可能的改进方向

1. **推理时计算log_probs**：
   - 修改vLLM/SGLang配置，启用logprobs返回
   - 存储生成时的真实log probabilities

2. **缓存机制**：
   - 在推理引擎中缓存生成时的中间状态
   - 支持精确重现生成时的计算

3. **混合方案**：
   - 对关键token保存真实log_probs
   - 对其他token使用重新计算的值

## 结论

ROLL框架当前的实现通过拼接后重新计算log probabilities，这是一种工程上的权衡：
- **优点**：架构清晰、易于维护、支持多种推理引擎
- **缺点**：计算的log_probs不是生成时的真实值，可能影响PPO的理论保证

对于大多数应用场景，这种差异可能是可以接受的，但在追求高精度的场景下，可能需要考虑改进方案。
