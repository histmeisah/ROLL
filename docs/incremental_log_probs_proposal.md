# 增量式Log Probabilities计算方案

## 1. 问题背景

### 1.1 当前实现的问题

当前ROLL框架在计算`old_log_probs`时存在以下问题：

```python
# 当前方式：知道整条轨迹后一次性计算
input_ids = [prompt + response]  # 完整序列
log_probs = model.forward(input_ids)  # 一次性前向传播
```

**核心问题**：
1. **因果性违背**：模型在计算position t的log_prob时，可以"看到"t+1, t+2...的未来信息
2. **注意力泄露**：self-attention可以关注到生成时不存在的未来token
3. **不符合自回归生成**：违背了`P(x_t|x_<t)`的条件概率定义

### 1.2 理论影响

在PPO中，重要性采样需要：
```
π_old(a_t|s_t) / π_old(a_t|s_t) = 1  (当策略未更新时)
```

但当前实现计算的是：
```
π_old(a_t|s_t, a_>t) ≠ π_old(a_t|s_t)
```

这导致即使策略未更新，ratio也可能不等于1，引入系统性偏差。

## 2. 改进方案：增量式计算

### 2.1 核心思想

模仿实际生成过程，逐步计算每个位置的log_prob：

```python
# 期望的方式：逐步增量计算
log_probs = []
for t in range(len(response)):
    # 只使用到position t为止的信息
    partial_ids = prompt + response[:t+1]
    logits_t = model.forward(partial_ids)[-1]  # 只取最后一个位置
    log_prob_t = log_softmax(logits_t)[response[t]]
    log_probs.append(log_prob_t)
```

### 2.2 为什么这更符合Off-Policy

1. **真实反映生成概率**：
   - On-policy: `π_current(a_t|s_t)`使用当前策略
   - Off-policy: `π_old(a_t|s_t)`使用生成时的历史策略
   - 增量计算确保两者使用相同的条件信息

2. **保持马尔可夫性质**：
   - 决策只依赖当前状态：`P(a_t|s_t)`
   - 不依赖未来信息：`P(a_t|s_t, s_>t)` ❌

3. **理论保证**：
   - PPO的收敛性依赖于准确的重要性采样
   - 增量计算提供无偏的ratio估计

## 3. 实现方案

### 3.1 方案一：KV Cache增量计算（推荐）

利用Transformer的KV cache机制，避免重复计算：

```python
class IncrementalLogProbsComputer:
    def compute_log_probs_incremental(self, input_ids, prompt_length):
        """
        增量计算log probabilities，模拟自回归生成过程
        
        Args:
            input_ids: 完整序列 [batch_size, seq_len]
            prompt_length: prompt长度，之后都是response
        """
        batch_size, seq_len = input_ids.shape
        response_length = seq_len - prompt_length
        
        # 初始化KV cache
        past_key_values = None
        log_probs = []
        
        # 1. 处理prompt部分（可以并行）
        prompt_ids = input_ids[:, :prompt_length]
        with torch.no_grad():
            outputs = self.model(
                input_ids=prompt_ids,
                use_cache=True,
                return_dict=True
            )
            past_key_values = outputs.past_key_values
        
        # 2. 逐token处理response（模拟生成过程）
        for t in range(response_length):
            current_position = prompt_length + t
            current_token = input_ids[:, current_position:current_position+1]
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                
                # 计算当前token的log_prob
                logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
                log_probs_t = F.log_softmax(logits, dim=-1)
                
                # 获取实际token的log_prob
                if t < response_length - 1:
                    next_token = input_ids[:, current_position + 1]
                    token_log_prob = log_probs_t.gather(1, next_token.unsqueeze(1))
                    log_probs.append(token_log_prob)
                
                # 更新KV cache
                past_key_values = outputs.past_key_values
        
        # 拼接所有log_probs
        return torch.cat(log_probs, dim=1)  # [batch_size, response_length-1]
```

### 3.2 方案二：批量Padding计算

如果KV cache不可用，可以使用padding技巧：

```python
def compute_log_probs_with_causal_mask(self, input_ids, prompt_length):
    """
    使用自定义attention mask确保因果性
    """
    batch_size, seq_len = input_ids.shape
    
    # 创建多个截断版本的输入
    all_log_probs = []
    
    for t in range(prompt_length, seq_len):
        # 创建截断到position t的输入
        truncated_ids = input_ids.clone()
        truncated_ids[:, t+1:] = self.pad_token_id
        
        # 创建相应的attention mask
        attention_mask = torch.ones_like(truncated_ids)
        attention_mask[:, t+1:] = 0
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=truncated_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            logits = outputs.logits[:, t, :]
            log_probs = F.log_softmax(logits, dim=-1)
            
            if t < seq_len - 1:
                next_token = input_ids[:, t + 1]
                token_log_prob = log_probs.gather(1, next_token.unsqueeze(1))
                all_log_probs.append(token_log_prob)
    
    return torch.cat(all_log_probs, dim=1)
```

### 3.3 方案三：推理时保存（最准确但需要架构改动）

修改vLLM/SGLang，在生成时保存log_probs：

```python
# 修改vLLM的SamplingParams
sampling_params = SamplingParams(
    max_tokens=max_new_tokens,
    temperature=temperature,
    logprobs=1,  # 启用log_probs返回
    # ...
)

# 在生成时提取并保存
class PolicyProxy:
    def generate(self, messages, lm_input, generation_config):
        # ... 现有代码 ...
        
        # 提取log_probs
        log_probs = []
        for completion_output in request_output.outputs:
            if completion_output.logprobs:
                token_logprobs = [
                    logprob_dict[token_id] 
                    for token_id, logprob_dict in zip(
                        completion_output.token_ids,
                        completion_output.logprobs
                    )
                ]
                log_probs.append(token_logprobs)
        
        # 保存到输出
        lm_output.batch["generation_log_probs"] = log_probs
```

## 4. 集成到ROLL框架

### 4.1 修改compute_log_probs方法

```python
# base_worker.py
def compute_log_probs(self, data: DataProto):
    """增强版compute_log_probs，支持增量计算"""
    # 检查是否需要增量计算
    use_incremental = data.meta_info.get("use_incremental_log_probs", True)
    
    if use_incremental and "prompt_length" in data.batch:
        # 使用增量计算
        log_probs = self.compute_log_probs_incremental(
            input_ids=data.batch["input_ids"],
            prompt_length=data.batch["prompt_length"]
        )
    else:
        # 降级到原有方式
        log_probs = self.compute_log_probs_original(data)
    
    return DataProto.from_dict(tensors={"log_probs": log_probs})
```

### 4.2 修改Pipeline集成

```python
# agentic_pipeline.py
def store_fresh_data_to_replay_buffer(self, fresh_batch: DataProto, global_step: int):
    """存储数据时使用增量计算"""
    try:
        # 标记使用增量计算
        fresh_batch.meta_info["use_incremental_log_probs"] = True
        
        # 添加prompt长度信息
        fresh_batch.batch["prompt_length"] = fresh_batch.batch["prompt_mask"].sum(dim=1)
        
        # 计算behavior_log_probs
        behavior_refs = self.actor_train.compute_log_probs(fresh_batch, blocking=False)
        behavior = DataProto.materialize_concat(data_refs=behavior_refs)
        
        if behavior.batch is not None and "log_probs" in behavior.batch:
            fresh_batch.batch["behavior_log_probs"] = behavior.batch["log_probs"]
    except Exception as e:
        logger.warning(f"Failed to compute incremental log_probs: {e}")
        # 降级处理
```

## 5. 性能优化

### 5.1 批处理优化

对于多个序列，可以并行处理相同长度的部分：

```python
def compute_log_probs_batch_optimized(self, batch_input_ids, batch_prompt_lengths):
    """批量优化的增量计算"""
    # 按response长度分组
    length_groups = defaultdict(list)
    for i, (ids, prompt_len) in enumerate(zip(batch_input_ids, batch_prompt_lengths)):
        response_len = len(ids) - prompt_len
        length_groups[response_len].append(i)
    
    # 每组并行处理
    all_log_probs = {}
    for response_len, indices in length_groups.items():
        group_ids = batch_input_ids[indices]
        group_prompt_lens = batch_prompt_lengths[indices]
        
        # 并行计算该组
        group_log_probs = self._compute_group_log_probs(
            group_ids, group_prompt_lens, response_len
        )
        
        for idx, log_probs in zip(indices, group_log_probs):
            all_log_probs[idx] = log_probs
    
    # 按原始顺序返回
    return [all_log_probs[i] for i in range(len(batch_input_ids))]
```

### 5.2 缓存策略

```python
class LogProbsCache:
    """缓存已计算的log_probs，避免重复计算"""
    def __init__(self, capacity=10000):
        self.cache = LRUCache(capacity)
    
    def get_or_compute(self, input_ids, prompt_length, compute_fn):
        # 生成缓存key
        cache_key = self._generate_key(input_ids, prompt_length)
        
        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 计算并缓存
        log_probs = compute_fn(input_ids, prompt_length)
        self.cache[cache_key] = log_probs
        return log_probs
```

## 6. 预期效果

### 6.1 理论改进
- **无偏估计**：准确反映生成时的概率分布
- **稳定训练**：减少因ratio偏差导致的训练不稳定
- **理论保证**：符合PPO的理论假设

### 6.2 实际影响
- **短序列**：影响较小，性能开销可接受
- **长序列**：显著改善，避免累积误差
- **复杂任务**：提高样本效率和最终性能

## 7. 实施建议

1. **分阶段实施**：
   - Phase 1: 实现KV cache版本，在小规模实验中验证
   - Phase 2: 优化性能，处理边界情况
   - Phase 3: 集成到主流程，提供配置选项

2. **兼容性保持**：
   - 提供配置开关，允许切换新旧方式
   - 保留原有API，确保向后兼容

3. **监控指标**：
   - 比较新旧方式的log_probs差异
   - 监控训练稳定性和收敛速度
   - 评估性能开销

## 8. 总结

增量式log_probs计算是对ROLL框架的重要改进，它：
- 提供理论上正确的off-policy概率估计
- 改善PPO训练的稳定性和效率
- 为未来更复杂的RL算法奠定基础

虽然实现上需要一定的工程努力，但这是值得的投资，特别是对于追求高质量RL训练的场景。
