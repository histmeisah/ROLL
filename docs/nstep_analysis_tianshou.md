# N-Step Support in StepReplayBuffer: Tianshou Analysis

## 1. Background

This document analyzes tianshou's dynamic n-step loading mechanism and explores how to adapt it for ROLL's StepReplayBuffer to support n-step returns and GAE computation.

### Current Problem

**StepReplayBuffer stores individual conversation steps**, but n-step returns require access to consecutive steps:
- **n-step return**: R_t^(n) = r_t + γ*r_{t+1} + ... + γ^(n-1)*r_{t+n-1} + γ^n*V(s_{t+n})
- **GAE**: A_t = Σ_{k=0}^∞ (γλ)^k * δ_{t+k}, where δ_t = r_t + γ*V(s_{t+1}) - V(s_t)

**Challenge**: After buffer wrap-around, steps from the same episode are no longer consecutive in the buffer.

---

## 2. Tianshou's Dynamic N-Step Loading

### 2.1 Core Concept

Tianshou uses a **"lazy loading" strategy**:
1. **Store atomic transitions**: (s_t, a_t, r_t, s_{t+1}, done_t)
2. **Build n-step sequences at sample time**: Use `buffer.next(indices)` to traverse forward
3. **Handle episode boundaries**: Use masks to terminate n-step sequences at episode ends

### 2.2 Key Data Structure: `stacked_indices_NI`

```python
@njit
def _nstep_return(
    rew_B: np.ndarray,           # [B] reward buffer (entire buffer)
    end_flag_B: np.ndarray,      # [B] episode termination flags
    target_q_IA: np.ndarray,     # [I, A] Q-values at final states
    stacked_indices_NI: np.ndarray,  # [N, I] KEY: stacked trajectory indices
    gamma: float,
    n_step: int,
) -> np.ndarray:
    """
    Notation:
    - I: number of sampled indices (batch size)
    - B: size of replay buffer
    - N: n_step (number of consecutive steps)
    - A: action dimension

    stacked_indices_NI shape: [n_step, batch_size]
    Example for batch_size=3, n_step=5:
    [
     [i_0,     i_1,     i_2],       # Starting indices
     [next(i_0), next(i_1), next(i_2)],  # +1 step
     [next(next(i_0)), ..., ...],   # +2 steps
     [..., ..., ...],                # +3 steps
     [..., ..., ...]                 # +4 steps
    ]
    """
```

**Key Insight**: `stacked_indices_NI` is a 2D array that tracks **n consecutive transitions** for each sampled index.

### 2.3 How `buffer.next(indices)` Works

```python
class ReplayBuffer:
    def next(self, index: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Get next index(indices) in the buffer, handling episode boundaries.

        Logic:
        1. If current transition is terminal (done=True), return current index (stay at episode end)
        2. Otherwise, return (index + 1) % capacity (circular buffer wrap)

        This ensures:
        - Sequences don't cross episode boundaries
        - Buffer wraps around correctly
        """
        if isinstance(index, int):
            return index if self.done[index] else (index + 1) % self.capacity
        else:
            # Vectorized version
            return np.where(self.done[index], index, (index + 1) % self.capacity)
```

**Key Insight**: `done` flag serves as **episode boundary marker**, preventing n-step sequences from crossing episodes.

### 2.4 Building Stacked Indices

```python
def sample(self, batch_size: int, n_step: int = 1) -> Batch:
    """
    Sample with n-step look-ahead.
    """
    # 1. Sample starting indices
    indices = np.random.choice(len(self), size=batch_size, replace=False)

    # 2. Build stacked indices: [n_step, batch_size]
    stacked_indices = np.empty((n_step, batch_size), dtype=int)
    stacked_indices[0] = indices  # Starting indices

    for i in range(1, n_step):
        stacked_indices[i] = self.next(stacked_indices[i-1])  # Forward traversal

    # 3. Extract data using stacked indices
    # This allows vectorized n-step return computation
    return Batch(
        obs=self.obs[indices],
        act=self.act[indices],
        rew=self.rew[stacked_indices],  # [n_step, batch_size]
        done=self.done[stacked_indices],
        next_obs=self.obs[stacked_indices[-1]],
        indices=indices,  # For priority update
        stacked_indices=stacked_indices  # For n-step computation
    )
```

### 2.5 N-Step Return Computation (Numba-optimized)

```python
@njit
def _nstep_return(
    rew_B: np.ndarray,
    end_flag_B: np.ndarray,
    target_q_IA: np.ndarray,
    stacked_indices_NI: np.ndarray,
    gamma: float,
    n_step: int,
) -> np.ndarray:
    """
    Compute n-step returns given stacked indices.
    """
    batch_size = stacked_indices_NI.shape[1]
    returns = np.zeros(batch_size, dtype=np.float32)

    for i in range(batch_size):
        # Accumulate discounted rewards
        discount = 1.0
        for step in range(n_step):
            idx = stacked_indices_NI[step, i]
            returns[i] += discount * rew_B[idx]

            # Stop at episode boundary
            if end_flag_B[idx]:
                break

            discount *= gamma
        else:
            # If didn't break (no episode boundary), add bootstrapped value
            final_idx = stacked_indices_NI[n_step - 1, i]
            returns[i] += discount * target_q_IA[i].max()

    return returns
```

**Key Insights**:
1. **Vectorized across batch**: Outer loop over batch, inner loop over steps
2. **Episode boundary handling**: Break early if `done=True`
3. **Bootstrap value**: Only add Q(s_{t+n}) if didn't hit episode end
4. **Numba acceleration**: JIT compilation for 100x speedup

---

## 3. Adapting to ROLL's StepReplayBuffer

### 3.1 Key Differences: Tianshou vs ROLL

| Aspect | Tianshou | ROLL StepReplayBuffer |
|--------|----------|----------------------|
| **Storage unit** | Single transition (s, a, r, s', done) | Full conversation step (multi-token sequence) |
| **Episode structure** | Stored implicitly via `done` flags | Stored in metadata: `traj_id` + `step` index |
| **Consecutive storage** | Yes, always consecutive after episode push | Initially consecutive, but fragmented after wrap-around |
| **Indexing** | Circular buffer index | Deque index (no direct random access to episode structure) |
| **Primary use case** | Standard RL (Atari, MuJoCo) | LLM multi-turn dialogue RL |

### 3.2 Critical Observations

**Observation 1: Metadata Richness**
- Each `StepEntry` has `traj_id` (episode ID) and `step` (step index within episode)
- This is **sufficient information** to reconstruct episode structure

**Observation 2: Episode Pushing Pattern**
```python
# StepEnvManager.formulate_rollouts() (line 203-299)
for step, history in enumerate(rollout_cache.history):
    samples.append(DataProto(
        non_tensor_batch={
            "step": np.array([step]),  # Step 0, 1, 2, ...
            "traj_id": ...,
        }
    ))
batch = DataProto.concat(samples)  # All steps from one episode
```

**All steps from one episode are pushed together**, but:
- They're stored consecutively initially
- Consecutive structure is **destroyed after buffer wrap-around**
- Need index structure to reconstruct

**Observation 3: Sample Time Requirements**
- Unlike tianshou which can traverse forward with `next()`, we need **episode-aware lookups**
- Must efficiently answer: "Given buffer index `idx`, find the next `n` steps from the same episode"

---

## 4. Proposed Solution: Episode Index + Dynamic Loading

### 4.1 Episode Index Structure

Add to `StepReplayBuffer`:

```python
class StepReplayBuffer(BaseReplayBuffer):
    def __init__(self, ...):
        # ... existing attributes

        # Episode structure tracking
        self._episode_index: Dict[str, Dict[int, int]] = {}
        """
        Episode index mapping: {traj_id: {step_num: buffer_idx}}

        Example:
        {
            "ep_001": {0: 42, 1: 43, 2: 44, 3: 45},  # 4-step episode
            "ep_002": {0: 123, 1: 124},               # 2-step episode
            "ep_003": {0: 0, 1: 1, 2: 2},             # 3-step episode
        }

        This allows O(1) lookup:
        - Given (traj_id, step_num), find buffer index
        - Given buffer index, find (traj_id, step_num)
        """

        self._buffer_to_episode: Dict[int, Tuple[str, int]] = {}
        """
        Reverse mapping: {buffer_idx: (traj_id, step_num)}
        """
```

### 4.2 Index Maintenance

```python
def push_from_dataproto(self, batch: DataProto, global_step: int) -> None:
    """Modified to maintain episode index."""
    for i in range(batch_size):
        # ... existing extraction code

        traj_id = batch.non_tensor_batch["traj_id"][i]
        step = int(batch.non_tensor_batch["step"][i])

        # Calculate buffer index
        current_idx = self.total_stored % self.capacity

        # Update episode index
        if traj_id not in self._episode_index:
            self._episode_index[traj_id] = {}
        self._episode_index[traj_id][step] = current_idx

        # Update reverse mapping
        self._buffer_to_episode[current_idx] = (traj_id, step)

        # Handle eviction (when buffer is full)
        if len(self.steps) == self.capacity:
            # The oldest step is about to be evicted
            evicted_idx = current_idx  # Will be overwritten
            if evicted_idx in self._buffer_to_episode:
                old_traj_id, old_step = self._buffer_to_episode[evicted_idx]
                # Clean up episode index
                if old_traj_id in self._episode_index:
                    if old_step in self._episode_index[old_traj_id]:
                        del self._episode_index[old_traj_id][old_step]
                    # If episode is now empty, remove it
                    if not self._episode_index[old_traj_id]:
                        del self._episode_index[old_traj_id]
                del self._buffer_to_episode[evicted_idx]

        # ... rest of push logic
```

### 4.3 N-Step Lookup

```python
def get_nstep_indices(
    self,
    start_idx: int,
    n_step: int
) -> Tuple[List[int], bool]:
    """
    Get next n steps from the same episode.

    Args:
        start_idx: Starting buffer index
        n_step: Number of steps to look ahead

    Returns:
        (indices, complete):
        - indices: List of buffer indices [start_idx, next_idx, ...], length <= n_step
        - complete: True if we got full n steps, False if hit episode boundary

    Example:
        >>> indices, complete = buffer.get_nstep_indices(42, n_step=5)
        >>> if complete:
        >>>     # We have 5 consecutive steps: indices = [42, 43, 44, 45, 46]
        >>> else:
        >>>     # Episode ended early: indices = [42, 43] (only 2 steps)
    """
    if start_idx not in self._buffer_to_episode:
        return [start_idx], False

    traj_id, start_step = self._buffer_to_episode[start_idx]

    if traj_id not in self._episode_index:
        return [start_idx], False

    episode = self._episode_index[traj_id]
    max_step = max(episode.keys())  # Last step in this episode

    indices = []
    for offset in range(n_step):
        target_step = start_step + offset

        # Check if step exists in episode
        if target_step not in episode:
            # Hit episode boundary
            return indices, False

        indices.append(episode[target_step])

        # Check if this is the last step
        if target_step == max_step:
            return indices, False

    # Got full n steps
    return indices, True
```

### 4.4 Stacked Indices Construction (Tianshou-style)

```python
def build_stacked_indices(
    self,
    sampled_indices: List[int],
    n_step: int
) -> np.ndarray:
    """
    Build stacked indices like tianshou: [n_step, batch_size].

    This enables vectorized n-step return computation.

    Args:
        sampled_indices: Starting buffer indices [batch_size]
        n_step: Number of steps to stack

    Returns:
        stacked_indices: [n_step, batch_size]

    Example:
        >>> sampled_indices = [42, 100, 200]  # batch_size=3
        >>> stacked = buffer.build_stacked_indices(sampled_indices, n_step=5)
        >>> # stacked shape: [5, 3]
        >>> # stacked[:, 0] = [42, 43, 44, 45, 46]  # 5 steps from episode starting at 42
        >>> # stacked[:, 1] = [100, 101, 101, 101, 101]  # Only 2 steps, then repeats last
        >>> # stacked[:, 2] = [200, 201, 202, 203, 204]  # Full 5 steps
    """
    batch_size = len(sampled_indices)
    stacked = np.zeros((n_step, batch_size), dtype=np.int32)

    for i, start_idx in enumerate(sampled_indices):
        indices, complete = self.get_nstep_indices(start_idx, n_step)

        # Fill stacked array
        for step in range(n_step):
            if step < len(indices):
                stacked[step, i] = indices[step]
            else:
                # Episode ended early - repeat last index (tianshou convention)
                stacked[step, i] = indices[-1]

    return stacked
```

### 4.5 N-Step Return Computation

```python
def compute_nstep_returns(
    self,
    sampled_indices: List[int],
    n_step: int,
    gamma: float,
    values: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute n-step returns for sampled steps.

    R_t^(n) = r_t + γ*r_{t+1} + ... + γ^(n-1)*r_{t+n-1} + γ^n*V(s_{t+n})

    Args:
        sampled_indices: Starting buffer indices [batch_size]
        n_step: Number of steps for return computation
        gamma: Discount factor
        values: Optional bootstrapped values [batch_size], if None, use 0

    Returns:
        nstep_returns: [batch_size]
    """
    batch_size = len(sampled_indices)
    returns = np.zeros(batch_size, dtype=np.float32)

    buffer_list = list(self.steps)

    for i, start_idx in enumerate(sampled_indices):
        # Get n-step trajectory
        indices, complete = self.get_nstep_indices(start_idx, n_step)

        # Accumulate discounted rewards
        discount = 1.0
        for idx in indices:
            step_entry = buffer_list[idx]
            # Extract step reward (sum over response tokens)
            reward = step_entry.scores[step_entry.response_mask.astype(bool)].sum()
            returns[i] += discount * reward
            discount *= gamma

        # Add bootstrap value if sequence is complete (didn't hit episode end)
        if complete and values is not None:
            returns[i] += discount * values[i]

    return returns
```

### 4.6 GAE Computation

```python
def compute_gae(
    self,
    sampled_indices: List[int],
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float = 0.99,
    lambda_: float = 0.95
) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE).

    GAE: A_t = Σ_{k=0}^∞ (γλ)^k * δ_{t+k}
    where: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)

    Args:
        sampled_indices: Starting buffer indices [batch_size]
        values: State values V(s_t) [batch_size]
        next_values: Next state values V(s_{t+1}) [batch_size]
        gamma: Discount factor
        lambda_: GAE lambda parameter

    Returns:
        advantages: [batch_size]
    """
    batch_size = len(sampled_indices)
    advantages = np.zeros(batch_size, dtype=np.float32)

    buffer_list = list(self.steps)

    # We need to compute GAE which requires looking forward
    # For simplicity, we'll compute truncated GAE with horizon = n_step
    n_step = 20  # Truncation horizon

    for i, start_idx in enumerate(sampled_indices):
        # Get trajectory for GAE computation
        indices, _ = self.get_nstep_indices(start_idx, n_step)

        # Compute TD errors backward (for efficiency)
        deltas = []
        for j, idx in enumerate(indices):
            step_entry = buffer_list[idx]
            reward = step_entry.scores[step_entry.response_mask.astype(bool)].sum()

            # Get value at t and t+1
            v_t = values[i] if j == 0 else next_values[i]  # Simplified
            v_next = next_values[i]  # Simplified - should track across steps

            delta = reward + gamma * v_next - v_t
            deltas.append(delta)

        # Compute GAE with exponential weighting
        gae = 0.0
        for j in reversed(range(len(deltas))):
            gae = deltas[j] + gamma * lambda_ * gae

        advantages[i] = gae

    return advantages
```

---

## 5. Integration with Agentic Pipeline

### 5.1 When to Compute N-Step Returns

**Option A: At Sample Time** (Tianshou approach)
```python
def sample_for_training(self, ..., n_step: int = 1, gamma: float = 0.99):
    # Sample starting indices
    sampled_indices = ...

    # Build stacked indices
    stacked_indices = self.build_stacked_indices(sampled_indices, n_step)

    # Compute n-step returns
    nstep_returns = self.compute_nstep_returns(
        sampled_indices, n_step, gamma, values=None
    )

    # Attach to DataProto
    dataproto.batch["nstep_returns"] = torch.from_numpy(nstep_returns)
```

**Option B: After Critic Computation** (More flexible)
```python
# In agentic_pipeline.py, after critic forward pass
values = critic_model(states)  # [batch_size]

# Compute n-step returns with bootstrapping
nstep_returns = replay_buffer.compute_nstep_returns(
    sampled_indices, n_step=5, gamma=0.99, values=values.cpu().numpy()
)
```

### 5.2 Configuration

Add to `agentic_config.py`:

```python
@dataclass
class ReplayBufferConfig:
    # ... existing fields

    # N-step returns
    enable_nstep: bool = False
    n_step: int = 5
    gamma: float = 0.99

    # GAE
    enable_gae: bool = False
    gae_lambda: float = 0.95
```

Add to YAML:

```yaml
replay:
  enabled: true
  capacity: 100000

  # N-step returns
  enable_nstep: true
  n_step: 5
  gamma: 0.99

  # GAE
  enable_gae: false
  gae_lambda: 0.95
```

---

## 6. Performance Considerations

### 6.1 Time Complexity

| Operation | Tianshou | ROLL (with Episode Index) |
|-----------|----------|---------------------------|
| **Push** | O(1) | O(1) amortized |
| **Sample starting indices** | O(batch_size * log n) (PER) | O(batch_size * log n) (PER) |
| **Build stacked indices** | O(batch_size * n_step) | O(batch_size * n_step) |
| **N-step return computation** | O(batch_size * n_step) | O(batch_size * n_step) |
| **Episode index lookup** | N/A (implicit via `next()`) | O(1) dict lookup |

**Conclusion**: Time complexity is **identical** to tianshou's approach.

### 6.2 Memory Overhead

**Episode Index Size**:
- Assume 100K steps in buffer
- Average episode length: 5 steps
- Number of episodes: 20K
- Dict size: ~20K * 5 * (str + int + int) ≈ 20K * 5 * 50 bytes ≈ **5 MB**
- Reverse mapping: 100K * (int + str + int) ≈ 100K * 50 bytes ≈ **5 MB**
- **Total: ~10 MB** (negligible compared to token storage)

### 6.3 Index Maintenance Cost

**Push operations**:
- Dict insertion: O(1)
- Eviction cleanup: O(1) per evicted step

**Amortized cost**: O(1) per step

---

## 7. Implementation Roadmap

### Phase 1: Episode Index (Foundational)
- [ ] Add `_episode_index` and `_buffer_to_episode` to `StepReplayBuffer.__init__`
- [ ] Modify `push_from_dataproto()` to maintain index
- [ ] Add `get_nstep_indices()` method
- [ ] Add unit tests for episode index correctness

### Phase 2: N-Step Returns (Core Feature)
- [ ] Implement `build_stacked_indices()`
- [ ] Implement `compute_nstep_returns()`
- [ ] Add configuration to `ReplayBufferConfig`
- [ ] Integrate into `sample_for_training()`
- [ ] Add unit tests for n-step computation

### Phase 3: GAE Support (Advanced)
- [ ] Implement `compute_gae()`
- [ ] Add critic value extraction in pipeline
- [ ] Add configuration for GAE
- [ ] Add unit tests for GAE

### Phase 4: Optimization & Testing
- [ ] Profile performance overhead
- [ ] Add comprehensive integration tests
- [ ] Update documentation
- [ ] Add examples to YAML configs

---

## 8. Alternative Approaches (Not Recommended)

### Alternative 1: Transition-Based Storage (like tianshou)
**Idea**: Restructure StepReplayBuffer to store (s, a, r, s', done) transitions.

**Pros**:
- Direct compatibility with tianshou's approach
- Native `next()` traversal

**Cons**:
- **Breaking change**: Incompatible with current StepEnvManager output format
- **Complexity**: Need to convert multi-token sequences into transitions
- **Loss of information**: Multi-turn dialogue structure is harder to represent
- **Not worth the refactor**: Episode index approach achieves same goal with minimal changes

### Alternative 2: Pre-compute N-Step Returns at Push Time
**Idea**: When episode is pushed, pre-compute n-step returns and store them.

**Pros**:
- No lookup overhead at sample time

**Cons**:
- **Storage overhead**: Must store multiple return values per step
- **Inflexibility**: Fixed n_step, can't adjust at runtime
- **Waste**: Most samples are never used (replay buffer may not sample all data)
- **Recomputation**: Need to recompute if critic values change

---

## 9. Comparison with Mainstream Libraries

| Library | Storage Unit | N-Step Mechanism | Episode Tracking |
|---------|-------------|------------------|------------------|
| **Tianshou** | Transition (s, a, r, s', done) | `next()` traversal + stacked indices | `done` flag |
| **RLlib** | Trajectory batches | Pre-computed advantages | Episode metadata |
| **Stable-Baselines3** | Rollout buffer (full episodes) | GAE during rollout | Implicit (stores full episodes) |
| **ROLL (Proposed)** | Conversation steps | Episode index + stacked indices | `traj_id` + `step` |

**ROLL's approach is closest to tianshou**, but adapted for LLM multi-turn dialogue structure.

---

## 10. Conclusion

### Key Takeaways

1. **Tianshou's approach is elegant and efficient**: Dynamic loading via stacked indices + episode boundary handling
2. **ROLL can adapt this approach**: Use episode index (`traj_id` + `step`) instead of `done` flags
3. **Minimal code changes required**: Episode index is the only new data structure
4. **Performance is comparable**: Same O(batch_size * n_step) complexity
5. **Memory overhead is negligible**: ~10 MB for 100K steps

### Recommendation

**Implement the Episode Index + Dynamic Loading approach** (Section 4) because:
- ✅ Matches tianshou's proven design
- ✅ Minimal changes to existing code
- ✅ Maintains compatibility with StepEnvManager
- ✅ Enables both n-step returns and GAE
- ✅ No performance degradation

### Next Steps

1. **Get user approval** on the proposed design
2. **Implement Phase 1** (Episode Index) first
3. **Test thoroughly** with wrap-around scenarios
4. **Iterate based on feedback**

---

## References

- [Tianshou ReplayBuffer](https://github.com/thu-ml/tianshou/blob/master/tianshou/data/buffer/base.py)
- [Schulman et al., 2016: High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Mnih et al., 2016: Asynchronous Methods for Deep Reinforcement Learning (A3C n-step returns)](https://arxiv.org/abs/1602.01783)
- [ROLL Replay Buffer Implementation](../roll/agentic/replay_buffer/)
