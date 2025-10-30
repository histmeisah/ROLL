"""
Step-Level Replay Buffer for ROLL Framework

Specifically designed to work with StepEnvManager data format.
Stores individual conversation steps with step-level rewards and penalties.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import random
import logging
import numpy as np
import torch
from transformers import PreTrainedTokenizer
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import pad_to_length
from .base_buffer import BaseReplayBuffer
from .segment_tree import SumSegmentTree, MinSegmentTree, next_power_of_2

logger = logging.getLogger(__name__)


@dataclass 
class StepEntry:
    """
    Single step entry for step-level replay buffer.
    Matches the output format of StepEnvManager.
    """
    # Core data from DataProto
    input_ids: np.ndarray
    attention_mask: np.ndarray
    position_ids: np.ndarray
    response_mask: np.ndarray
    prompt_mask: np.ndarray
    scores: np.ndarray  # Token-level scores with step reward on response tokens
    penalty: float      # Step-level penalty scalar
    behavior_log_probs: np.ndarray  # Log probabilities from behavior policy for off-policy analysis
    
    # Metadata from non_tensor_batch
    env_id: str
    group_id: str
    messages_list: List[Dict]
    tag: str
    frames: List
    step_scores: List
    episode_scores: List
    traj_group_id: str
    traj_id: str
    state_hash: str  # Added missing state_hash field
    step: int  # CRITICAL: Step index within episode, required for gigpo
    
    # Storage metadata
    stored_at_step: int
    step_length: int

    # Priority-related metadata
    priority: float = 1.0       # Current priority value (intrinsic value)
    sample_count: int = 0       # Number of times sampled (for statistics)
    global_step: int = 0        # Global training step when stored (for age calculation)


class StepReplayBuffer(BaseReplayBuffer):
    """
    Replay buffer specialized for step-level data from StepEnvManager.
    
    Key Design Principles:
    1. Step-level storage: individual conversation turns as atomic units
    2. StepEnvManager compatibility: matches exact data format
    3. Step-level penalty handling: penalties are per-step values, not cumulative
    4. Efficient step sampling: can sample from any step independently
    """
    
    def __init__(
        self,
        capacity: int = 1000000,  # Number of individual steps
        batch_size: int = 128,    # Should match rollout batch size
        seed: int = 42,
        priority_fn: callable = None,  # Priority calculation function
        priority_exponent: float = 1.0,  # Priority exponent (alpha in PER)
        priority_kwargs: dict = None,  # Additional kwargs for priority function
        enable_nstep: bool = False,  # Enable n-step returns
        n_step: int = 5,  # Number of steps for n-step returns
        gamma: float = 0.99,  # Discount factor for n-step returns
        age_decay: float = 1000.0,  # Age decay constant for freshness weighting
        use_advantage_priority: bool = False  # Whether to update priority with advantages after training
    ):
        super().__init__(capacity, batch_size, seed)
        self.steps = deque(maxlen=capacity)
        self.rng = random.Random(seed)

        # Priority-related attributes
        from .priority_functions import uniform_priority
        self.priority_fn = priority_fn or uniform_priority
        self.priority_exponent = priority_exponent
        self.priority_kwargs = priority_kwargs or {}

        # Segment Tree for efficient O(log n) prioritized sampling (PER)
        # Capacity must be power of 2 for segment tree
        self._tree_capacity = next_power_of_2(capacity)
        self._it_sum = SumSegmentTree(self._tree_capacity)
        self._it_min = MinSegmentTree(self._tree_capacity)
        self._max_priority = 1.0  # Track maximum priority for new samples

        # N-Step configuration
        self.enable_nstep = enable_nstep
        self.n_step = n_step
        self.gamma = gamma

        # Age-based priority configuration
        self.age_decay = age_decay
        self.use_advantage_priority = use_advantage_priority
        self.current_global_step = 0  # Track current global step for age calculation

        # Episode Index for n-step returns and GAE
        # Maps (traj_id, step) -> buffer_idx for efficient episode structure lookup
        self._episode_index: Dict[str, Dict[int, int]] = {}
        """
        Episode index mapping: {traj_id: {step_num: buffer_idx}}

        Example:
        {
            "ep_001": {0: 42, 1: 43, 2: 44, 3: 45},  # 4-step episode
            "ep_002": {0: 123, 1: 124},               # 2-step episode
        }
        """

        self._buffer_to_episode: Dict[int, Tuple[str, int]] = {}
        """
        Reverse mapping: {buffer_idx: (traj_id, step_num)}
        Enables O(1) lookup from buffer index to episode structure.
        """

        logger.info(f"Initialized StepReplayBuffer with capacity={capacity}, tree_capacity={self._tree_capacity}, "
                   f"priority_fn={self.priority_fn.__name__}, enable_nstep={enable_nstep}, n_step={n_step}, gamma={gamma}, "
                   f"age_decay={age_decay}, use_advantage_priority={use_advantage_priority}")
    
    @property
    def buffer_type(self) -> str:
        return "step"
    
    def push_from_dataproto(self, batch: DataProto, global_step: int) -> None:
        """
        Store step data from StepEnvManager.

        Args:
            batch: DataProto from StepEnvManager containing individual steps
            global_step: Current training step
        """
        # Update current global step for age calculation
        self.current_global_step = global_step

        batch_size = batch.batch["input_ids"].shape[0]
        
        for i in range(batch_size):
            # Extract tensor data
            input_ids = batch.batch["input_ids"][i].cpu().numpy()
            attention_mask = batch.batch["attention_mask"][i].cpu().numpy()
            position_ids = batch.batch["position_ids"][i].cpu().numpy()
            response_mask = batch.batch["response_mask"][i].cpu().numpy()
            prompt_mask = batch.batch["prompt_mask"][i].cpu().numpy()
            scores = batch.batch["scores"][i].cpu().numpy()
            penalty = float(batch.batch["penalty"][i].cpu().item())
            
            # Extract behavior policy log_probs if available
            behavior_log_probs = None
            if "behavior_log_probs" in batch.batch:
                behavior_log_probs = batch.batch["behavior_log_probs"][i].cpu().numpy()
            else:
                # CRITICAL FIX: If no behavior_log_probs, create zeros with length input_ids - 1
                # to align with next-token prediction semantics
                behavior_log_probs = np.zeros_like(input_ids[:-1], dtype=np.float32)
            
            # Extract metadata
            env_id = batch.non_tensor_batch["env_ids"][i]
            group_id = batch.non_tensor_batch["group_ids"][i]
            messages_list = batch.non_tensor_batch["messages_list"][i]
            tag = batch.non_tensor_batch["tags"][i]
            frames = batch.non_tensor_batch["frames"][i]
            step_scores = batch.non_tensor_batch["step_scores"][i]
            episode_scores = batch.non_tensor_batch["episode_scores"][i]
            traj_group_id = batch.non_tensor_batch["traj_group_id"][i]
            traj_id = batch.non_tensor_batch["traj_id"][i]
            state_hash = batch.non_tensor_batch["state_hash"][i]  # Extract state_hash
            
            # Extract step index - CRITICAL for gigpo algorithm
            step = int(batch.non_tensor_batch["step"][i]) if "step" in batch.non_tensor_batch else 0
            
            # Calculate step length from attention mask
            step_length = int(attention_mask.sum())
            
            step_entry = StepEntry(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                response_mask=response_mask,
                prompt_mask=prompt_mask,
                scores=scores,
                penalty=penalty,
                behavior_log_probs=behavior_log_probs,
                env_id=env_id,
                group_id=group_id,
                messages_list=messages_list,
                tag=tag,
                frames=frames,
                step_scores=step_scores,
                episode_scores=episode_scores,
                traj_group_id=traj_group_id,
                traj_id=traj_id,
                state_hash=state_hash,  # Add state_hash
                step=step,  # Add step index for gigpo
                stored_at_step=global_step,
                step_length=step_length,
                global_step=global_step  # Store global step for age calculation
            )

            # Calculate priority for this step
            try:
                priority = self.priority_fn(step_entry, global_step, **self.priority_kwargs)
                step_entry.priority = float(priority)
            except Exception as e:
                logger.warning(f"Failed to calculate priority, using default 1.0: {e}")
                step_entry.priority = 1.0

            # IMPORTANT: Use total_stored for consistent indexing across buffer wrap-around
            # deque automatically handles capacity, but segment tree needs explicit index
            current_idx = self.total_stored % self.capacity

            # Handle eviction when buffer is full (clean up episode index)
            if len(self.steps) == self.capacity:
                self._cleanup_evicted_step(current_idx)

            # Update segment trees with priority^alpha (PER convention)
            # New samples get max priority to ensure they're sampled at least once
            priority_alpha = max(step_entry.priority, self._max_priority) ** self.priority_exponent
            self._it_sum[current_idx] = priority_alpha
            self._it_min[current_idx] = priority_alpha
            self._max_priority = max(self._max_priority, step_entry.priority)

            # Update episode index for n-step returns
            if traj_id not in self._episode_index:
                self._episode_index[traj_id] = {}
            self._episode_index[traj_id][step] = current_idx
            self._buffer_to_episode[current_idx] = (traj_id, step)

            # Append to deque (auto-evicts oldest when full)
            self.steps.append(step_entry)
            self.total_stored += 1

        # Periodic garbage collection to prevent memory leaks
        if self.total_stored % 100000 == 0 and self.total_stored > 0:
            import gc
            gc.collect()
            logger.info(f"Replay buffer GC triggered at {self.total_stored} steps stored")

        logger.debug(f"Stored {batch_size} steps. Total stored: {len(self.steps)}")
    
    def can_sample(self, batch_size: Optional[int] = None) -> bool:
        """Check if buffer has enough steps for sampling."""
        required_size = batch_size or self.batch_size
        return len(self.steps) >= required_size
    
    def sample_for_training(
        self,
        batch_size: Optional[int] = None,
        device: str = 'cpu',
        tokenizer: Optional[PreTrainedTokenizer] = None,
        sequence_length: int = 4096,
        sampling_mode: str = "trajectory",
        steps_per_episode: int = 1,
        sample_method: str = "uniform",
        candidates_per_group: int = 1,
        group_sampling: str = "uniform",
        compute_importance_weights: bool = False,
        importance_weight_beta: float = 0.4
    ) -> Optional[Tuple[DataProto, List[int]]]:
        """
        Sample steps and reconstruct DataProto format.

        Args:
            batch_size: Number of steps to sample
            device: Target device for tensors ('cpu' or 'cuda')
            tokenizer: Tokenizer for text processing
            sequence_length: Maximum sequence length for padding/truncation
            sampling_mode: "trajectory" or "step" sampling mode (ignored, always step)
            steps_per_episode: Number of steps per episode (ignored for step mode)
            sample_method: Sampling method ("uniform", "weighted", etc.)
            candidates_per_group: Number of candidates per group
            group_sampling: Group sampling strategy ("uniform", etc.)
            compute_importance_weights: Whether to compute importance weights for PER
            importance_weight_beta: Beta parameter for importance weight (0.4 -> 1.0 annealing)

        Returns:
            Tuple of (DataProto batch, sampled_indices)
            - DataProto contains training batch with optional importance_weights
            - sampled_indices: list of buffer indices for priority update after training
        """
        sample_size = batch_size or self.batch_size

        if not self.can_sample(sample_size):
            logger.debug(f"Insufficient steps for sampling: {len(self.steps)} < {sample_size}")
            return None, []

        # Sample steps based on priority function
        # Deterministic strategies: uniform, lifo, fifo
        # Weighted strategies: reward, td_error, recency, combined, etc.
        buffer_list = list(self.steps)
        buffer_size = len(buffer_list)
        priority_fn_name = self.priority_fn.__name__

        # Track sampled indices for priority updates and importance weights
        sampled_indices = []

        if priority_fn_name == "lifo_priority":
            # LIFO (Last-In-First-Out): Deterministic sampling of newest N steps
            # Recommended for Echo mode (train_steps_per_env_step=1) for near-on-policy training
            start_idx = max(0, buffer_size - sample_size)
            sampled_indices = list(range(start_idx, buffer_size))
            sampled_steps = buffer_list[start_idx:]
            logger.debug(f"LIFO sampling: selected last {len(sampled_steps)} steps (indices {start_idx} to {buffer_size})")

        elif priority_fn_name == "fifo_priority":
            # FIFO (First-In-First-Out): Deterministic sampling of oldest N steps
            # Ensures all data is used before eviction
            sampled_indices = list(range(sample_size))
            sampled_steps = buffer_list[:sample_size]
            logger.debug(f"FIFO sampling: selected first {len(sampled_steps)} steps")

        elif priority_fn_name == "uniform_priority":
            # Uniform random sampling: Standard replay buffer behavior
            sampled_indices = self.rng.sample(range(buffer_size), sample_size)
            sampled_steps = [buffer_list[i] for i in sampled_indices]
            logger.debug(f"Uniform sampling: randomly selected {len(sampled_steps)} steps")

        else:
            # Weighted priority-based sampling using Segment Tree (O(log n) per sample)
            # This is the core of Prioritized Experience Replay (PER)
            sampled_indices = self._sample_proportional(sample_size, buffer_size)
            sampled_steps = [buffer_list[i] for i in sampled_indices]

            # Update sample counts for statistics
            for idx in sampled_indices:
                buffer_list[idx].sample_count += 1

            logger.debug(f"PER sampling ({priority_fn_name}): sampled {len(sampled_steps)} steps with priority_alpha={self.priority_exponent}")
        
        # Use pipeline's sequence_length for consistent padding (like RolloutScheduler)
        # This ensures compatibility with original ROLL behavior and prevents length mismatch
        max_seq_len = sequence_length
        
        # Prepare tensors
        target_device = torch.device(device if device is not None else 'cpu')

        batch_input_ids = torch.zeros((sample_size, max_seq_len), dtype=torch.long, device=target_device)
        batch_attention_mask = torch.zeros((sample_size, max_seq_len), dtype=torch.bool, device=target_device)
        batch_position_ids = torch.zeros((sample_size, max_seq_len), dtype=torch.long, device=target_device)
        batch_response_mask = torch.zeros((sample_size, max_seq_len), dtype=torch.bool, device=target_device)
        batch_prompt_mask = torch.zeros((sample_size, max_seq_len), dtype=torch.bool, device=target_device)
        batch_scores = torch.zeros((sample_size, max_seq_len), dtype=torch.float, device=target_device)
        batch_penalties = torch.zeros(sample_size, dtype=torch.float, device=target_device)
        # old_log_probs follow next-token semantics: length is (sequence_length - 1)
        batch_old_log_probs = torch.zeros((sample_size, max_seq_len - 1), dtype=torch.float, device=target_device)
        
        # Padding values (consistent with RolloutScheduler._apply_pipeline_padding)
        pad_token_id = tokenizer.pad_token_id if tokenizer else 0
        
        # Prepare non-tensor batch
        env_ids = []
        group_ids = []
        messages_lists = []
        tags = []
        frames_lists = []
        step_scores_lists = []
        episode_scores_lists = []
        traj_group_ids = []
        traj_ids = []
        state_hashes = []  # Add state_hashes list
        steps = []  # Add steps list for gigpo
        
        for i, step in enumerate(sampled_steps):
            original_seq_len = len(step.input_ids)
            
            # Apply consistent truncation/padding like RolloutScheduler
            effective_seq_len = min(original_seq_len, max_seq_len)
            
            # Use pad_to_length for consistent behavior with original ROLL env_manager
            step_input_ids = torch.from_numpy(step.input_ids)
            step_attention_mask = torch.from_numpy(step.attention_mask.astype(bool))
            step_position_ids = torch.from_numpy(step.position_ids)
            step_response_mask = torch.from_numpy(step.response_mask.astype(bool))
            step_prompt_mask = torch.from_numpy(step.prompt_mask.astype(bool))
            step_scores = torch.from_numpy(step.scores)
            
            batch_input_ids[i] = pad_to_length(step_input_ids, max_seq_len, pad_token_id)
            # Keep attention_mask padded with 0 to match RolloutScheduler
            batch_attention_mask[i] = pad_to_length(step_attention_mask, max_seq_len, 0)
            batch_position_ids[i] = pad_to_length(step_position_ids, max_seq_len, 0)
            batch_response_mask[i] = pad_to_length(step_response_mask, max_seq_len, 0)
            batch_prompt_mask[i] = pad_to_length(step_prompt_mask, max_seq_len, 0)
            batch_scores[i] = pad_to_length(step_scores, max_seq_len, 0.0)
            
            batch_penalties[i] = step.penalty
            
            # Handle behavior_log_probs with pad_to_length to next-token length (sequence_length-1)
            step_behavior_log_probs = torch.from_numpy(step.behavior_log_probs)
            batch_old_log_probs[i] = pad_to_length(step_behavior_log_probs, max_seq_len - 1, 0.0)
            
            # Collect non-tensor data
            env_ids.append(step.env_id)
            group_ids.append(step.group_id)
            messages_lists.append(step.messages_list)
            tags.append(step.tag)
            frames_lists.append(step.frames)
            step_scores_lists.append(step.step_scores)
            episode_scores_lists.append(step.episode_scores)
            traj_group_ids.append(step.traj_group_id)
            traj_ids.append(step.traj_id)
            state_hashes.append(step.state_hash)  # Collect state_hash
            steps.append(step.step)  # Collect step index for gigpo
        
        # Create DataProto in the exact format StepEnvManager produces
        dataproto = DataProto()
        dataproto.batch = TensorDict({
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "position_ids": batch_position_ids,
            "response_mask": batch_response_mask,
            "prompt_mask": batch_prompt_mask,
            "scores": batch_scores,
            "penalty": batch_penalties,
            "old_log_probs": batch_old_log_probs,
        }, batch_size=[sample_size])
        
        dataproto.non_tensor_batch = {
            "env_ids": np.array(env_ids, dtype=object),
            "group_ids": np.array(group_ids, dtype=object),
            "messages_list": np.array(messages_lists, dtype=object),
            "tags": np.array(tags, dtype=object),
            "frames": np.array(frames_lists, dtype=object),
            "step_scores": np.array(step_scores_lists, dtype=object),
            "episode_scores": np.array(episode_scores_lists, dtype=object),
            "traj_group_id": np.array(traj_group_ids, dtype=object),
            "traj_id": np.array(traj_ids, dtype=object),
            "state_hash": np.array(state_hashes, dtype=object),  # Add state_hash field
            "step": np.array(steps, dtype=object),  # CRITICAL: Add step field for gigpo
        }
        
        dataproto.meta_info = {
            "from_replay_buffer": True,
            "buffer_type": "step",
            "sample_size": sample_size,
            "buffer_utilization": len(self.steps) / self.capacity,
            "sampled_indices": sampled_indices  # For priority update after training
        }

        # Compute importance weights for PER (off-policy correction)
        if compute_importance_weights and priority_fn_name not in ["lifo_priority", "fifo_priority", "uniform_priority"]:
            importance_weights = self.compute_importance_weights(sampled_indices, beta=importance_weight_beta)
            dataproto.batch["importance_weights"] = torch.from_numpy(importance_weights).to(target_device)
            logger.debug(f"Computed importance weights with beta={importance_weight_beta:.2f}, mean={importance_weights.mean():.4f}")

        # Compute n-step returns if enabled
        if self.enable_nstep:
            nstep_returns, completeness_mask = self.compute_nstep_returns(
                sampled_indices=sampled_indices,
                n_step=self.n_step,
                gamma=self.gamma,
                bootstrap_values=None  # Bootstrap values computed in pipeline if needed
            )
            dataproto.batch["nstep_returns"] = torch.from_numpy(nstep_returns).to(target_device)
            dataproto.batch["nstep_completeness"] = torch.from_numpy(completeness_mask.astype(np.float32)).to(target_device)
            dataproto.meta_info["nstep_complete_ratio"] = float(completeness_mask.mean())
            logger.debug(
                f"Computed n-step returns: n_step={self.n_step}, gamma={self.gamma}, "
                f"complete_ratio={completeness_mask.mean():.2f}"
            )

        logger.debug(f"Sampled {sample_size} steps for training (indices: {len(sampled_indices)})")
        return dataproto, sampled_indices

    def _sample_proportional(self, batch_size: int, buffer_size: int) -> List[int]:
        """
        Sample indices based on priorities using Segment Tree.

        This implements stratified sampling from Prioritized Experience Replay:
        - Divide total priority into batch_size equal ranges
        - Sample uniformly within each range
        - Use SumSegmentTree.find_prefixsum_idx for O(log n) lookup

        Time complexity: O(batch_size * log n)

        Args:
            batch_size: Number of samples to draw
            buffer_size: Current buffer size

        Returns:
            List of sampled indices
        """
        indices = []
        p_total = self._it_sum.sum(0, buffer_size)

        if p_total <= 0:
            # Fallback to uniform if no valid priorities
            logger.warning("Total priority is 0, falling back to uniform sampling")
            return self.rng.sample(range(buffer_size), batch_size)

        # Stratified sampling: divide into batch_size segments
        every_range_len = p_total / batch_size

        for i in range(batch_size):
            # Sample uniformly within this segment
            mass = self.rng.random() * every_range_len + i * every_range_len
            # Find the index corresponding to this priority mass
            idx = self._it_sum.find_prefixsum_idx(mass)
            # Ensure index is within buffer bounds
            idx = min(idx, buffer_size - 1)
            indices.append(idx)

        return indices

    def update_priorities(self, indices: List[int], priorities: np.ndarray, current_global_step: Optional[int] = None) -> None:
        """
        Update priorities for sampled steps after training, with age-aware weighting.

        This implements a two-factor priority system:
        1. Intrinsic value (advantage-based): How surprising/valuable the sample is
        2. Freshness weight (age-based): How recent the sample is

        Final effective priority = intrinsic_value * freshness_weight
        where freshness_weight = exp(-age / age_decay)

        Time complexity: O(k * log n) where k = len(indices)

        Args:
            indices: Buffer indices of sampled steps
            priorities: New intrinsic priority values (e.g., |advantage|, |TD-error|, |loss|)
            current_global_step: Current training step for age calculation (optional, uses self.current_global_step if None)

        Example:
            >>> # After training and computing advantages
            >>> advantages = batch["advantages"]  # [batch_size, seq_len]
            >>> response_mask = batch["response_mask"]  # [batch_size, seq_len]
            >>> priorities = masked_mean(abs(advantages), response_mask).cpu().numpy()
            >>> buffer.update_priorities(sampled_indices, priorities, current_global_step)
        """
        assert len(indices) == len(priorities), \
            f"Indices and priorities length mismatch: {len(indices)} vs {len(priorities)}"

        buffer_list = list(self.steps)
        global_step = current_global_step if current_global_step is not None else self.current_global_step

        for idx, intrinsic_priority in zip(indices, priorities):
            if not (0 <= idx < len(buffer_list)):
                logger.warning(f"Invalid index {idx} for buffer size {len(buffer_list)}, skipping")
                continue

            # Ensure intrinsic priority is positive (add small epsilon)
            intrinsic_priority = max(float(intrinsic_priority), 1e-6)

            # Compute age-based freshness weight
            sample_age = global_step - buffer_list[idx].global_step
            freshness_weight = np.exp(-sample_age / self.age_decay)

            # Compute effective priority: intrinsic value × freshness
            effective_priority = intrinsic_priority * freshness_weight

            # Update segment trees with effective_priority^alpha
            priority_alpha = effective_priority ** self.priority_exponent
            self._it_sum[idx] = priority_alpha
            self._it_min[idx] = priority_alpha

            # Update step entry intrinsic priority (store raw value for debugging)
            buffer_list[idx].priority = intrinsic_priority

            # Track maximum priority
            self._max_priority = max(self._max_priority, effective_priority)

        logger.debug(f"Updated priorities for {len(indices)} steps with age decay. "
                    f"Max effective priority: {self._max_priority:.4f}")

    def get_effective_priority(self, idx: int, current_global_step: Optional[int] = None) -> float:
        """
        Compute effective priority for a given buffer index.

        effective_priority = intrinsic_priority * exp(-age / age_decay)

        Args:
            idx: Buffer index
            current_global_step: Current training step (optional)

        Returns:
            Effective priority value
        """
        buffer_list = list(self.steps)
        if not (0 <= idx < len(buffer_list)):
            return 0.0

        global_step = current_global_step if current_global_step is not None else self.current_global_step
        sample_age = global_step - buffer_list[idx].global_step
        freshness_weight = np.exp(-sample_age / self.age_decay)

        return buffer_list[idx].priority * freshness_weight

    @staticmethod
    def compute_advantage_priorities(batch: DataProto) -> np.ndarray:
        """
        Compute advantage-based priorities from a DataProto batch.

        This extracts advantages from the batch and computes mean absolute advantage
        per trajectory/step, masked by response tokens.

        Args:
            batch: DataProto containing "advantages" and "response_mask"

        Returns:
            priorities: [batch_size] array of advantage-based priorities

        Example:
            >>> # After compute_advantage() in pipeline
            >>> priorities = StepReplayBuffer.compute_advantage_priorities(batch)
            >>> replay_buffer.update_priorities(sampled_indices, priorities, global_step)
        """
        if "advantages" not in batch.batch or "response_mask" not in batch.batch:
            raise ValueError("Batch must contain 'advantages' and 'response_mask' for advantage-based priorities")

        advantages = batch.batch["advantages"]  # [batch_size, seq_len]
        response_mask = batch.batch["response_mask"]  # [batch_size, seq_len]

        # Compute masked mean absolute advantage per sample
        abs_advantages = torch.abs(advantages)
        masked_advantages = abs_advantages * response_mask.float()

        # Sum over tokens and divide by number of response tokens
        sum_advantages = masked_advantages.sum(dim=1)  # [batch_size]
        num_tokens = response_mask.sum(dim=1).float().clamp(min=1.0)  # [batch_size], avoid division by zero

        priorities = (sum_advantages / num_tokens).cpu().numpy()

        return priorities

    def compute_importance_weights(
        self,
        indices: List[int],
        beta: float = 0.4
    ) -> np.ndarray:
        """
        Compute importance sampling weights for off-policy correction.

        Importance weights correct for the bias introduced by prioritized sampling.
        Formula: w_i = (N * P(i))^(-beta) / max_w

        Beta annealing schedule (common in PER):
        - Start: beta = 0.4 (partial correction)
        - End: beta = 1.0 (full correction)
        - Anneal linearly over training

        Args:
            indices: Buffer indices of sampled steps
            beta: Importance weight exponent (0 = no correction, 1 = full correction)

        Returns:
            Importance weights normalized by max weight, shape [batch_size]

        Example:
            >>> indices, weights = buffer.sample_with_weights(batch_size=128, beta=0.6)
            >>> loss = compute_loss(batch) * torch.from_numpy(weights)
        """
        buffer_size = len(self.steps)

        # Get minimum priority for normalization
        p_min = self._it_min.min(0, buffer_size)
        p_total = self._it_sum.sum(0, buffer_size)

        if p_total <= 0:
            # Fallback: uniform weights
            return np.ones(len(indices), dtype=np.float32)

        # Compute max weight for normalization
        # max_weight occurs at minimum priority
        max_weight = (p_min / p_total * buffer_size) ** (-beta)

        weights = []
        for idx in indices:
            # Get priority for this sample
            p_sample = self._it_sum[idx]
            # Compute probability
            prob = p_sample / p_total
            # Compute importance weight
            weight = (prob * buffer_size) ** (-beta)
            # Normalize by max weight
            weights.append(weight / max_weight)

        return np.array(weights, dtype=np.float32)

    def get_stats(self) -> dict:
        """Get buffer statistics including priority information."""
        base_stats = super().get_stats()

        # Add priority statistics from segment tree
        current_size = len(self.steps)
        if current_size > 0:
            # Extract priorities from segment tree
            priorities = np.array([self._it_sum[i] for i in range(current_size)])
            base_stats.update({
                "priority/mean": float(priorities.mean()),
                "priority/std": float(priorities.std()),
                "priority/max": float(priorities.max()),
                "priority/min": float(priorities.min()),
                "priority_fn": self.priority_fn.__name__,
                "priority_exponent": self.priority_exponent,
                "max_priority": self._max_priority,
            })

        # Add episode index statistics
        if self._episode_index:
            base_stats.update({
                "episode_index/num_episodes": len(self._episode_index),
                "episode_index/avg_episode_length": np.mean([len(ep) for ep in self._episode_index.values()]),
                "episode_index/total_indexed_steps": sum(len(ep) for ep in self._episode_index.values()),
            })

        return base_stats

    def _cleanup_evicted_step(self, evicted_idx: int) -> None:
        """
        Clean up episode index when a step is evicted from buffer.

        Args:
            evicted_idx: Buffer index being evicted (overwritten)
        """
        if evicted_idx not in self._buffer_to_episode:
            return

        old_traj_id, old_step = self._buffer_to_episode[evicted_idx]

        # Remove from episode index
        if old_traj_id in self._episode_index:
            if old_step in self._episode_index[old_traj_id]:
                del self._episode_index[old_traj_id][old_step]

            # If episode is now empty, remove it entirely
            if not self._episode_index[old_traj_id]:
                del self._episode_index[old_traj_id]
                logger.debug(f"Episode {old_traj_id} fully evicted from buffer")

        # Remove from reverse mapping
        del self._buffer_to_episode[evicted_idx]

    def get_nstep_indices(
        self,
        start_idx: int,
        n_step: int
    ) -> Tuple[List[int], bool]:
        """
        Get next n steps from the same episode (tianshou-style traversal).

        This is the core method for n-step returns and GAE computation.
        It mimics tianshou's next() method but uses explicit episode index.

        Args:
            start_idx: Starting buffer index
            n_step: Number of steps to look ahead

        Returns:
            (indices, complete):
            - indices: List of buffer indices [start_idx, next_idx, ...], length <= n_step
            - complete: True if we got full n steps without hitting episode boundary

        Example:
            >>> indices, complete = buffer.get_nstep_indices(42, n_step=5)
            >>> if complete:
            >>>     # We have 5 consecutive steps: indices = [42, 43, 44, 45, 46]
            >>> else:
            >>>     # Episode ended early: indices = [42, 43] (only 2 steps available)
        """
        if start_idx not in self._buffer_to_episode:
            logger.warning(f"Buffer index {start_idx} not in episode index")
            return [start_idx], False

        traj_id, start_step = self._buffer_to_episode[start_idx]

        if traj_id not in self._episode_index:
            logger.warning(f"Episode {traj_id} not in episode index")
            return [start_idx], False

        episode = self._episode_index[traj_id]
        max_step = max(episode.keys())  # Last step in this episode

        indices = []
        for offset in range(n_step):
            target_step = start_step + offset

            # Check if step exists in episode
            if target_step not in episode:
                # Hit episode boundary (step was evicted or doesn't exist)
                return indices, False

            indices.append(episode[target_step])

            # Check if this is the last step (after adding it to indices)
            if target_step == max_step:
                # Reached end of episode, but we've collected up to max_step
                # Return False because we can't get more steps beyond this
                return indices, False

        # Got full n steps without hitting boundary
        return indices, True

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
            Each column contains n consecutive steps from same episode

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
                    # This allows vectorized computation while handling variable episode lengths
                    stacked[step, i] = indices[-1] if indices else start_idx

        return stacked

    def compute_nstep_returns(
        self,
        sampled_indices: List[int],
        n_step: Optional[int] = None,
        gamma: Optional[float] = None,
        bootstrap_values: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute n-step returns for sampled steps.

        N-step return formula:
        R_t^(n) = r_t + γ*r_{t+1} + ... + γ^(n-1)*r_{t+n-1} + γ^n*V(s_{t+n})

        Args:
            sampled_indices: Starting buffer indices [batch_size]
            n_step: Number of steps for return computation (default: self.n_step)
            gamma: Discount factor (default: self.gamma)
            bootstrap_values: Optional bootstrapped values [batch_size]
                            If None, no bootstrap is added (Monte Carlo return)

        Returns:
            Tuple of:
            - nstep_returns: [batch_size] computed n-step returns
            - completeness_mask: [batch_size] boolean array
                                True if full n-step sequence was available
                                False if hit episode boundary early

        Example:
            >>> returns, complete = buffer.compute_nstep_returns(
            ...     sampled_indices=[0, 10, 20],
            ...     n_step=5,
            ...     gamma=0.99,
            ...     bootstrap_values=critic_values
            ... )
            >>> # returns.shape = [3], complete.shape = [3]
            >>> # complete[0] = True means indices[0] had full 5-step sequence
        """
        n_step = n_step or self.n_step
        gamma = gamma or self.gamma

        batch_size = len(sampled_indices)
        returns = np.zeros(batch_size, dtype=np.float32)
        completeness_mask = np.zeros(batch_size, dtype=bool)

        buffer_list = list(self.steps)

        for i, start_idx in enumerate(sampled_indices):
            # Get n-step trajectory
            indices, complete = self.get_nstep_indices(start_idx, n_step)
            completeness_mask[i] = complete

            if not indices:
                logger.warning(f"No indices found for start_idx={start_idx}, traj may be evicted")
                continue

            # Accumulate discounted rewards
            discount = 1.0
            for idx in indices:
                if idx >= len(buffer_list):
                    logger.warning(f"Index {idx} out of bounds for buffer size {len(buffer_list)}")
                    break

                step_entry = buffer_list[idx]

                # Extract step reward: sum over response tokens
                response_mask_bool = step_entry.response_mask.astype(bool)
                reward = float(step_entry.scores[response_mask_bool].sum())

                returns[i] += discount * reward
                discount *= gamma

            # Add bootstrap value if trajectory is complete and values provided
            if complete and bootstrap_values is not None:
                returns[i] += discount * bootstrap_values[i]

        return returns, completeness_mask

    def compute_gae(
        self,
        sampled_indices: List[int],
        values: np.ndarray,
        gamma: Optional[float] = None,
        lambda_: float = 0.95,
        n_step: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE) for sampled steps.

        GAE formula:
        A_t = Σ_{k=0}^∞ (γλ)^k * δ_{t+k}
        where: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)

        This implementation uses truncated GAE with horizon n_step.

        Args:
            sampled_indices: Starting buffer indices [batch_size]
            values: State values V(s_t) for each sampled step [batch_size]
            gamma: Discount factor (default: self.gamma)
            lambda_: GAE lambda parameter (exponential smoothing)
            n_step: Truncation horizon for GAE computation

        Returns:
            Tuple of:
            - advantages: [batch_size] computed GAE advantages
            - completeness_mask: [batch_size] boolean array indicating complete sequences

        Example:
            >>> advantages, complete = buffer.compute_gae(
            ...     sampled_indices=[0, 10, 20],
            ...     values=critic_values,  # [3]
            ...     gamma=0.99,
            ...     lambda_=0.95,
            ...     n_step=20
            ... )
        """
        gamma = gamma or self.gamma

        batch_size = len(sampled_indices)
        advantages = np.zeros(batch_size, dtype=np.float32)
        completeness_mask = np.zeros(batch_size, dtype=bool)

        buffer_list = list(self.steps)

        for i, start_idx in enumerate(sampled_indices):
            # Get trajectory for GAE computation
            indices, complete = self.get_nstep_indices(start_idx, n_step)
            completeness_mask[i] = complete

            if len(indices) < 2:
                # Need at least 2 steps to compute TD error
                advantages[i] = 0.0
                continue

            # Compute TD errors for each step in the trajectory
            deltas = []
            for j, idx in enumerate(indices[:-1]):  # Exclude last step
                if idx >= len(buffer_list):
                    break

                step_entry = buffer_list[idx]
                next_idx = indices[j + 1]

                if next_idx >= len(buffer_list):
                    break

                next_step_entry = buffer_list[next_idx]

                # Extract reward
                response_mask_bool = step_entry.response_mask.astype(bool)
                reward = float(step_entry.scores[response_mask_bool].sum())

                # Get values
                # For the first step, use provided value; for others, reuse if available
                # Note: This is a simplified version. Ideally, we should recompute values
                # for all steps, but that would require a critic forward pass
                v_t = values[i] if j == 0 else 0.0  # Simplified
                v_next = 0.0  # Simplified

                # Compute TD error
                delta = reward + gamma * v_next - v_t
                deltas.append(delta)

            # Compute GAE using backward accumulation
            gae = 0.0
            for delta in reversed(deltas):
                gae = delta + gamma * lambda_ * gae

            advantages[i] = gae

        return advantages, completeness_mask