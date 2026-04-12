"""
Group-Level Replay Buffer for ROLL Framework

Stores and samples trajectory groups as atomic units, ensuring GRPO compatibility.
Each group contains K trajectories sharing the same traj_group_id (same prompt/state),
so group-level reward normalization works correctly on replay data.

When group_size=1, degrades to standard per-trajectory replay buffer behavior.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import random
import numpy as np
import torch
from transformers import PreTrainedTokenizer
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import pad_to_length
from roll.utils.logging import get_logger
from .base_buffer import BaseReplayBuffer
from .segment_tree import SumSegmentTree, MinSegmentTree, next_power_of_2
from .trajectory_buffer import TrajectoryEntry

logger = get_logger()


@dataclass
class TrajectoryGroup:
    """
    A group of K trajectories sharing the same traj_group_id.
    This is the atomic storage/sampling unit for GRPO-compatible replay.
    """
    traj_group_id: str
    trajectories: List[TrajectoryEntry]
    group_size: int  # K, the number of trajectories in this group

    # Group-level metadata
    tag: str = ""
    mean_episode_score: float = 0.0
    stored_at_step: int = 0

    # Priority-related
    priority: float = 1.0
    sample_count: int = 0
    global_step: int = 0

    def __post_init__(self):
        self.group_size = len(self.trajectories)
        if self.trajectories:
            scores = [
                float(t.episode_scores) if isinstance(t.episode_scores, (int, float, np.floating))
                else float(t.episode_scores[-1]) if hasattr(t.episode_scores, '__len__') and len(t.episode_scores) > 0
                else 0.0
                for t in self.trajectories
            ]
            self.mean_episode_score = float(np.mean(scores))
            self.tag = self.trajectories[0].tag


class GroupReplayBuffer(BaseReplayBuffer):
    """
    Replay buffer that stores and samples trajectory groups as atomic units.

    Designed for GRPO compatibility: each group contains K trajectories
    from the same prompt/state, so group-level reward normalization
    works correctly on replay data.

    When group_size=1, this degrades to standard per-trajectory replay.

    Storage unit: TrajectoryGroup (K trajectories)
    Sampling unit: N groups → N×K trajectories as DataProto
    Priority unit: group-level (mean reward, mean advantage, etc.)
    """

    def __init__(
        self,
        capacity: int = 10000,
        batch_size: int = 32,
        seed: int = 42,
        priority_fn: callable = None,
        priority_exponent: float = 0.6,
        age_decay: float = 1000.0,
        enable_age_decay: bool = False,
        eviction_strategy: str = "fifo",
    ):
        """
        Args:
            capacity: Maximum number of groups (not individual trajectories)
            batch_size: Default number of groups to sample (actual trajectory count = batch_size × K)
            seed: Random seed
            priority_fn: Priority function for group-level priority
            priority_exponent: Alpha in PER (0=uniform, 1=full prioritization)
            age_decay: Age decay constant for freshness weighting
            enable_age_decay: Enable age-based freshness weighting
            eviction_strategy: "fifo" or "smart"
        """
        super().__init__(capacity, batch_size, seed)
        self.groups: List[Optional[TrajectoryGroup]] = [None] * capacity
        self.valid_mask: List[bool] = [False] * capacity
        self.num_valid: int = 0
        self.rng = random.Random(seed)
        self.eviction_strategy = eviction_strategy.lower()

        # Priority
        from .priority_functions import uniform_priority
        self.priority_fn = priority_fn or uniform_priority
        self.priority_exponent = priority_exponent

        # Segment tree for O(log n) prioritized sampling
        self._tree_capacity = next_power_of_2(capacity)
        self._it_sum = SumSegmentTree(self._tree_capacity)
        self._it_min = MinSegmentTree(self._tree_capacity)
        self._max_priority = 1.0

        # Age decay
        self.enable_age_decay = enable_age_decay
        self.age_decay = age_decay
        self.current_global_step = 0

        logger.info(
            f"GroupReplayBuffer: capacity={capacity} groups, "
            f"priority_fn={self.priority_fn.__name__}, "
            f"priority_exponent={priority_exponent}, "
            f"eviction={eviction_strategy}"
        )

    @property
    def buffer_type(self) -> str:
        return "group"

    # ─── Push ────────────────────────────────────────────────────────────

    def push_from_dataproto(self, batch: DataProto, global_step: int) -> None:
        """
        Store trajectory data grouped by traj_group_id.

        Collects trajectories sharing the same traj_group_id into groups,
        then stores each complete group as one atomic unit.
        """
        self.current_global_step = global_step
        batch_size = batch.batch["input_ids"].shape[0]

        # Step 1: group trajectories by traj_group_id
        group_map: Dict[str, List[int]] = defaultdict(list)
        for i in range(batch_size):
            tgid = str(batch.non_tensor_batch["traj_group_id"][i])
            group_map[tgid].append(i)

        # Step 2: build TrajectoryGroup for each group and store
        num_stored = 0
        for tgid, indices in group_map.items():
            entries = []
            for i in indices:
                entry = self._extract_trajectory_entry(batch, i, global_step)
                entries.append(entry)

            group = TrajectoryGroup(
                traj_group_id=tgid,
                trajectories=entries,
                group_size=len(entries),
                stored_at_step=global_step,
                global_step=global_step,
            )

            # Compute group-level priority from the first trajectory
            # (priority_fn expects a TrajectoryEntry; use group mean score as proxy)
            try:
                priority = self.priority_fn(entries[0], global_step, age_decay=self.age_decay)
                group.priority = float(priority)
            except Exception as e:
                logger.warning(f"Failed to calculate group priority, using 1.0: {e}")
                group.priority = 1.0

            self._store_group(group)
            num_stored += 1

        logger.debug(
            f"Pushed {num_stored} groups ({batch_size} trajectories) at step {global_step}. "
            f"Buffer: {self.num_valid}/{self.capacity} groups"
        )

    def _extract_trajectory_entry(self, batch: DataProto, idx: int, global_step: int) -> TrajectoryEntry:
        """Extract a single TrajectoryEntry from a DataProto batch at index idx."""
        input_ids = batch.batch["input_ids"][idx].cpu().numpy()
        attention_mask = batch.batch["attention_mask"][idx].cpu().numpy()
        position_ids = batch.batch["position_ids"][idx].cpu().numpy()
        response_mask = batch.batch["response_mask"][idx].cpu().numpy()
        prompt_mask = batch.batch["prompt_mask"][idx].cpu().numpy()
        scores = batch.batch["scores"][idx].cpu().numpy()
        penalty = float(batch.batch["penalty"][idx].cpu().item())

        behavior_log_probs = None
        if "behavior_log_probs" in batch.batch:
            behavior_log_probs = batch.batch["behavior_log_probs"][idx].cpu().numpy()
        else:
            target_len = max(int(input_ids.shape[0]) - 1, 0)
            behavior_log_probs = np.zeros((target_len,), dtype=np.float32)

        return TrajectoryEntry(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            response_mask=response_mask,
            prompt_mask=prompt_mask,
            scores=scores,
            penalty=penalty,
            behavior_log_probs=behavior_log_probs,
            env_id=batch.non_tensor_batch["env_ids"][idx],
            group_id=batch.non_tensor_batch["group_ids"][idx],
            messages_list=batch.non_tensor_batch["messages_list"][idx],
            tag=batch.non_tensor_batch["tags"][idx],
            frames=batch.non_tensor_batch["frames"][idx],
            step_scores=batch.non_tensor_batch["step_scores"][idx],
            episode_scores=batch.non_tensor_batch["episode_scores"][idx],
            traj_group_id=batch.non_tensor_batch["traj_group_id"][idx],
            traj_id=batch.non_tensor_batch["traj_id"][idx],
            stored_at_step=global_step,
            episode_length=int(attention_mask.sum()),
            global_step=global_step,
        )

    def _store_group(self, group: TrajectoryGroup) -> None:
        """Store a group into the buffer, evicting if full."""
        if self.num_valid >= self.capacity:
            slot_idx = self._evict_and_get_slot()
        else:
            slot_idx = self._find_empty_slot()
            self.num_valid += 1

        self.groups[slot_idx] = group
        self.valid_mask[slot_idx] = True

        # New groups get max_priority to ensure they're sampled at least once
        priority_alpha = self._max_priority ** self.priority_exponent
        self._it_sum[slot_idx] = priority_alpha
        self._it_min[slot_idx] = priority_alpha
        self.total_stored += 1

    def _find_empty_slot(self) -> int:
        for i in range(self.capacity):
            if not self.valid_mask[i]:
                return i
        raise RuntimeError("No empty slot found but num_valid < capacity")

    def _evict_and_get_slot(self) -> int:
        """FIFO eviction: evict the group with the smallest global_step."""
        oldest_idx = -1
        oldest_step = float('inf')
        for i in range(self.capacity):
            if self.valid_mask[i] and self.groups[i].global_step < oldest_step:
                oldest_step = self.groups[i].global_step
                oldest_idx = i
        if oldest_idx == -1:
            oldest_idx = 0
        # Clear the slot
        self.valid_mask[oldest_idx] = False
        self.groups[oldest_idx] = None
        self._it_sum[oldest_idx] = 0.0
        self._it_min[oldest_idx] = float('inf')
        self.num_valid -= 1
        return oldest_idx

    # ─── Sample ──────────────────────────────────────────────────────────

    def can_sample(self, batch_size: Optional[int] = None) -> bool:
        required = batch_size or self.batch_size
        return self.num_valid >= required

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
        importance_weight_beta: float = 0.4,
    ) -> Optional[Tuple[DataProto, List[int]]]:
        """
        Sample N complete groups and unpack into N×K trajectories.

        Args:
            batch_size: Number of GROUPS to sample (not trajectories).
                        Actual trajectory count = batch_size × group_size.
            sample_method: "uniform", "lifo", "fifo", or priority-based
            Others: see base class

        Returns:
            (DataProto, sampled_slot_indices)
            DataProto contains N×K trajectories with group structure preserved.
        """
        num_groups = batch_size or self.batch_size
        if not self.can_sample(num_groups):
            return None, []

        # Collect valid groups
        valid_groups: List[Tuple[int, TrajectoryGroup]] = []
        for i in range(self.capacity):
            if self.valid_mask[i]:
                valid_groups.append((i, self.groups[i]))

        # Sort by global_step for deterministic strategies
        valid_groups.sort(key=lambda x: x[1].global_step)

        # Select groups
        sampled_slots: List[int] = []
        sampled_groups: List[TrajectoryGroup] = []

        priority_fn_name = self.priority_fn.__name__

        if priority_fn_name == "lifo_priority":
            # Newest N groups
            selected = valid_groups[-num_groups:]
            sampled_slots = [s for s, _ in selected]
            sampled_groups = [g for _, g in selected]
        elif priority_fn_name == "fifo_priority":
            # Oldest N groups
            selected = valid_groups[:num_groups]
            sampled_slots = [s for s, _ in selected]
            sampled_groups = [g for _, g in selected]
        elif priority_fn_name == "uniform_priority":
            selected = self.rng.sample(valid_groups, num_groups)
            sampled_slots = [s for s, _ in selected]
            sampled_groups = [g for _, g in selected]
        else:
            # PER: weighted sampling via segment tree
            slot_indices = self._sample_proportional_slots(num_groups)
            sampled_slots = slot_indices
            sampled_groups = [self.groups[i] for i in slot_indices]
            for idx in slot_indices:
                self.groups[idx].sample_count += 1

        # Unpack groups into flat trajectory list, preserving group order
        all_trajectories: List[TrajectoryEntry] = []
        for group in sampled_groups:
            all_trajectories.extend(group.trajectories)

        total_trajs = len(all_trajectories)
        if total_trajs == 0:
            return None, []

        # Build DataProto from flat trajectory list
        max_seq_len = sequence_length
        target_device = torch.device(device if device else 'cpu')

        batch_input_ids = torch.zeros((total_trajs, max_seq_len), dtype=torch.long, device=target_device)
        batch_attention_mask = torch.zeros((total_trajs, max_seq_len), dtype=torch.bool, device=target_device)

        # Detect VLM multi-dimensional position_ids
        first_pos = all_trajectories[0].position_ids
        if first_pos.ndim > 1:
            pos_dims = first_pos.shape[0]
            batch_position_ids = torch.zeros((total_trajs, pos_dims, max_seq_len), dtype=torch.long, device=target_device)
        else:
            batch_position_ids = torch.zeros((total_trajs, max_seq_len), dtype=torch.long, device=target_device)

        batch_response_mask = torch.zeros((total_trajs, max_seq_len), dtype=torch.bool, device=target_device)
        batch_prompt_mask = torch.zeros((total_trajs, max_seq_len), dtype=torch.bool, device=target_device)
        batch_scores = torch.zeros((total_trajs, max_seq_len), dtype=torch.float32, device=target_device)
        batch_penalties = torch.zeros((total_trajs,), dtype=torch.float32, device=target_device)
        batch_old_log_probs = torch.zeros((total_trajs, max_seq_len - 1), dtype=torch.float32, device=target_device)

        # Non-tensor data
        env_ids, group_ids, messages_lists, tags = [], [], [], []
        frames_lists, step_scores_lists, episode_scores_lists = [], [], []
        traj_group_ids, traj_ids = [], []

        for i, traj in enumerate(all_trajectories):
            seq_len = min(len(traj.input_ids), max_seq_len)

            batch_input_ids[i] = pad_to_length(torch.from_numpy(traj.input_ids), max_seq_len, 0)
            batch_attention_mask[i] = pad_to_length(torch.from_numpy(traj.attention_mask), max_seq_len, False)

            pos_tensor = torch.from_numpy(traj.position_ids)
            if pos_tensor.ndim > 1:
                for d in range(pos_tensor.shape[0]):
                    batch_position_ids[i, d] = pad_to_length(pos_tensor[d], max_seq_len, 0)
            else:
                batch_position_ids[i] = pad_to_length(pos_tensor, max_seq_len, 0)

            batch_response_mask[i] = pad_to_length(torch.from_numpy(traj.response_mask), max_seq_len, False)
            batch_prompt_mask[i] = pad_to_length(torch.from_numpy(traj.prompt_mask), max_seq_len, False)
            batch_scores[i] = pad_to_length(torch.from_numpy(traj.scores), max_seq_len, 0.0)
            batch_penalties[i] = traj.penalty
            batch_old_log_probs[i] = pad_to_length(
                torch.from_numpy(traj.behavior_log_probs), max_seq_len - 1, 0.0
            )

            env_ids.append(traj.env_id)
            group_ids.append(traj.group_id)
            messages_lists.append(traj.messages_list)
            tags.append(traj.tag)
            frames_lists.append(traj.frames)
            step_scores_lists.append(traj.step_scores)
            episode_scores_lists.append(traj.episode_scores)
            traj_group_ids.append(traj.traj_group_id)
            traj_ids.append(traj.traj_id)

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
        }, batch_size=[total_trajs])

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
        }

        dataproto.meta_info = {
            "from_replay_buffer": True,
            "buffer_type": "group",
            "num_groups": num_groups,
            "group_sizes": [g.group_size for g in sampled_groups],
            "total_trajectories": total_trajs,
            "buffer_utilization": self.num_valid / self.capacity,
            "sampled_indices": sampled_slots,
        }

        # PER importance weights (group-level, broadcast to all trajectories in group)
        if compute_importance_weights and priority_fn_name not in [
            "lifo_priority", "fifo_priority", "uniform_priority"
        ]:
            group_weights = self.compute_importance_weights(sampled_slots, beta=importance_weight_beta)
            # Broadcast group weight to each trajectory in the group
            traj_weights = []
            for gi, group in enumerate(sampled_groups):
                traj_weights.extend([group_weights[gi]] * group.group_size)
            dataproto.batch["importance_weights"] = torch.tensor(
                traj_weights, dtype=torch.float32, device=target_device
            )

        logger.debug(
            f"Sampled {num_groups} groups → {total_trajs} trajectories "
            f"(group_sizes={[g.group_size for g in sampled_groups]})"
        )
        return dataproto, sampled_slots

    # ─── PER helpers ─────────────────────────────────────────────────────

    def _sample_proportional_slots(self, num_groups: int) -> List[int]:
        """Sample slot indices proportional to priority using segment tree."""
        indices = []
        p_total = self._it_sum.sum(0, self._tree_capacity - 1)
        segment = p_total / num_groups
        for i in range(num_groups):
            a = segment * i
            b = segment * (i + 1)
            upperbound = self.rng.uniform(a, b)
            idx = self._it_sum.find_prefixsum_idx(upperbound)
            if idx >= self.capacity or not self.valid_mask[idx]:
                # Fallback: pick a random valid slot
                valid_slots = [j for j in range(self.capacity) if self.valid_mask[j]]
                idx = self.rng.choice(valid_slots)
            indices.append(idx)
        return indices

    def compute_importance_weights(self, indices: List[int], beta: float = 0.4) -> np.ndarray:
        """Compute PER importance sampling weights for given slot indices."""
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum(0, self._tree_capacity - 1)
        max_weight = (p_min * self.num_valid) ** (-beta)

        for idx in indices:
            p_sample = self._it_sum[idx] / self._it_sum.sum(0, self._tree_capacity - 1)
            weight = (p_sample * self.num_valid) ** (-beta)
            weights.append(weight / max_weight)

        return np.array(weights, dtype=np.float32)

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """Update group-level priorities after training."""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < self.capacity and self.valid_mask[idx]:
                priority = max(float(priority), 1e-6)
                self.groups[idx].priority = priority
                self._max_priority = max(self._max_priority, priority)
                priority_alpha = priority ** self.priority_exponent
                self._it_sum[idx] = priority_alpha
                self._it_min[idx] = priority_alpha

    # ─── Stats ───────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        stats = {
            "buffer_type": "group",
            "capacity": self.capacity,
            "num_groups": self.num_valid,
            "total_stored_groups": self.total_stored,
            "utilization": self.num_valid / self.capacity if self.capacity > 0 else 0.0,
        }
        if self.num_valid > 0:
            group_sizes = [
                self.groups[i].group_size
                for i in range(self.capacity) if self.valid_mask[i]
            ]
            scores = [
                self.groups[i].mean_episode_score
                for i in range(self.capacity) if self.valid_mask[i]
            ]
            stats["total_trajectories"] = sum(group_sizes)
            stats["avg_group_size"] = float(np.mean(group_sizes))
            stats["avg_episode_score"] = float(np.mean(scores))
        return stats
