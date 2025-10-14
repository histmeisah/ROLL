"""
Advanced features for distributed replay buffer:
- Fault tolerance with automatic recovery
- Priority sampling
- Checkpointing
- Dynamic shard rebalancing
"""

import os
import pickle
import hashlib
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

import ray
import torch
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger

logger = get_logger()


class ShardStatus(Enum):
    """Shard health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@ray.remote
class FaultTolerantBufferShard:
    """
    Enhanced shard with fault tolerance and advanced features.
    """

    def __init__(self, shard_id: int, capacity: int,
                 enable_checkpoint: bool = True,
                 checkpoint_dir: Optional[str] = None):
        self.shard_id = shard_id
        self.capacity = capacity
        self.enable_checkpoint = enable_checkpoint
        self.checkpoint_dir = checkpoint_dir or f"/tmp/replay_buffer_shard_{shard_id}"

        # Priority sampling support
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.max_priority = 1.0
        self.priority_alpha = 0.6  # Priority exponent

        # Fault tolerance
        self.status = ShardStatus.HEALTHY
        self.backup_shard = None
        self.last_checkpoint_step = 0

        # Initialize base storage
        self._init_storage()

        # Create checkpoint directory
        if self.enable_checkpoint:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _init_storage(self):
        """Initialize storage structures"""
        # Use local storage since we're already inside a Ray Actor
        # We cannot create another Ray Actor inside this Ray Actor
        import numpy as np
        from collections import deque

        # Initialize local storage structures
        self.buffer = deque(maxlen=self.capacity)
        self.priorities = deque(maxlen=self.capacity)
        self.position = 0
        self.current_size = 0

    def push_batch(self, tensor_dict, non_tensor, meta, global_step: int) -> bool:
        """
        Push batch without priority (for uniform sampling).

        Args:
            tensor_dict: Tensor data dictionary
            non_tensor: Non-tensor data
            meta: Meta information
            global_step: Current training step

        Returns:
            Success status
        """
        batch_data = (tensor_dict, non_tensor, meta, global_step)
        return self.push_batch_with_priority(batch_data, priority=1.0)

    def push_batch_with_priority(self, batch_data: Tuple,
                                priority: Optional[float] = None) -> bool:
        """
        Push batch with priority for priority sampling.

        Args:
            batch_data: Tuple of (tensor_dict, non_tensor, meta, global_step)
            priority: Initial priority (defaults to max_priority)

        Returns:
            Success status
        """
        try:
            # Set priority for new samples
            if priority is None:
                priority = self.max_priority

            # Push to local storage
            self.buffer.append(batch_data)
            self.priorities.append(priority)
            self.current_size = len(self.buffer)

            # Update max priority
            self.max_priority = max(self.max_priority, priority)

            return True

        except Exception as e:
            logger.error(f"Shard {self.shard_id} push error: {e}")
            self.status = ShardStatus.DEGRADED
            return False

    def sample_with_priority(self, n_samples: int,
                            beta: float = 0.4) -> Optional[Tuple]:
        """
        Priority-based sampling.

        Args:
            n_samples: Number of samples
            beta: Importance sampling weight exponent

        Returns:
            Tuple of (data, indices, weights)
        """
        import numpy as np
        import random

        if len(self.buffer) == 0:
            return None

        # Calculate sampling probabilities
        valid_size = len(self.buffer)
        priorities_array = np.array(list(self.priorities)[:valid_size])
        probs = priorities_array ** self.priority_alpha
        probs = probs / probs.sum()

        # Sample indices
        indices = np.random.choice(valid_size, min(n_samples, valid_size), p=probs, replace=True)

        # Calculate importance sampling weights
        weights = (valid_size * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize

        # Get samples
        sampled_data = [self.buffer[i] for i in indices]

        # Combine sampled data (assuming they're tuples of tensor_dict, non_tensor, meta, global_step)
        if sampled_data:
            # Simply return the first sample for now (will need proper batching later)
            # Add weights to meta
            tensor_dict, non_tensor, meta, global_step = sampled_data[0]
            meta["importance_weights"] = weights.astype(np.float32)
            meta["sample_indices"] = indices
            return (tensor_dict, non_tensor, meta)

        return None

    def sample_uniform(self, n_samples: int) -> Optional[Tuple]:
        """
        Uniform random sampling without priorities.

        Args:
            n_samples: Number of samples to get

        Returns:
            Tuple of (tensor_dict, non_tensor_batch, meta_info)
        """
        import numpy as np
        import torch

        if len(self.buffer) == 0:
            return None

        valid_size = len(self.buffer)

        # Sample indices uniformly
        indices = np.random.choice(valid_size, min(n_samples, valid_size), replace=True)

        # Get samples
        sampled_data = [self.buffer[i] for i in indices]

        if sampled_data:
            # Properly batch all samples together
            all_tensor_dicts = []
            all_non_tensors = []
            all_metas = []

            for data in sampled_data:
                tensor_dict, non_tensor, meta, global_step = data
                all_tensor_dicts.append(tensor_dict)
                all_non_tensors.append(non_tensor)
                all_metas.append(meta)

            # Combine tensor dicts - convert to actual batch
            if all_tensor_dicts:
                # Create a proper batched tensor dict
                batched_tensor_dict = {}
                for key in all_tensor_dicts[0].keys():
                    # Stack tensors along batch dimension
                    tensors = []
                    for td in all_tensor_dicts:
                        if key in td:
                            if isinstance(td[key], torch.Tensor):
                                tensors.append(td[key])
                            else:
                                tensors.append(torch.tensor(td[key]))
                    if tensors:
                        batched_tensor_dict[key] = torch.stack(tensors, dim=0)
            else:
                batched_tensor_dict = {}

            # Combine non-tensor data
            batched_non_tensor = {}
            if all_non_tensors:
                for key in all_non_tensors[0].keys():
                    batched_non_tensor[key] = [nt[key] for nt in all_non_tensors if key in nt]

            # Combine meta info
            combined_meta = all_metas[0] if all_metas else {}
            combined_meta["sample_indices"] = indices
            combined_meta["num_samples"] = len(sampled_data)

            return (batched_tensor_dict, batched_non_tensor, combined_meta)

        return None

    def update_priorities(self, indices: np.ndarray,
                         priorities: np.ndarray) -> bool:
        """
        Update priorities for sampled data.

        Args:
            indices: Sample indices
            priorities: New priorities (e.g., TD errors)

        Returns:
            Success status
        """
        try:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority

            self.max_priority = max(self.max_priority, priorities.max())
            return True

        except Exception as e:
            logger.error(f"Priority update error: {e}")
            return False

    def checkpoint(self, global_step: int) -> bool:
        """
        Save shard state to disk.

        Args:
            global_step: Current training step

        Returns:
            Success status
        """
        if not self.enable_checkpoint:
            return True

        try:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{global_step}.pkl"
            )

            # Save state
            state = {
                "shard_id": self.shard_id,
                "trajectories": list(self.buffer),
                "priorities": list(self.priorities),
                "max_priority": self.max_priority,
                "total_stored": self.current_size,
                "global_step": global_step
            }

            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f)

            self.last_checkpoint_step = global_step

            # Clean old checkpoints
            self._clean_old_checkpoints(keep_last=3)

            logger.info(f"Shard {self.shard_id} checkpointed at step {global_step}")
            return True

        except Exception as e:
            logger.error(f"Checkpoint failed: {e}")
            return False

    def restore(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Restore shard state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Success status
        """
        try:
            if checkpoint_path is None:
                # Find latest checkpoint
                checkpoints = sorted([
                    f for f in os.listdir(self.checkpoint_dir)
                    if f.startswith("checkpoint_")
                ])
                if not checkpoints:
                    logger.warning(f"No checkpoints found for shard {self.shard_id}")
                    return False
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoints[-1])

            # Load state
            with open(checkpoint_path, 'rb') as f:
                state = pickle.load(f)

            # Restore state
            from collections import deque
            self.buffer = deque(state["trajectories"], maxlen=self.capacity)
            self.priorities = deque(state["priorities"], maxlen=self.capacity)
            self.max_priority = state["max_priority"]
            self.current_size = state["total_stored"]

            logger.info(f"Shard {self.shard_id} restored from {checkpoint_path}")
            self.status = ShardStatus.HEALTHY
            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            self.status = ShardStatus.FAILED
            return False

    def _clean_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoint files"""
        try:
            checkpoints = sorted([
                f for f in os.listdir(self.checkpoint_dir)
                if f.startswith("checkpoint_")
            ])
            if len(checkpoints) > keep_last:
                for old_checkpoint in checkpoints[:-keep_last]:
                    os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))
        except Exception as e:
            logger.error(f"Clean checkpoint error: {e}")

    def get_health_status(self) -> Dict:
        """Get shard health status"""
        return {
            "shard_id": self.shard_id,
            "status": self.status.value,
            "size": len(self.buffer),
            "capacity": self.capacity,
            "last_checkpoint": self.last_checkpoint_step,
            "node_id": ray.get_runtime_context().get_node_id()
        }


@ray.remote
class RebalancingManager:
    """
    Manages dynamic shard rebalancing across nodes.
    """

    def __init__(self, target_balance_ratio: float = 0.2):
        """
        Args:
            target_balance_ratio: Maximum allowed imbalance ratio
        """
        self.target_balance_ratio = target_balance_ratio
        self.rebalance_history = []

    def check_balance(self, shard_stats: List[Dict]) -> bool:
        """
        Check if shards need rebalancing.

        Args:
            shard_stats: Statistics from all shards

        Returns:
            True if rebalancing is needed
        """
        sizes = [s["size"] for s in shard_stats]
        if not sizes or len(sizes) < 2:
            return False

        avg_size = np.mean(sizes)
        max_deviation = max(abs(s - avg_size) for s in sizes)
        imbalance_ratio = max_deviation / (avg_size + 1e-8)

        return imbalance_ratio > self.target_balance_ratio

    def compute_rebalance_plan(self, shard_stats: List[Dict]) -> List[Dict]:
        """
        Compute data migration plan for rebalancing.

        Args:
            shard_stats: Statistics from all shards

        Returns:
            List of migration operations
        """
        sizes = {s["shard_id"]: s["size"] for s in shard_stats}
        avg_size = np.mean(list(sizes.values()))

        migrations = []

        # Find overloaded and underloaded shards
        overloaded = [(sid, size) for sid, size in sizes.items()
                     if size > avg_size * (1 + self.target_balance_ratio)]
        underloaded = [(sid, size) for sid, size in sizes.items()
                      if size < avg_size * (1 - self.target_balance_ratio)]

        # Plan migrations
        for source_id, source_size in overloaded:
            for target_id, target_size in underloaded:
                transfer_size = min(
                    source_size - avg_size,
                    avg_size - target_size
                )
                if transfer_size > 0:
                    migrations.append({
                        "source_shard": source_id,
                        "target_shard": target_id,
                        "num_samples": int(transfer_size)
                    })

        return migrations


class DistributedReplayBufferWithFaultTolerance:
    """
    Enhanced distributed replay buffer with fault tolerance,
    priority sampling, and dynamic rebalancing.
    """

    def __init__(self, capacity: int, batch_size: int,
                 num_shards: int = 4,
                 enable_priority: bool = True,
                 enable_checkpoint: bool = True,
                 enable_rebalancing: bool = False,
                 enable_fault_tolerance: bool = True,
                 **kwargs):
        self.capacity = capacity
        self.batch_size = batch_size
        self.num_shards = num_shards
        self.enable_priority = enable_priority
        self.enable_checkpoint = enable_checkpoint
        self.enable_rebalancing = enable_rebalancing
        self.enable_fault_tolerance = enable_fault_tolerance

        # Create enhanced shards
        self.shards = []
        capacity_per_shard = capacity // num_shards

        for i in range(num_shards):
            shard = FaultTolerantBufferShard.remote(
                shard_id=i,
                capacity=capacity_per_shard,
                enable_checkpoint=enable_checkpoint,
                **kwargs
            )
            self.shards.append(shard)

        # Optional rebalancing manager
        if enable_rebalancing:
            self.rebalancer = RebalancingManager.remote()

        logger.info(f"Fault-tolerant distributed buffer initialized: "
                   f"{num_shards} shards, priority={enable_priority}, "
                   f"checkpoint={enable_checkpoint}")

    def push_with_priority(self, batch: DataProto, global_step: int,
                          priority: Optional[float] = None) -> None:
        """Push with optional priority - stores individual samples, not batches"""

        # Get batch size from tensor dict
        batch_size = batch.batch.shape[0] if batch.batch is not None else 0

        if batch_size == 0:
            logger.warning("Empty batch received in push_with_priority")
            return

        # Split batch into individual samples and distribute to shards
        tensor_dict = batch.batch.to_dict()
        non_tensor = batch.non_tensor_batch

        for sample_idx in range(batch_size):
            # Extract single sample from batch
            single_sample_dict = {}
            for key, tensor in tensor_dict.items():
                if isinstance(tensor, torch.Tensor):
                    # Extract single sample (keeping all dimensions except batch)
                    single_sample_dict[key] = tensor[sample_idx]
                else:
                    single_sample_dict[key] = tensor[sample_idx] if hasattr(tensor, '__getitem__') else tensor

            # Extract corresponding non-tensor data
            single_non_tensor = {}
            if non_tensor:
                for key, values in non_tensor.items():
                    if isinstance(values, (list, np.ndarray)) and len(values) >= batch_size:
                        single_non_tensor[key] = values[sample_idx]
                    else:
                        # Keep as is if not indexable or wrong size
                        single_non_tensor[key] = values

            # Create meta info for single sample
            single_meta = batch.meta_info.copy() if batch.meta_info else {}
            single_meta["original_batch_size"] = batch_size
            single_meta["sample_index_in_batch"] = sample_idx

            # Select shard for this sample (round-robin)
            shard_idx = (global_step * batch_size + sample_idx) % self.num_shards
            shard = self.shards[shard_idx]

            # Prepare single sample data
            sample_data = (
                single_sample_dict,
                single_non_tensor,
                single_meta,
                global_step
            )

            # Push single sample to shard
            if self.enable_priority:
                shard.push_batch_with_priority.remote(sample_data, priority)
            else:
                shard.push_batch.remote(*sample_data)

    def sample_with_importance_weights(self, batch_size: int,
                                      beta: float = 0.4) -> Optional[DataProto]:
        """Sample with importance sampling weights"""
        if not self.enable_priority:
            raise ValueError("Priority sampling not enabled")

        samples_per_shard = batch_size // self.num_shards
        futures = []

        for shard in self.shards:
            future = shard.sample_with_priority.remote(
                samples_per_shard, beta
            )
            futures.append(future)

        results = ray.get(futures)

        # Combine results
        # ... (implementation similar to base sample method)

        return None  # Placeholder

    def update_priorities(self, indices: np.ndarray,
                         priorities: np.ndarray) -> None:
        """Update priorities after training"""
        if not self.enable_priority:
            return

        # Route updates to appropriate shards
        # ... (implementation needed)

    def checkpoint_all(self, global_step: int) -> bool:
        """Checkpoint all shards"""
        if not self.enable_checkpoint:
            return True

        futures = [
            shard.checkpoint.remote(global_step)
            for shard in self.shards
        ]
        results = ray.get(futures)

        success = all(results)
        if success:
            logger.info(f"All shards checkpointed at step {global_step}")
        else:
            logger.error("Some shards failed to checkpoint")

        return success

    def restore_all(self) -> bool:
        """Restore all shards from latest checkpoints"""
        futures = [
            shard.restore.remote()
            for shard in self.shards
        ]
        results = ray.get(futures)

        success = all(results)
        if success:
            logger.info("All shards restored from checkpoints")
        else:
            logger.error("Some shards failed to restore")

        return success

    def check_and_recover_failed_shards(self) -> None:
        """Check shard health and recover failed shards"""
        futures = [
            shard.get_health_status.remote()
            for shard in self.shards
        ]
        health_stats = ray.get(futures)

        for i, status in enumerate(health_stats):
            if status["status"] == ShardStatus.FAILED.value:
                logger.warning(f"Shard {i} has failed, attempting recovery...")

                # Try to restore from checkpoint
                success = ray.get(self.shards[i].restore.remote())

                if not success:
                    # Create new shard as last resort
                    logger.error(f"Recovery failed for shard {i}, creating new shard")
                    self.shards[i] = FaultTolerantBufferShard.remote(
                        shard_id=i,
                        capacity=self.capacity // self.num_shards,
                        enable_checkpoint=self.enable_checkpoint
                    )

    def rebalance_if_needed(self) -> None:
        """Check and perform rebalancing if needed"""
        if not self.enable_rebalancing:
            return

        # Get shard statistics
        futures = [
            shard.get_health_status.remote()
            for shard in self.shards
        ]
        stats = ray.get(futures)

        # Check if rebalancing is needed
        need_rebalance = ray.get(
            self.rebalancer.check_balance.remote(stats)
        )

        if need_rebalance:
            logger.info("Initiating shard rebalancing...")
            plan = ray.get(
                self.rebalancer.compute_rebalance_plan.remote(stats)
            )
            # Execute rebalancing plan
            # ... (implementation needed)

    # ===== Missing Methods Required by BaseReplayBuffer Interface =====

    def push_from_dataproto(self, batch, global_step: int) -> None:
        """
        Store data from a DataProto batch into the replay buffer.
        Wrapper for push_with_priority to match base interface.

        IMPORTANT: This method preserves ALL fields in the batch, including
        behavior_log_probs which is critical for off-policy monitoring.
        """
        # Verify critical fields are present and warn if missing
        if batch.batch is not None and "behavior_log_probs" not in batch.batch:
            logger.warning(f"push_from_dataproto: behavior_log_probs missing in batch at step {global_step}")

        # Use default priority if priority sampling is enabled
        default_priority = 1.0 if self.enable_priority else None
        self.push_with_priority(batch, global_step, priority=default_priority)

    def sample_for_training(self, batch_size=None, device='cpu',
                           tokenizer=None, sequence_length=4096,
                           sampling_mode="trajectory", steps_per_episode=1,
                           sample_method="uniform", candidates_per_group=1,
                           group_sampling="uniform"):
        """
        Sample a batch of data for training.
        Implements both priority and uniform sampling.
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Check if we have enough data
        if not self.can_sample(batch_size):
            logger.debug(f"Not enough data in buffer for batch_size={batch_size}")
            return None

        try:
            # Determine how many samples per shard
            samples_per_shard = max(1, batch_size // self.num_shards)
            remainder = batch_size % self.num_shards

            # Sample from each shard
            futures = []
            for i, shard in enumerate(self.shards):
                # Add remainder to first shards
                shard_batch_size = samples_per_shard + (1 if i < remainder else 0)
                if shard_batch_size > 0:
                    if self.enable_priority:
                        future = shard.sample_with_priority.remote(shard_batch_size, beta=0.4)
                    else:
                        future = shard.sample_uniform.remote(shard_batch_size)
                    futures.append(future)

            # Gather results
            results = ray.get(futures)

            # Filter out None results
            valid_results = [r for r in results if r is not None]

            if not valid_results:
                logger.debug("All shards returned None")
                return None

            # Combine results into a DataProto batch
            from roll.distributed.scheduler.protocol import DataProto
            from tensordict import TensorDict
            import torch

            # Initialize combined batch
            combined_batch = DataProto()

            # Aggregate tensor dicts
            tensor_dicts = []
            non_tensor_batches = []
            meta_infos = []

            for result in valid_results:
                if len(result) >= 3:
                    tensor_dict, non_tensor, meta = result[:3]
                    if tensor_dict:
                        tensor_dicts.append(tensor_dict)
                    if non_tensor:
                        non_tensor_batches.append(non_tensor)
                    if meta:
                        meta_infos.append(meta)

            # Combine tensor dicts
            if tensor_dicts:
                # Convert dicts to TensorDict and concatenate
                td_list = []
                for td in tensor_dicts:
                    if isinstance(td, dict):
                        # Convert to tensors if needed
                        tensor_data = {}
                        for k, v in td.items():
                            if not isinstance(v, torch.Tensor):
                                v = torch.tensor(v)
                            # Ensure all tensors have batch dimension
                            if v.dim() == 1:
                                v = v.unsqueeze(0)  # Add batch dimension if missing
                            elif v.dim() == 0:
                                v = v.unsqueeze(0)  # Scalar to batch size 1
                            tensor_data[k] = v
                        # Get actual batch size from first tensor
                        batch_dim = list(tensor_data.values())[0].shape[0] if tensor_data else 1
                        td_list.append(TensorDict(tensor_data, batch_size=[batch_dim]))
                    elif isinstance(td, TensorDict):
                        td_list.append(td)
                    else:
                        logger.warning(f"Unknown tensor dict type: {type(td)}")

                if td_list:
                    # Concatenate all TensorDicts along batch dimension
                    combined_td = torch.cat(td_list, dim=0)
                    combined_batch.batch = combined_td

                    # Log if behavior_log_probs is missing
                    if "behavior_log_probs" not in combined_td:
                        logger.warning("behavior_log_probs missing after combining tensor dicts in sample_for_training")

            # Combine non-tensor batches
            if non_tensor_batches:
                combined_non_tensor = {}
                for key in non_tensor_batches[0].keys():
                    combined_non_tensor[key] = []
                    for batch in non_tensor_batches:
                        if key in batch:
                            if isinstance(batch[key], list):
                                combined_non_tensor[key].extend(batch[key])
                            else:
                                combined_non_tensor[key].append(batch[key])
                combined_batch.non_tensor_batch = combined_non_tensor

            # Combine meta info
            if meta_infos:
                combined_meta = meta_infos[0].copy()
                if self.enable_priority and "importance_weights" in combined_meta:
                    # Aggregate importance weights
                    all_weights = []
                    for meta in meta_infos:
                        if "importance_weights" in meta:
                            all_weights.extend(meta["importance_weights"])
                    combined_meta["importance_weights"] = all_weights
                combined_batch.meta_info = combined_meta

            return combined_batch

        except Exception as e:
            logger.error(f"Error sampling from distributed buffer: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def can_sample(self, batch_size=None) -> bool:
        """
        Check if the buffer has enough data for sampling.

        For distributed buffer, we check if we have at least batch_size items total.
        """
        if batch_size is None:
            batch_size = self.batch_size

        try:
            # Get total size across all shards
            stats = self.get_stats()
            total_size = stats.get("total_size", 0)

            # We need at least batch_size items total
            return total_size >= batch_size
        except Exception as e:
            logger.debug(f"Error checking buffer size: {e}")
            # Conservative approach: assume we cannot sample if there's an error
            return False

    @property
    def buffer_type(self) -> str:
        """Return the type of buffer for identification."""
        return "DistributedReplayBufferWithFaultTolerance"

    def get_stats(self) -> Dict:
        """
        Get statistics about the buffer state.

        Returns:
            Dictionary with buffer statistics
        """
        try:
            # Gather stats from all shards
            stats_futures = [
                shard.get_health_status.remote()
                for shard in self.shards
            ]
            shard_stats = ray.get(stats_futures)

            # Aggregate statistics
            total_size = sum(s.get("size", 0) for s in shard_stats)
            total_capacity = sum(s.get("capacity", 0) for s in shard_stats)
            healthy_shards = sum(1 for s in shard_stats if s.get("status") == "healthy")

            # Calculate utilization
            utilization = total_size / total_capacity if total_capacity > 0 else 0.0

            return {
                "buffer_type": self.buffer_type,
                "total_size": total_size,
                "total_stored": total_size,  # Same as total_size for compatibility
                "capacity": total_capacity,  # Use 'capacity' for compatibility
                "total_capacity": total_capacity,
                "utilization": utilization,  # Buffer utilization ratio
                "num_shards": self.num_shards,
                "healthy_shards": healthy_shards,
                "enable_priority": self.enable_priority,
                "enable_fault_tolerance": self.enable_fault_tolerance,
                "shard_stats": shard_stats
            }
        except Exception as e:
            logger.error(f"Error getting buffer stats: {e}")
            return {
                "buffer_type": self.buffer_type,
                "total_stored": 0,  # Default value for compatibility
                "capacity": 0,  # Default value for compatibility
                "utilization": 0.0,  # Default value for compatibility
                "error": str(e)
            }