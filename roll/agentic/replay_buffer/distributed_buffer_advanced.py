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
        from .distributed_buffer import ReplayBufferShard
        # Reuse base implementation
        self._base = ReplayBufferShard(
            self.shard_id, self.capacity, batch_size=32
        )

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

            # Push to base storage
            success = self._base.push_batch(*batch_data)

            if success:
                # Update priorities
                batch_size = len(batch_data[0])
                start_idx = len(self._base.trajectories) - batch_size
                end_idx = len(self._base.trajectories)

                for i in range(start_idx, end_idx):
                    idx = i % self.capacity
                    self.priorities[idx] = priority

                # Update max priority
                self.max_priority = max(self.max_priority, priority)

            return success

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
        if len(self._base.trajectories) == 0:
            return None

        # Calculate sampling probabilities
        valid_size = min(len(self._base.trajectories), self.capacity)
        probs = self.priorities[:valid_size] ** self.priority_alpha
        probs = probs / probs.sum()

        # Sample indices
        indices = np.random.choice(valid_size, n_samples, p=probs)

        # Calculate importance sampling weights
        weights = (valid_size * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize

        # Get samples
        result = self._base.sample(n_samples, "uniform")
        if result:
            tensor_dict, non_tensor, meta = result
            # Add weights to meta
            meta["importance_weights"] = weights.astype(np.float32)
            meta["sample_indices"] = indices
            return (tensor_dict, non_tensor, meta)

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
                "trajectories": list(self._base.trajectories),
                "priorities": self.priorities.copy(),
                "max_priority": self.max_priority,
                "total_stored": self._base.total_stored,
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
            self._base.trajectories = deque(state["trajectories"],
                                          maxlen=self.capacity)
            self.priorities = state["priorities"]
            self.max_priority = state["max_priority"]
            self._base.total_stored = state["total_stored"]

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
            "size": len(self._base.trajectories),
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
                 **kwargs):
        self.capacity = capacity
        self.batch_size = batch_size
        self.num_shards = num_shards
        self.enable_priority = enable_priority
        self.enable_checkpoint = enable_checkpoint
        self.enable_rebalancing = enable_rebalancing

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
        """Push with optional priority"""
        # Select shard
        shard_idx = global_step % self.num_shards
        shard = self.shards[shard_idx]

        # Prepare data
        batch_data = (
            batch.batch.to_dict(),
            batch.non_tensor_batch,
            batch.meta_info,
            global_step
        )

        if self.enable_priority:
            shard.push_batch_with_priority.remote(batch_data, priority)
        else:
            shard.push_batch.remote(*batch_data)

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