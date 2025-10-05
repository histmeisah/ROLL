"""
Distributed Replay Buffer using Ray for multi-machine training.

This implementation provides:
1. Sharded storage across multiple nodes
2. Efficient sampling and storage
3. Fault tolerance and monitoring
"""

import gc
import time
import hashlib
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

import ray
import torch
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger
from roll.utils.utils import pad_to_length
from .base_buffer import BaseReplayBuffer

logger = get_logger()


@dataclass
class BufferStats:
    """Statistics for monitoring buffer performance"""
    node_id: str
    shard_id: int
    total_stored: int
    current_size: int
    memory_usage_mb: float
    last_gc_time: float
    sample_count: int
    avg_sample_time_ms: float


@ray.remote
class ReplayBufferShard:
    """
    A single shard of the distributed replay buffer.
    Each shard runs on a different node and manages a portion of the data.
    """

    def __init__(self, shard_id: int, capacity: int, batch_size: int,
                 gc_interval: int = 100000, use_compression: bool = True):
        self.shard_id = shard_id
        self.capacity = capacity
        self.batch_size = batch_size
        self.gc_interval = gc_interval
        self.use_compression = use_compression

        # Storage
        self.trajectories = deque(maxlen=capacity)
        self.total_stored = 0

        # Monitoring
        self.sample_count = 0
        self.total_sample_time = 0
        self.last_gc_time = time.time()

        # Get node information
        self.node_id = ray.get_runtime_context().get_node_id()
        logger.info(f"ReplayBufferShard {shard_id} initialized on node {self.node_id}")

    def push_batch(self, batch_tensor_dict: Dict, batch_non_tensor: Dict,
                   batch_meta: Dict, global_step: int) -> bool:
        """
        Store a batch of trajectories in this shard.

        Args:
            batch_tensor_dict: Tensor data as dict
            batch_non_tensor: Non-tensor data
            batch_meta: Metadata
            global_step: Current training step

        Returns:
            bool: Success status
        """
        try:
            # Recreate TensorDict from serialized data
            batch_tensordict = TensorDict(batch_tensor_dict)
            batch_size = batch_tensordict.batch_size[0]

            for i in range(batch_size):
                # Extract single trajectory
                trajectory = {
                    "tensordict": batch_tensordict[i],
                    "non_tensor": {k: v[i] for k, v in batch_non_tensor.items()},
                    "global_step": global_step,
                    "timestamp": time.time()
                }

                # Apply compression if enabled
                if self.use_compression:
                    trajectory = self._compress_trajectory(trajectory)

                self.trajectories.append(trajectory)
                self.total_stored += 1

            # Periodic GC
            if self.total_stored % self.gc_interval == 0:
                self._perform_gc()

            return True

        except Exception as e:
            logger.error(f"Shard {self.shard_id} push error: {e}")
            return False

    def sample(self, n_samples: int, sample_method: str = "uniform") -> Optional[Tuple]:
        """
        Sample trajectories from this shard.

        Args:
            n_samples: Number of samples to retrieve
            sample_method: Sampling strategy

        Returns:
            Tuple of (tensor_dict, non_tensor_dict, meta_dict) or None
        """
        start_time = time.time()

        if len(self.trajectories) == 0:
            return None

        # Sample indices
        if sample_method == "uniform":
            indices = np.random.choice(len(self.trajectories),
                                     min(n_samples, len(self.trajectories)),
                                     replace=False)
        elif sample_method == "lifo":
            indices = range(max(0, len(self.trajectories) - n_samples),
                          len(self.trajectories))
        else:
            indices = range(min(n_samples, len(self.trajectories)))

        # Collect samples
        tensor_dicts = []
        non_tensor_dicts = []

        for idx in indices:
            traj = self.trajectories[idx]

            # Decompress if needed
            if self.use_compression:
                traj = self._decompress_trajectory(traj)

            tensor_dicts.append(traj["tensordict"])
            non_tensor_dicts.append(traj["non_tensor"])

        # Stack into batch
        if tensor_dicts:
            batch_tensordict = torch.stack(tensor_dicts)
            batch_non_tensor = self._stack_non_tensor(non_tensor_dicts)

            # Update stats
            self.sample_count += 1
            self.total_sample_time += (time.time() - start_time) * 1000

            # Convert to serializable format
            return (batch_tensordict.to_dict(), batch_non_tensor, {})

        return None

    def _compress_trajectory(self, trajectory: Dict) -> Dict:
        """Compress trajectory to save memory"""
        td = trajectory["tensordict"]

        # Compress token IDs to int16
        for key in ["input_ids", "input_encode_ids", "responses_ids"]:
            if key in td.keys():
                td[key] = td[key].to(torch.int16)

        # Compress scores to float16
        for key in ["advantages", "rewards", "returns", "values"]:
            if key in td.keys():
                td[key] = td[key].to(torch.float16)

        return trajectory

    def _decompress_trajectory(self, trajectory: Dict) -> Dict:
        """Decompress trajectory for use"""
        td = trajectory["tensordict"]

        # Restore token IDs to int64
        for key in ["input_ids", "input_encode_ids", "responses_ids"]:
            if key in td.keys():
                td[key] = td[key].to(torch.int64)

        # Restore scores to float32
        for key in ["advantages", "rewards", "returns", "values"]:
            if key in td.keys():
                td[key] = td[key].to(torch.float32)

        return trajectory

    def _stack_non_tensor(self, non_tensor_list: List[Dict]) -> Dict:
        """Stack non-tensor data into batch format"""
        result = {}
        if non_tensor_list:
            keys = non_tensor_list[0].keys()
            for key in keys:
                values = [d[key] for d in non_tensor_list]
                result[key] = np.array(values, dtype=object)
        return result

    def _perform_gc(self):
        """Perform garbage collection"""
        gc.collect()
        self.last_gc_time = time.time()
        logger.info(f"Shard {self.shard_id}: GC at {self.total_stored} trajectories")

    def get_stats(self) -> BufferStats:
        """Get shard statistics"""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        avg_sample_time = 0
        if self.sample_count > 0:
            avg_sample_time = self.total_sample_time / self.sample_count

        return BufferStats(
            node_id=self.node_id,
            shard_id=self.shard_id,
            total_stored=self.total_stored,
            current_size=len(self.trajectories),
            memory_usage_mb=memory_mb,
            last_gc_time=self.last_gc_time,
            sample_count=self.sample_count,
            avg_sample_time_ms=avg_sample_time
        )

    def clear(self):
        """Clear all data"""
        self.trajectories.clear()
        gc.collect()
        logger.info(f"Shard {self.shard_id} cleared")


@ray.remote
class ReplayBufferManager:
    """
    Manager/Coordinator for the distributed replay buffer.
    Handles routing, load balancing, and monitoring.
    """

    def __init__(self, num_shards: int, capacity_per_shard: int,
                 batch_size: int, **kwargs):
        self.num_shards = num_shards
        self.capacity_per_shard = capacity_per_shard
        self.total_capacity = num_shards * capacity_per_shard
        self.batch_size = batch_size

        # Create shards
        self.shards = []
        for i in range(num_shards):
            shard = ReplayBufferShard.remote(
                shard_id=i,
                capacity=capacity_per_shard,
                batch_size=batch_size,
                **kwargs
            )
            self.shards.append(shard)

        # Round-robin counter for load balancing
        self.push_counter = 0
        self.total_pushed = 0

        logger.info(f"ReplayBufferManager initialized with {num_shards} shards")

    def push(self, batch_tensor_dict: Dict, batch_non_tensor: Dict,
             batch_meta: Dict, global_step: int) -> bool:
        """
        Push batch to appropriate shard(s).

        Uses round-robin distribution for load balancing.
        """
        # Select shard using round-robin
        shard_idx = self.push_counter % self.num_shards
        shard = self.shards[shard_idx]

        # Async push to shard
        result = shard.push_batch.remote(
            batch_tensor_dict, batch_non_tensor,
            batch_meta, global_step
        )

        self.push_counter += 1
        self.total_pushed += 1

        # Log progress
        if self.total_pushed % 100 == 0:
            logger.info(f"Manager: Pushed {self.total_pushed} batches")

        return True  # Return immediately, don't wait for result

    def sample(self, batch_size: int, sample_method: str = "uniform") -> Optional[Tuple]:
        """
        Sample from all shards and combine results.

        Args:
            batch_size: Total batch size to sample
            sample_method: Sampling strategy

        Returns:
            Combined batch or None
        """
        # Calculate samples per shard
        samples_per_shard = batch_size // self.num_shards
        remainder = batch_size % self.num_shards

        # Async sample from all shards
        futures = []
        for i, shard in enumerate(self.shards):
            n_samples = samples_per_shard + (1 if i < remainder else 0)
            if n_samples > 0:
                future = shard.sample.remote(n_samples, sample_method)
                futures.append(future)

        # Wait for results
        results = ray.get(futures)

        # Filter out None results
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            return None

        # Combine results
        combined_tensor_dicts = []
        combined_non_tensor = []

        for tensor_dict, non_tensor_dict, _ in valid_results:
            combined_tensor_dicts.append(TensorDict(tensor_dict))
            combined_non_tensor.append(non_tensor_dict)

        # Stack all samples
        final_tensor_dict = torch.cat(combined_tensor_dicts, dim=0)
        final_non_tensor = self._combine_non_tensor_batches(combined_non_tensor)

        return (final_tensor_dict, final_non_tensor, {})

    def _combine_non_tensor_batches(self, batches: List[Dict]) -> Dict:
        """Combine multiple non-tensor batches"""
        if not batches:
            return {}

        result = {}
        keys = batches[0].keys()

        for key in keys:
            arrays = [batch[key] for batch in batches if key in batch]
            if arrays:
                result[key] = np.concatenate(arrays, axis=0)

        return result

    def get_all_stats(self) -> List[BufferStats]:
        """Get statistics from all shards"""
        futures = [shard.get_stats.remote() for shard in self.shards]
        stats = ray.get(futures)

        # Log summary
        total_stored = sum(s.total_stored for s in stats)
        total_memory = sum(s.memory_usage_mb for s in stats)
        avg_sample_time = np.mean([s.avg_sample_time_ms for s in stats
                                   if s.sample_count > 0] or [0])

        logger.info(f"Buffer Summary: {total_stored} trajectories, "
                   f"{total_memory:.1f} MB total memory, "
                   f"{avg_sample_time:.2f} ms avg sample time")

        return stats

    def clear_all(self):
        """Clear all shards"""
        futures = [shard.clear.remote() for shard in self.shards]
        ray.get(futures)
        self.push_counter = 0
        self.total_pushed = 0
        logger.info("All shards cleared")


class DistributedReplayBuffer(BaseReplayBuffer):
    """
    Client interface for the distributed replay buffer.
    Provides the same interface as local buffers but uses Ray actors.
    """

    def __init__(self, capacity: int, batch_size: int,
                 num_shards: int = 4, **kwargs):
        """
        Initialize distributed replay buffer.

        Args:
            capacity: Total capacity across all shards
            batch_size: Batch size for sampling
            num_shards: Number of shards to create
            **kwargs: Additional arguments for shards
        """
        super().__init__(capacity, batch_size)

        self.num_shards = num_shards
        self.capacity_per_shard = capacity // num_shards

        # Create manager
        self.manager = ReplayBufferManager.remote(
            num_shards=num_shards,
            capacity_per_shard=self.capacity_per_shard,
            batch_size=batch_size,
            **kwargs
        )

        # Async push tracking
        self.pending_pushes = deque(maxlen=1000)
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True
        )
        self._cleanup_thread.start()

        logger.info(f"DistributedReplayBuffer initialized with "
                   f"{num_shards} shards, {capacity} total capacity")

    def push_from_dataproto(self, batch: DataProto, global_step: int) -> None:
        """
        Push batch to distributed buffer (non-blocking).

        Args:
            batch: DataProto batch to store
            global_step: Current training step
        """
        # Serialize batch for transmission
        batch_tensor_dict = batch.batch.to_dict()
        batch_non_tensor = batch.non_tensor_batch.copy()
        batch_meta = batch.meta_info.copy()

        # Async push to manager
        future = self.manager.push.remote(
            batch_tensor_dict,
            batch_non_tensor,
            batch_meta,
            global_step
        )

        # Track pending push
        self.pending_pushes.append(future)

    def sample_for_training(self, batch_size: Optional[int] = None,
                           **kwargs) -> Optional[DataProto]:
        """
        Sample batch from distributed buffer (blocking).

        Args:
            batch_size: Batch size to sample
            **kwargs: Additional sampling arguments

        Returns:
            DataProto batch or None
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Sample from manager (blocking)
        result = ray.get(self.manager.sample.remote(
            batch_size,
            kwargs.get("sample_method", "uniform")
        ))

        if result is None:
            return None

        tensor_dict, non_tensor_dict, meta_dict = result

        # Create DataProto
        return DataProto(
            batch=tensor_dict,
            non_tensor_batch=non_tensor_dict,
            meta_info=meta_dict
        )

    def _cleanup_worker(self):
        """Background thread to clean up completed pushes"""
        while True:
            try:
                if len(self.pending_pushes) > 100:
                    # Check oldest pushes
                    ready = []
                    remaining = []

                    for future in list(self.pending_pushes):
                        try:
                            # Non-blocking check with timeout=0
                            ray.get(future, timeout=0)
                            ready.append(future)
                        except ray.GetTimeoutError:
                            remaining.append(future)
                        except Exception as e:
                            logger.error(f"Push failed: {e}")
                            ready.append(future)  # Remove failed pushes

                    # Update pending list
                    self.pending_pushes = deque(remaining, maxlen=1000)

                    if ready:
                        logger.debug(f"Cleaned {len(ready)} completed pushes")

                time.sleep(1.0)  # Check every second

            except Exception as e:
                logger.error(f"Cleanup thread error: {e}")
                time.sleep(5.0)

    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        stats = ray.get(self.manager.get_all_stats.remote())

        # Aggregate stats
        return {
            "num_shards": self.num_shards,
            "total_stored": sum(s.total_stored for s in stats),
            "total_capacity": self.capacity,
            "pending_pushes": len(self.pending_pushes),
            "shards": [
                {
                    "shard_id": s.shard_id,
                    "node_id": s.node_id,
                    "size": s.current_size,
                    "memory_mb": s.memory_usage_mb
                }
                for s in stats
            ]
        }

    def wait_pending_pushes(self, timeout: float = 10.0):
        """Wait for all pending pushes to complete"""
        if self.pending_pushes:
            try:
                ray.get(list(self.pending_pushes), timeout=timeout)
                self.pending_pushes.clear()
            except ray.GetTimeoutError:
                logger.warning(f"Timeout waiting for {len(self.pending_pushes)} pushes")

    def clear(self):
        """Clear all data"""
        ray.get(self.manager.clear_all.remote())
        self.pending_pushes.clear()