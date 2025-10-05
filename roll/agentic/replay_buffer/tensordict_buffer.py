"""
TensorDict-based Replay Buffer for ROLL Framework

Implements an efficient replay buffer using TensorDict for optimal performance,
following ROLL's native data handling approach while optimizing memory usage.
"""

import logging
from typing import Optional, Dict, Any
import torch
from tensordict import TensorDict
from roll.distributed.scheduler.protocol import DataProto
from .base_buffer import BaseReplayBuffer

logger = logging.getLogger(__name__)


class TensorDictReplayBuffer(BaseReplayBuffer):
    """
    High-performance replay buffer using TensorDict with memory optimization.

    Key features:
    1. Stores data in TensorDict format (same as ROLL)
    2. Uses dynamic allocation instead of pre-allocation to save memory
    3. Stores only actual length data, not padded
    4. Efficient batch operations
    """

    def __init__(
        self,
        capacity: int = 100000,
        batch_size: int = 128,
        use_compression: bool = True,  # Use smaller dtypes
        device: str = 'cpu',
        gc_interval: int = 100000,
        seed: int = 42
    ):
        """
        Initialize TensorDict-based replay buffer.

        Args:
            capacity: Maximum number of trajectories/steps to store
            batch_size: Default batch size for sampling
            use_compression: Whether to use memory-efficient dtypes
            device: Device to store tensors ('cpu' or 'cuda')
            gc_interval: Interval for garbage collection
            seed: Random seed
        """
        super().__init__(capacity, batch_size, seed)

        self.use_compression = use_compression
        self.device = device
        self.gc_interval = gc_interval

        # Store batches as list of TensorDicts
        # This avoids pre-allocation while maintaining efficiency
        self.buffer = []
        self.buffer_size = 0  # Total number of samples

        # Metadata storage (lightweight)
        self.metadata_buffer = []

        # Statistics
        self.memory_usage_mb = 0

        logger.info(f"Initialized TensorDictReplayBuffer: capacity={capacity}, "
                   f"compression={use_compression}, device={device}")

    @property
    def buffer_type(self) -> str:
        return "tensordict"

    def _compress_dtype(self, tensor: torch.Tensor, field_name: str) -> torch.Tensor:
        """Compress tensor dtype to save memory if compression is enabled."""
        if not self.use_compression:
            return tensor

        # Optimize dtypes based on field type
        if field_name == "input_ids":
            # Vocabulary typically < 65536, use int16
            return tensor.to(torch.int16)
        elif field_name in ["attention_mask", "response_mask", "prompt_mask"]:
            # Boolean masks, already optimal
            return tensor.to(torch.bool)
        elif field_name in ["scores", "penalty", "old_log_probs"]:
            # Use float16 for values, sufficient precision
            return tensor.to(torch.float16)
        else:
            return tensor

    def push_from_dataproto(self, batch: DataProto, global_step: int) -> None:
        """
        Store batch data in TensorDict format.

        This matches ROLL's native format, avoiding conversions.
        """
        if batch.batch is None:
            return

        batch_size = batch.batch.batch_size[0]

        # Get actual sequence lengths to store only non-padded data
        if "attention_mask" in batch.batch:
            # Calculate actual lengths from attention mask
            actual_lengths = batch.batch["attention_mask"].sum(dim=1)
            max_actual_length = int(actual_lengths.max().item())
        else:
            max_actual_length = batch.batch["input_ids"].shape[1]

        # Create optimized TensorDict with only actual length data
        tensor_dict = {}
        for key, tensor in batch.batch.items():
            # Detach and move to CPU
            tensor_cpu = tensor.detach().cpu()

            # Truncate to actual length to save memory
            if len(tensor_cpu.shape) >= 2 and tensor_cpu.shape[1] > max_actual_length:
                if key == "old_log_probs":
                    # old_log_probs is seq_len-1
                    tensor_cpu = tensor_cpu[:, :max_actual_length-1]
                else:
                    tensor_cpu = tensor_cpu[:, :max_actual_length]

            # Apply compression if enabled
            tensor_cpu = self._compress_dtype(tensor_cpu, key)

            tensor_dict[key] = tensor_cpu

        # Add actual length information for later padding
        tensor_dict["actual_length"] = actual_lengths.cpu().to(torch.int16)

        # Create TensorDict batch
        td_batch = TensorDict(tensor_dict, batch_size=[batch_size])

        # Store the batch
        self.buffer.append(td_batch)
        self.buffer_size += batch_size
        self.total_stored += batch_size

        # Store metadata separately (lightweight)
        if batch.non_tensor_batch:
            self.metadata_buffer.append({
                "global_step": global_step,
                "batch_indices": list(range(len(self.metadata_buffer) * batch_size,
                                           (len(self.metadata_buffer) + 1) * batch_size)),
                "data": batch.non_tensor_batch
            })

        # Maintain capacity by removing old batches
        while self.buffer_size > self.capacity:
            removed_batch = self.buffer.pop(0)
            self.buffer_size -= removed_batch.batch_size[0]
            if self.metadata_buffer:
                self.metadata_buffer.pop(0)

        # Periodic garbage collection
        if self.total_stored % self.gc_interval == 0 and self.total_stored > 0:
            import gc
            gc.collect()
            if torch.cuda.is_available() and self.device == 'cuda':
                torch.cuda.empty_cache()
            logger.info(f"Replay buffer GC triggered at {self.total_stored} items stored")

        # Update memory usage estimate
        self._update_memory_usage()

        logger.debug(f"Stored batch of size {batch_size}, buffer now has {self.buffer_size} items")

    def can_sample(self, batch_size: Optional[int] = None) -> bool:
        """Check if buffer has enough data for sampling."""
        required_size = batch_size or self.batch_size
        return self.buffer_size >= required_size

    def sample_for_training(
        self,
        batch_size: Optional[int] = None,
        device: str = 'cpu',
        tokenizer: Optional[Any] = None,
        sequence_length: int = 4096,
        **kwargs
    ) -> Optional[DataProto]:
        """
        Sample a batch using efficient TensorDict operations.

        This is where we see the performance benefit of keeping TensorDict format.
        """
        sample_size = batch_size or self.batch_size

        if not self.can_sample(sample_size):
            return None

        # Collect samples from different batches
        sampled_tensordicts = []
        remaining = sample_size

        while remaining > 0:
            # Randomly select a batch
            batch_idx = torch.randint(0, len(self.buffer), (1,)).item()
            batch = self.buffer[batch_idx]
            batch_batch_size = batch.batch_size[0]

            # Sample indices from this batch
            n_samples = min(remaining, batch_batch_size)
            if n_samples == batch_batch_size:
                # Take the whole batch
                sampled_tensordicts.append(batch)
            else:
                # Sample subset
                indices = torch.randperm(batch_batch_size)[:n_samples]
                sampled_tensordicts.append(batch[indices])

            remaining -= n_samples

        # Concatenate all samples efficiently
        if len(sampled_tensordicts) == 1:
            sampled_batch = sampled_tensordicts[0]
        else:
            # TensorDict.cat is optimized for this operation
            sampled_batch = torch.cat(sampled_tensordicts, dim=0)

        # Decompress dtypes if needed
        if self.use_compression:
            decompressed = {}
            for key, tensor in sampled_batch.items():
                if key == "actual_length":
                    continue  # Skip metadata

                # Restore original dtypes
                if key == "input_ids":
                    decompressed[key] = tensor.to(torch.long)
                elif key in ["scores", "penalty", "old_log_probs"]:
                    decompressed[key] = tensor.to(torch.float32)
                else:
                    decompressed[key] = tensor

            sampled_batch.update(decompressed)

        # Pad to sequence_length if needed
        sampled_batch = self._pad_batch(sampled_batch, sequence_length, tokenizer)

        # Move to target device
        if device != 'cpu':
            sampled_batch = sampled_batch.to(device)

        # Remove actual_length from final output
        if "actual_length" in sampled_batch:
            del sampled_batch["actual_length"]

        # Create DataProto
        return DataProto(
            batch=sampled_batch,
            non_tensor_batch={},  # Could sample metadata if needed
            meta_info={
                "from_replay_buffer": True,
                "buffer_type": self.buffer_type,
                "sample_size": sample_size,
                "memory_usage_mb": self.memory_usage_mb
            }
        )

    def _pad_batch(self, batch: TensorDict, target_length: int, tokenizer: Optional[Any]) -> TensorDict:
        """Pad batch to target sequence length."""
        current_length = batch["input_ids"].shape[1]

        if current_length >= target_length:
            # Truncate if needed
            for key in batch.keys():
                if key == "actual_length":
                    continue
                if len(batch[key].shape) >= 2:
                    if key == "old_log_probs":
                        batch[key] = batch[key][:, :target_length-1]
                    else:
                        batch[key] = batch[key][:, :target_length]
        else:
            # Pad to target length
            pad_length = target_length - current_length
            pad_token_id = tokenizer.pad_token_id if tokenizer else 0

            padded = {}
            for key, tensor in batch.items():
                if key == "actual_length":
                    padded[key] = tensor
                    continue

                if len(tensor.shape) >= 2:
                    # Determine padding value
                    if key == "input_ids":
                        pad_value = pad_token_id
                    elif key in ["attention_mask", "response_mask", "prompt_mask"]:
                        pad_value = 0
                    elif key in ["scores", "penalty", "old_log_probs"]:
                        pad_value = 0.0
                    else:
                        pad_value = 0

                    # Apply padding
                    if key == "old_log_probs":
                        # old_log_probs is seq_len-1
                        padding = torch.full((tensor.shape[0], target_length - 1 - tensor.shape[1]),
                                           pad_value, dtype=tensor.dtype)
                    else:
                        padding = torch.full((tensor.shape[0], pad_length),
                                           pad_value, dtype=tensor.dtype)

                    padded[key] = torch.cat([tensor, padding], dim=1)
                else:
                    padded[key] = tensor

            batch.update(padded)

        return batch

    def _update_memory_usage(self):
        """Estimate current memory usage."""
        total_elements = 0
        total_bytes = 0

        for td_batch in self.buffer:
            for key, tensor in td_batch.items():
                elements = tensor.numel()
                bytes_per_element = tensor.element_size()
                total_elements += elements
                total_bytes += elements * bytes_per_element

        self.memory_usage_mb = total_bytes / (1024 * 1024)

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        stats = super().get_stats()
        stats.update({
            "memory_usage_mb": self.memory_usage_mb,
            "num_batches": len(self.buffer),
            "compression_enabled": self.use_compression,
            "device": self.device
        })
        return stats


class TensorDictTrajectoryBuffer(TensorDictReplayBuffer):
    """Trajectory-specific TensorDict buffer."""

    @property
    def buffer_type(self) -> str:
        return "tensordict_trajectory"


class TensorDictStepBuffer(TensorDictReplayBuffer):
    """Step-specific TensorDict buffer."""

    @property
    def buffer_type(self) -> str:
        return "tensordict_step"