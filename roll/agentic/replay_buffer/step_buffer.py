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
        seed: int = 42
    ):
        super().__init__(capacity, batch_size, seed)
        self.steps = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        logger.info(f"Initialized StepReplayBuffer with capacity={capacity}")
    
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
                step_length=step_length
            )
            
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
    
    def sample_for_training(self, batch_size: Optional[int] = None, device: str = 'cpu',
                            tokenizer: Optional[PreTrainedTokenizer] = None, sequence_length: int = 4096,
                            sampling_mode: str = "trajectory", steps_per_episode: int = 1,
                            sample_method: str = "uniform", candidates_per_group: int = 1,
                            group_sampling: str = "uniform") -> Optional[DataProto]:
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
            
        Returns:
            DataProto batch matching StepEnvManager output format
        """
        sample_size = batch_size or self.batch_size
        
        if not self.can_sample(sample_size):
            logger.debug(f"Insufficient steps for sampling: {len(self.steps)} < {sample_size}")
            return None
        
        # Sample steps
        sampled_steps = self.rng.sample(list(self.steps), sample_size)
        
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
            "buffer_utilization": len(self.steps) / self.capacity
        }
        
        logger.debug(f"Sampled {sample_size} steps for training")
        return dataproto