"""
Trajectory-Level Replay Buffer for ROLL Framework

Specifically designed to work with TrajEnvManager data format.
Stores complete episodes as single units with episode-level rewards and penalties.
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
class TrajectoryEntry:
    """
    Single trajectory entry for trajectory-level replay buffer.
    Matches the output format of TrajEnvManager.
    """
    # Core data from DataProto
    input_ids: np.ndarray
    attention_mask: np.ndarray
    position_ids: np.ndarray
    response_mask: np.ndarray
    prompt_mask: np.ndarray
    scores: np.ndarray  # Token-level scores with episode reward on last response token
    penalty: float      # Episode-level penalty scalar
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
    
    # Storage metadata
    stored_at_step: int
    episode_length: int


class TrajectoryReplayBuffer(BaseReplayBuffer):
    """
    Replay buffer specialized for trajectory-level data from TrajEnvManager.
    
    Key Design Principles:
    1. Episode-level storage: complete trajectories as atomic units
    2. TrajEnvManager compatibility: matches exact data format
    3. Episode-level penalty handling: penalties are episode totals, not step-wise
    4. Efficient trajectory sampling: maintains episode boundaries
    """
    
    def __init__(
        self, 
        capacity: int = 100000,  # Number of complete trajectories
        batch_size: int = 128,   # Should match rollout batch size
        seed: int = 42
    ):
        super().__init__(capacity, batch_size, seed)
        self.trajectories = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        logger.info(f"Initialized TrajectoryReplayBuffer with capacity={capacity}")
    
    @property
    def buffer_type(self) -> str:
        return "trajectory"
    
    def push_from_dataproto(self, batch: DataProto, global_step: int) -> None:
        """
        Store trajectory data from TrajEnvManager.
        
        Args:
            batch: DataProto from TrajEnvManager containing complete episodes
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
                # If no behavior_log_probs, create zeros with next-token length (len(input_ids)-1)
                target_len = max(int(input_ids.shape[0]) - 1, 0)
                behavior_log_probs = np.zeros((target_len,), dtype=np.float32)
            
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
            
            # Calculate episode length from attention mask
            episode_length = int(attention_mask.sum())
            
            trajectory = TrajectoryEntry(
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
                stored_at_step=global_step,
                episode_length=episode_length
            )
            
            self.trajectories.append(trajectory)
            self.total_stored += 1
            
        logger.debug(f"Stored {batch_size} trajectories. Total stored: {len(self.trajectories)}")
    
    def can_sample(self, batch_size: Optional[int] = None) -> bool:
        """Check if buffer has enough trajectories for sampling."""
        required_size = batch_size or self.batch_size
        return len(self.trajectories) >= required_size
    
    def sample_for_training(self, batch_size: Optional[int] = None, device: str = 'cpu',
                            tokenizer: Optional[PreTrainedTokenizer] = None, sequence_length: int = 4096,
                            sampling_mode: str = "trajectory", steps_per_episode: int = 1,
                            sample_method: str = "uniform", candidates_per_group: int = 1,
                            group_sampling: str = "uniform") -> Optional[DataProto]:
        """
        Sample trajectories and reconstruct DataProto format.
        
        Args:
            batch_size: Number of trajectories to sample
            device: Target device for tensors ('cpu' or 'cuda')
            tokenizer: Tokenizer for text processing
            sequence_length: Maximum sequence length for padding/truncation
            sampling_mode: "trajectory" or "step" sampling mode (ignored, always trajectory)
            steps_per_episode: Number of steps per episode (ignored for trajectory mode)
            sample_method: Sampling method ("uniform", "weighted", etc.)
            candidates_per_group: Number of candidates per group
            group_sampling: Group sampling strategy ("uniform", etc.)
            
        Returns:
            DataProto batch matching TrajEnvManager output format
        """
        sample_size = batch_size or self.batch_size
        
        if not self.can_sample(sample_size):
            logger.debug(f"Insufficient trajectories for sampling: {len(self.trajectories)} < {sample_size}")
            return None
        
        # Sample trajectories
        sampled_trajectories = self.rng.sample(list(self.trajectories), sample_size)
        
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
        
        for i, traj in enumerate(sampled_trajectories):
            original_seq_len = len(traj.input_ids)
            
            # Apply consistent truncation/padding like RolloutScheduler
            effective_seq_len = min(original_seq_len, max_seq_len)
            
            # Use pad_to_length for consistent behavior with original ROLL env_manager
            traj_input_ids = torch.from_numpy(traj.input_ids)
            traj_attention_mask = torch.from_numpy(traj.attention_mask.astype(bool))
            traj_position_ids = torch.from_numpy(traj.position_ids)
            traj_response_mask = torch.from_numpy(traj.response_mask.astype(bool))
            traj_prompt_mask = torch.from_numpy(traj.prompt_mask.astype(bool))
            traj_scores = torch.from_numpy(traj.scores)
            
            batch_input_ids[i] = pad_to_length(traj_input_ids, max_seq_len, pad_token_id)
            # Keep attention_mask padded with 0 to match RolloutScheduler
            batch_attention_mask[i] = pad_to_length(traj_attention_mask, max_seq_len, 0)
            batch_position_ids[i] = pad_to_length(traj_position_ids, max_seq_len, 0)
            batch_response_mask[i] = pad_to_length(traj_response_mask, max_seq_len, 0)
            batch_prompt_mask[i] = pad_to_length(traj_prompt_mask, max_seq_len, 0)
            batch_scores[i] = pad_to_length(traj_scores, max_seq_len, 0.0)
            
            batch_penalties[i] = traj.penalty
            
            # Handle behavior_log_probs with pad_to_length to next-token length (sequence_length-1)
            traj_behavior_log_probs = torch.from_numpy(traj.behavior_log_probs)
            batch_old_log_probs[i] = pad_to_length(traj_behavior_log_probs, max_seq_len - 1, 0.0)
            
            # Collect non-tensor data
            env_ids.append(traj.env_id)
            group_ids.append(traj.group_id)
            messages_lists.append(traj.messages_list)
            tags.append(traj.tag)
            frames_lists.append(traj.frames)
            step_scores_lists.append(traj.step_scores)
            episode_scores_lists.append(traj.episode_scores)
            traj_group_ids.append(traj.traj_group_id)
            traj_ids.append(traj.traj_id)
        
        # Create DataProto in the exact format TrajEnvManager produces
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
        }
        
        dataproto.meta_info = {
            "from_replay_buffer": True,
            "buffer_type": "trajectory",
            "sample_size": sample_size,
            "buffer_utilization": len(self.trajectories) / self.capacity
        }
        
        logger.debug(f"Sampled {sample_size} trajectories for training")
        return dataproto
