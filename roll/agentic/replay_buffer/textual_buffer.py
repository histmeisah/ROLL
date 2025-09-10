"""
Step-Based Textual Replay Buffer for ROLL Framework

This implementation stores individual conversation steps rather than full trajectories,
using ROLL's efficient tokenization tools for optimal performance and memory usage.

Key Features:
1. Step-level storage: each conversation turn stored separately  
2. ROLL-native tokenization: uses token_ids_to_assistant_mask for efficiency
3. Hybrid data: stores both text and pre-tokenized data
4. Memory efficient: avoids duplicate context storage
5. Backward compatible: can reconstruct trajectory format for training
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Literal
from collections import deque
import random
import logging
import time
import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from roll.distributed.scheduler.protocol import DataProto
from roll.agentic.rollout.token_mask_utils import token_ids_to_assistant_mask, split_by_token

logger = logging.getLogger(__name__)


@dataclass
class StepTransition:
    """
    Individual conversation step transition with configurable storage modes.
    
    Storage modes:
    - hybrid: stores both text and tokens
    - text_only: stores only text (tokens computed during sampling)
    - tokens_only: stores only tokens (text not available)
    """
    # Context information (always stored for reconstruction)
    context_messages: List[Dict]     # Full conversation context up to this step
    response_message: Dict           # Current assistant response message
    
    # Text representations (stored based on storage_mode)
    context_text: Optional[str] = None      # Formatted context text (text_only, hybrid)
    response_text: Optional[str] = None     # Assistant response content (text_only, hybrid)
    
    # Tokenized data (stored based on storage_mode)
    input_ids: Optional[np.ndarray] = None          # Complete input sequence tokens (tokens_only, hybrid)
    attention_mask: Optional[np.ndarray] = None     # Attention mask (tokens_only, hybrid)
    response_mask: Optional[np.ndarray] = None      # Response prediction mask (tokens_only, hybrid)
    
    # Training signals (inherited from episode or computed per step)
    reward: float = 0.0
    advantage: float = 0.0
    old_log_prob: float = 0.0
    ref_log_prob: float = 0.0
    
    # Metadata
    episode_id: str = ""
    step_index: int = 0               # Position in conversation
    total_steps: int = 0             # Total steps in episode
    timestamp: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


class TextualReplayBuffer:
    """
    Step-Based Textual Replay Buffer using ROLL's efficient tokenization.
    
    Core Design Principles:
    1. Step-level storage: individual conversation turns rather than full trajectories
    2. ROLL-native efficiency: uses token_ids_to_assistant_mask for tokenization
    3. Memory optimization: avoids redundant context storage
    4. Backward compatibility: can reconstruct trajectory format for training
    5. High capacity: supports 1M+ steps with minimal memory footprint
    """
    
    def __init__(
        self, 
        capacity: int = 1000000,
        seed: int = 42,
        batch_size: int = 128,  # Should match rollout_batch_size
        storage_mode: Literal["hybrid", "text_only", "tokens_only"] = "hybrid",
        lazy_tokenization: bool = False
    ):
        """
        Initialize step-based textual replay buffer with configurable storage mode.
        
        Args:
            capacity: Maximum number of step transitions to store (default 1M)
            seed: Random seed for reproducible sampling
            batch_size: Default batch size for sampling (should match rollout_batch_size)
            storage_mode: Storage strategy - "hybrid", "text_only", or "tokens_only"
            lazy_tokenization: If True, tokenize only during sampling (only for text_only mode)
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.storage_mode = storage_mode
        self.lazy_tokenization = lazy_tokenization
        self.step_transitions: deque[StepTransition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        
        # Episode tracking for optional trajectory reconstruction
        self.episode_steps: Dict[str, List[int]] = {}  # episode_id -> step indices
        
        logger.info(f"Initialized StepBasedTextualReplayBuffer:")
        logger.info(f"  - Capacity: {capacity:,} step transitions")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Storage mode: {storage_mode}")
        logger.info(f"  - Lazy tokenization: {lazy_tokenization}")
    
    def push_from_dataproto(self, batch: DataProto, tokenizer) -> int:
        """
        Extract individual conversation steps from DataProto using ROLL's efficient tokenization.
        
        Uses ROLL's token_ids_to_assistant_mask for optimal performance and compatibility.
        Each trajectory is decomposed into individual conversation steps.
        
        Args:
            batch: Complete DataProto with all training keys computed
            tokenizer: ROLL tokenizer instance
            
        Returns:
            Number of step transitions added to buffer
        """
        if len(batch) == 0:
            logger.warning("Received empty batch for replay buffer storage")
            return 0
            
        # Extract messages from ROLL's native format
        messages_list = batch.non_tensor_batch.get('messages_list', [])
        if len(messages_list) != len(batch):
            logger.error(f"Missing messages_list: {len(messages_list)} != {len(batch)}")
            return 0
        
        # Extract basic episode rewards from scores (env_manager original format)
        episode_rewards = []
        if 'scores' in batch.batch:
            scores = batch.batch['scores'].cpu().numpy()
            for i in range(len(batch)):
                # Sum scores for this episode (basic reward)
                episode_reward = float(scores[i].sum()) if len(scores.shape) > 1 else float(scores[i])
                episode_rewards.append({'reward': episode_reward})
        else:
            # Fallback: use episode_scores from non_tensor_batch
            episode_scores = batch.non_tensor_batch.get('episode_scores', [])
            for score in episode_scores:
                episode_rewards.append({'reward': float(score)})
                
        timestamp = time.time()
        global_step = batch.meta_info.get('global_step', 0)
        
        total_steps_added = 0
        
        for episode_idx, messages in enumerate(messages_list):
            try:
                # Use ROLL's efficient tokenization (matching traj_env_manager.py)
                # Follow the exact same tokenization process as the original training
                lm_input_texts = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
                inputs = tokenizer(lm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False)
                
                token_ids = inputs.input_ids[0].tolist()
                token_ids_split = split_by_token(token_ids, token_ids[0])
                
                response_masks_list = token_ids_to_assistant_mask(
                    messages=messages, 
                    input_ids_list=token_ids_split, 
                    tokenizer=tokenizer
                )
                token_ids_list = token_ids_split
                
                # Extract individual conversation steps
                episode_steps = self._extract_conversation_steps(
                    messages=messages,
                    token_ids_list=token_ids_list,
                    response_masks_list=response_masks_list,
                    episode_idx=episode_idx,
                    episode_signals=episode_rewards[episode_idx],
                    timestamp=timestamp,
                    global_step=global_step
                )
                
                # Add steps to buffer
                episode_id = f"ep_{episode_idx}_{int(timestamp)}"
                step_indices = []
                
                for step in episode_steps:
                    step.episode_id = episode_id
                    self.step_transitions.append(step)
                    step_indices.append(len(self.step_transitions) - 1)
                    total_steps_added += 1
                
                # Track episode for optional reconstruction
                self.episode_steps[episode_id] = step_indices
                
                logger.debug(f"Episode {episode_idx}: extracted {len(episode_steps)} conversation steps")
                
            except Exception as e:
                logger.warning(f"Failed to extract steps from episode {episode_idx}: {e}")
                continue
        
        logger.info(f"Added {total_steps_added} step transitions from {len(messages_list)} episodes "
                   f"(total steps in buffer: {len(self.step_transitions)})")
        return total_steps_added
    
    
    def _extract_conversation_steps(self, messages: List[Dict], token_ids_list: List[List[int]], 
                                  response_masks_list: List[List[int]], episode_idx: int,
                                  episode_signals: Dict, timestamp: float, global_step: int) -> List[StepTransition]:
        """
        Extract individual conversation steps from tokenized messages.
        
        Uses ROLL's pre-tokenized data to create step transitions efficiently.
        """
        steps = []
        context_messages = []
        context_tokens = []
        context_masks = []
        
        for msg_idx, (msg, tokens, masks) in enumerate(zip(messages, token_ids_list, response_masks_list)):
            # Add current message to context
            context_messages.append(msg)
            context_tokens.extend(tokens)
            context_masks.extend(masks)
            
            # Create step when we encounter an assistant message
            if msg['role'] == 'assistant':
                # Create step based on storage mode
                step = self._create_step_transition(
                    context_messages=context_messages.copy(),
                    response_message=msg,
                    context_tokens=context_tokens,
                    context_masks=context_masks,
                    episode_signals=episode_signals,
                    step_index=len(steps),
                    total_steps=sum(1 for m in messages if m['role'] == 'assistant'),
                    timestamp=timestamp,
                    episode_idx=episode_idx,
                    global_step=global_step
                )
                steps.append(step)
        
        return steps
    
    def _create_step_transition(self, context_messages: List[Dict], response_message: Dict, 
                              context_tokens: List[int], context_masks: List[int],
                              episode_signals: Dict, step_index: int, total_steps: int,
                              timestamp: float, episode_idx: int, global_step: int) -> StepTransition:
        """Create step transition based on configured storage mode."""
        
        # Always store context messages for reconstruction
        # Only store basic reward - other training signals will be computed by pipeline
        base_step = StepTransition(
            context_messages=context_messages,
            response_message=response_message,
            reward=episode_signals['reward'],
            advantage=0.0,  # Will be computed by pipeline
            old_log_prob=0.0,  # Will be computed by pipeline
            ref_log_prob=0.0,  # Will be computed by pipeline
            episode_id="",  # Will be set later
            step_index=step_index,
            total_steps=total_steps,
            timestamp=timestamp,
            meta={'episode_idx': episode_idx, 'global_step': global_step}
        )
        
        # Storage mode specific logic
        if self.storage_mode in ["hybrid", "text_only"]:
            # Store text representations
            base_step.context_text = self._format_conversation_context(context_messages[:-1])
            base_step.response_text = response_message['content']
        
        if self.storage_mode in ["hybrid", "tokens_only"]:
            # Store tokenized data
            base_step.input_ids = np.array(context_tokens, dtype=np.int32)
            base_step.attention_mask = np.array([1] * len(context_tokens), dtype=np.bool_)
            base_step.response_mask = np.array(context_masks, dtype=np.bool_)
        
        return base_step
    
    def _format_conversation_context(self, context_messages: List[Dict]) -> str:
        """Format conversation context messages into a readable string."""
        if not context_messages:
            return ""
        
        formatted_parts = []
        for msg in context_messages:
            role = msg['role'].capitalize()
            content = msg['content'].strip()
            formatted_parts.append(f"{role}: {content}")
        
        return "\n".join(formatted_parts)
    
    def sample_for_training(self, batch_size: Optional[int] = None, device: str = 'cpu', tokenizer=None, sequence_length: int = 4096,
                             sampling_mode: str = "trajectory", steps_per_episode: int = 1) -> Optional[DataProto]:
        """
        Sample step transitions for training based on storage mode.
        
        Args:
            batch_size: Number of step transitions to sample
            device: Target device for tensors
            tokenizer: Required for text_only mode with lazy_tokenization
            
        Returns:
            DataProto ready for training
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.step_transitions) < batch_size:
            logger.debug(f"Insufficient step samples: {len(self.step_transitions)} < {batch_size}")
            return None
        
        # Select sampling mode
        if sampling_mode == "step":
            try:
                sampled_steps = self.rng.sample(list(self.step_transitions), batch_size)
            except ValueError as e:
                logger.error(f"Step sampling error: {e}")
                return None
        elif sampling_mode == "trajectory":
            # Sample episodes, then within each episode take up to steps_per_episode assistant steps
            # Build index by episode_id
            episode_to_steps: Dict[str, List[StepTransition]] = {}
            for step in self.step_transitions:
                episode_to_steps.setdefault(step.episode_id, []).append(step)
            if not episode_to_steps:
                return None
            # Randomize episode order
            episode_ids = list(episode_to_steps.keys())
            self.rng.shuffle(episode_ids)
            sampled_steps = []
            for ep_id in episode_ids:
                steps = episode_to_steps[ep_id]
                # sort by step_index to maintain order, then take head
                steps_sorted = sorted(steps, key=lambda s: s.step_index)
                take = min(steps_per_episode, len(steps_sorted))
                sampled_steps.extend(steps_sorted[:take])
                if len(sampled_steps) >= batch_size:
                    break
            if len(sampled_steps) > batch_size:
                sampled_steps = sampled_steps[:batch_size]
            if len(sampled_steps) < batch_size:
                logger.debug(f"Trajectory sampling returned {len(sampled_steps)} < requested {batch_size}")
        else:
            logger.error(f"Unknown sampling_mode: {sampling_mode}")
            return None
        
        # Handle different storage modes
        if self.storage_mode == "tokens_only":
            return self._sample_tokens_only(sampled_steps, batch_size, device)
        elif self.storage_mode == "text_only":
            return self._sample_text_only(sampled_steps, batch_size, device, tokenizer)
        else:  # hybrid - use env_worker compatible output
            return self._sample_hybrid_as_env_worker(sampled_steps, batch_size, device, sequence_length, tokenizer)
    
    def _sample_tokens_only(self, sampled_steps: List[StepTransition], batch_size: int, device: str) -> DataProto:
        """Sample for tokens_only storage mode."""
        max_len = max(step.input_ids.shape[0] for step in sampled_steps)
        
        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        response_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        
        for i, step in enumerate(sampled_steps):
            seq_len = step.input_ids.shape[0]
            input_ids[i, :seq_len] = torch.from_numpy(step.input_ids)
            attention_mask[i, :seq_len] = torch.from_numpy(step.attention_mask)
            response_mask[i, :seq_len] = torch.from_numpy(step.response_mask)
        
        # Training signals
        rewards = torch.tensor([step.reward for step in sampled_steps], dtype=torch.float32, device=device)
        advantages = torch.tensor([step.advantage for step in sampled_steps], dtype=torch.float32, device=device)
        old_log_probs = torch.tensor([step.old_log_prob for step in sampled_steps], dtype=torch.float32, device=device)
        ref_log_probs = torch.tensor([step.ref_log_prob for step in sampled_steps], dtype=torch.float32, device=device)
        
        # Expand to sequence length (ROLL format requirement)
        old_log_probs_expanded = old_log_probs.unsqueeze(-1).expand(-1, max_len)
        ref_log_probs_expanded = ref_log_probs.unsqueeze(-1).expand(-1, max_len)
        
        return DataProto.from_dict(
            tensors={
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'response_mask': response_mask,
                'response_level_rewards': rewards,
                'advantages': advantages,
                'old_log_probs': old_log_probs_expanded,
                'ref_log_probs': ref_log_probs_expanded,
            },
            non_tensors={
                'step_indices': [step.step_index for step in sampled_steps],
                'episode_ids': [step.episode_id for step in sampled_steps],
            },
            meta_info={
                'from_replay_buffer': True,
                'sampling_method': 'tokens_only',
                'storage_mode': self.storage_mode
            }
        )
    
    def _sample_text_only(self, sampled_steps: List[StepTransition], batch_size: int, device: str, tokenizer) -> DataProto:
        """Sample for text_only storage mode - tokenize on demand."""
        if tokenizer is None:
            raise ValueError("Tokenizer required for text_only storage mode")
        
        # Reconstruct messages and tokenize
        all_input_ids = []
        all_attention_masks = []
        all_response_masks = []
        
        for step in sampled_steps:
            # Use stored context_messages to recreate tokenization (matching traj_env_manager.py)
            # Follow the exact same tokenization process as the original training
            lm_input_texts = tokenizer.apply_chat_template(step.context_messages, add_generation_prompt=False, tokenize=False)
            inputs = tokenizer(lm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False)
            
            token_ids = inputs.input_ids[0].tolist()
            token_ids_split = split_by_token(token_ids, token_ids[0])
            
            response_masks_list = token_ids_to_assistant_mask(
                messages=step.context_messages, 
                input_ids_list=token_ids_split, 
                tokenizer=tokenizer
            )
            token_ids_list = token_ids_split
            
            # Flatten tokens and masks
            input_ids = []
            response_mask = []
            for tokens, masks in zip(token_ids_list, response_masks_list):
                input_ids.extend(tokens)
                response_mask.extend(masks)
            
            all_input_ids.append(np.array(input_ids, dtype=np.int32))
            all_attention_masks.append(np.array([1] * len(input_ids), dtype=np.bool_))
            all_response_masks.append(np.array(response_mask, dtype=np.bool_))
        
        # Convert to tensors and pad
        max_len = max(len(ids) for ids in all_input_ids)
        
        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        response_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        
        for i, (ids, att_mask, resp_mask) in enumerate(zip(all_input_ids, all_attention_masks, all_response_masks)):
            seq_len = len(ids)
            input_ids[i, :seq_len] = torch.from_numpy(ids)
            attention_mask[i, :seq_len] = torch.from_numpy(att_mask)
            response_mask[i, :seq_len] = torch.from_numpy(resp_mask)
        
        # Training signals
        rewards = torch.tensor([step.reward for step in sampled_steps], dtype=torch.float32, device=device)
        advantages = torch.tensor([step.advantage for step in sampled_steps], dtype=torch.float32, device=device)
        old_log_probs = torch.tensor([step.old_log_prob for step in sampled_steps], dtype=torch.float32, device=device)
        ref_log_probs = torch.tensor([step.ref_log_prob for step in sampled_steps], dtype=torch.float32, device=device)
        
        # Expand to sequence length
        old_log_probs_expanded = old_log_probs.unsqueeze(-1).expand(-1, max_len)
        ref_log_probs_expanded = ref_log_probs.unsqueeze(-1).expand(-1, max_len)
        
        return DataProto.from_dict(
            tensors={
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'response_mask': response_mask,
                'response_level_rewards': rewards,
                'advantages': advantages,
                'old_log_probs': old_log_probs_expanded,
                'ref_log_probs': ref_log_probs_expanded,
            },
            non_tensors={
                'context_texts': [step.context_text for step in sampled_steps],
                'responses': [step.response_text for step in sampled_steps],
                'step_indices': [step.step_index for step in sampled_steps],
                'episode_ids': [step.episode_id for step in sampled_steps],
            },
            meta_info={
                'from_replay_buffer': True,
                'sampling_method': 'text_only_lazy_tokenization',
                'storage_mode': self.storage_mode
            }
        )
    
    def _sample_hybrid_as_env_worker(self, sampled_steps: List[StepTransition], batch_size: int, 
                                    device: str, sequence_length: int, tokenizer) -> DataProto:
        """
        Sample stored step transitions and return original dynamic-length data.
        
        ✅ PADDING MOVED TO PIPELINE: No longer padding here, let pipeline handle unified padding.
        This acts as environment buffer providing raw pre-computed data to ROLL's training pipeline.
        """
        from roll.distributed.scheduler.protocol import TensorDict
        
        # Collect stored tokenized data (avoid re-tokenization)
        all_input_ids = []
        all_attention_mask = []
        all_response_mask = []
        
        # Non-tensor data for ROLL pipeline
        messages_list = []
        env_ids = []
        group_ids = []
        tags = []
        frames = []
        episode_scores = []
        
        for step in sampled_steps:
            # Use stored pre-tokenized data (no re-tokenization!)
            # ✅ PADDING MOVED TO PIPELINE: Return original dynamic lengths
            # ⚠️ FIX DTYPE: Convert int32 to int64 for PyTorch compatibility
            input_ids = torch.from_numpy(step.input_ids).long().unsqueeze(0).to(device)
            attention_mask = torch.from_numpy(step.attention_mask).unsqueeze(0).to(device)
            response_mask = torch.from_numpy(step.response_mask).unsqueeze(0).to(device)
            
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_response_mask.append(response_mask)
            
            # Collect metadata (mimic env_manager format)
            messages_list.append(step.context_messages)
            env_ids.append(f"replay_{step.episode_id}_{step.step_index}")
            group_ids.append("replay_group") 
            tags.append("replay")
            frames.append([])
            episode_scores.append(step.reward)
        
        # Handle variable-length tensors by padding to sequence_length (unified padding)
        # This ensures compatibility with fresh batch data that's also padded to sequence_length
        max_len = sequence_length
        
        # Pad all tensors to max_len for concatenation
        padded_input_ids = []
        padded_attention_mask = []
        padded_response_mask = []
        
        for input_ids, attention_mask, response_mask in zip(all_input_ids, all_attention_mask, all_response_mask):
            current_len = input_ids.size(1)
            if current_len < max_len:
                # Pad to max_len for concatenation
                pad_length = max_len - current_len
                input_ids = torch.nn.functional.pad(input_ids, (0, pad_length), value=tokenizer.pad_token_id)
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_length), value=0)
                response_mask = torch.nn.functional.pad(response_mask, (0, pad_length), value=0)
            
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
            padded_response_mask.append(response_mask)
        
        # Batch tensors 
        batch_input_ids = torch.cat(padded_input_ids, dim=0)
        batch_attention_mask = torch.cat(padded_attention_mask, dim=0)
        batch_response_mask = torch.cat(padded_response_mask, dim=0)
        
        # Create position_ids and other required fields (following env_manager logic)
        batch_position_ids = batch_attention_mask.cumsum(dim=-1)
        # Fix prompt_mask to exclude padding tokens
        batch_prompt_mask = batch_attention_mask.bool() & torch.logical_not(batch_response_mask.bool())
        
        # Create score tensor with rewards at last response tokens
        batch_scores = torch.zeros_like(batch_input_ids, dtype=torch.float, device=device)
        for i, step in enumerate(sampled_steps):
            response_indices = torch.where(batch_response_mask[i])[0]
            if len(response_indices) > 0:
                last_response_idx = response_indices[-1].item()
                batch_scores[i, last_response_idx] = step.reward
        
        # Create penalty tensor (default zeros like env_manager)
        batch_penalty = torch.zeros(batch_size, dtype=torch.float, device=device)
        
        # Construct DataProto exactly like env_manager
        lm_input = DataProto()
        lm_input.batch = TensorDict(
            {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
                "position_ids": batch_position_ids,
                "response_mask": batch_response_mask,
                "prompt_mask": batch_prompt_mask,
                "scores": batch_scores,
                "penalty": batch_penalty,
            },
            batch_size=batch_size
        )
        
        # Add non-tensor batch data (exactly like env_manager)
        # Create trajectory IDs following env_manager pattern
        traj_group_ids = []
        traj_ids = []
        for step in sampled_steps:
            traj_group_id = f"replay_replay_group_{step.episode_id}_42"  # Following env_manager pattern
            traj_id = f"{traj_group_id}_replay_{step.episode_id}_{step.step_index}"
            traj_group_ids.append(traj_group_id)
            traj_ids.append(traj_id)
        
        lm_input.non_tensor_batch = {
            "messages_list": np.array(messages_list, dtype=object),
            "env_ids": np.array(env_ids, dtype=object),
            "group_ids": np.array(group_ids, dtype=object),
            "tags": np.array(tags, dtype=object),
            "frames": np.array(frames, dtype=object),
            "step_scores": np.array([[step.reward] for step in sampled_steps], dtype=object),
            "episode_scores": np.array(episode_scores, dtype=object),
            "traj_group_id": np.array(traj_group_ids, dtype=object),  # Missing field
            "traj_id": np.array(traj_ids, dtype=object),  # Missing field
        }
        
        # Add meta info
        lm_input.meta_info = {
            "from_replay_buffer": True,
            "replay_step_count": len(sampled_steps),
            "storage_mode": self.storage_mode
        }
        
        return lm_input
    
    def sample_for_text_ops(self, batch_size: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Sample step transitions for text operations: return context-response pairs.
        
        Args:
            batch_size: Number of text samples to return
            
        Returns:
            List of (context_text, response_text) pairs from individual steps
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.step_transitions) < batch_size:
            return []
        
        try:
            sampled_steps = self.rng.sample(list(self.step_transitions), batch_size)
            return [(step.context_text, step.response_text) for step in sampled_steps]
        except ValueError:
            return []
    
    def sample_episode_trajectory(self, episode_id: str) -> Optional[List[StepTransition]]:
        """
        Sample a complete episode trajectory by episode ID.
        
        Args:
            episode_id: The episode ID to reconstruct
            
        Returns:
            List of step transitions in chronological order, or None if not found
        """
        if episode_id not in self.episode_steps:
            return None
        
        step_indices = self.episode_steps[episode_id]
        
        # Filter out indices that may have been evicted from deque
        valid_steps = []
        for idx in step_indices:
            # Check if index is still valid (deque may have rotated)
            if idx < len(self.step_transitions):
                step = self.step_transitions[idx]
                if step.episode_id == episode_id:
                    valid_steps.append(step)
        
        # Sort by step index to maintain chronological order
        valid_steps.sort(key=lambda x: x.step_index)
        return valid_steps if valid_steps else None
    
    def can_sample(self, min_size: Optional[int] = None) -> bool:
        """Check if buffer has enough step samples for training."""
        min_size = min_size or self.batch_size
        return len(self.step_transitions) >= min_size
    
    def __len__(self) -> int:
        """Return current number of stored step transitions."""
        return len(self.step_transitions)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics for monitoring."""
        if not self.step_transitions:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'episodes': 0,
                'steps_per_episode': 0.0
            }
        
        recent_steps = list(self.step_transitions)[-1000:]  # Last 1000 samples
        
        # Calculate episode statistics
        unique_episodes = len(set(step.episode_id for step in recent_steps))
        steps_per_episode = len(recent_steps) / max(unique_episodes, 1)
        episode_ids = [step.episode_id for step in self.step_transitions]
        episode_counts: Dict[str, int] = {}
        for eid in episode_ids:
            episode_counts[eid] = episode_counts.get(eid, 0) + 1
        avg_steps_per_ep_all = float(np.mean(list(episode_counts.values()))) if episode_counts else 0.0
        
        return {
            'size': len(self.step_transitions),
            'capacity': self.capacity,
            'utilization': len(self.step_transitions) / self.capacity,
            'episodes': len(self.episode_steps),
            'steps_per_episode': steps_per_episode,
            'avg_steps_per_episode_all': avg_steps_per_ep_all,
            'avg_reward': np.mean([step.reward for step in recent_steps]),
            'avg_advantage': np.mean([step.advantage for step in recent_steps]),
            'batch_size': self.batch_size,
            'storage_mode': 'step_based'
        }
