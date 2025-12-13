"""
Math Reasoning environment with integrated Bandit prompt selection.

This environment:
1. Samples mathematical problems from dataset
2. Uses centralized BanditActor to select prompts
3. Formats observations with selected prompts
4. Verifies answers and computes rewards
5. Updates bandit statistics

Key fix: Uses Ray Named Actor pattern to share BanditActor across distributed workers.
"""

import re
import ray
import pickle
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging

from roll.agentic.env.base import BaseEnv
from .config import MathReasoningBanditConfig, DEFAULT_BANDIT_ACTOR_NAME
from .dataset import MathDataset

logger = logging.getLogger(__name__)


def get_bandit_actor_by_name(actor_name: str = DEFAULT_BANDIT_ACTOR_NAME):
    """
    Get BanditActor using Ray Named Actor pattern.

    This is the key fix: Ray Named Actors can be accessed across different
    processes/workers using the actor name.

    Args:
        actor_name: Name of the Ray actor to retrieve

    Returns:
        Ray actor handle or None if not found
    """
    try:
        actor = ray.get_actor(actor_name)
        logger.info(f"Successfully retrieved BanditActor: {actor_name}")
        return actor
    except ValueError:
        # Actor doesn't exist yet
        logger.debug(f"BanditActor '{actor_name}' not found")
        return None
    except Exception as e:
        logger.warning(f"Failed to get BanditActor '{actor_name}': {e}")
        return None


# Global prompt template cache to avoid reloading across environment instances
_PROMPT_CACHE = {}


def load_prompt_templates(config_path: str, preset_name: str) -> Optional[List]:
    """
    Load prompt templates from YAML config file.

    Uses a global cache to avoid reloading templates across environment instances.

    Args:
        config_path: Path to prompt YAML config
        preset_name: Name of the preset to load

    Returns:
        List of PromptTemplate objects or None
    """
    global _PROMPT_CACHE

    if not config_path:
        return None

    cache_key = f"{config_path}:{preset_name}"
    if cache_key in _PROMPT_CACHE:
        logger.debug(f"Using cached prompt templates for preset '{preset_name}'")
        return _PROMPT_CACHE[cache_key]

    try:
        from roll.algorithms.bandit.prompt_loader import PromptLoader
        loader = PromptLoader(config_path)
        templates = loader.get_preset_prompts(preset_name)
        _PROMPT_CACHE[cache_key] = templates
        logger.info(f"Loaded {len(templates)} prompt templates from preset '{preset_name}' (cached)")
        return templates
    except Exception as e:
        logger.warning(f"Failed to load prompt templates: {e}")
        return None


# Global encoder cache to avoid loading SentenceTransformer multiple times
# across environment instances (which would cause memory issues and hangs)
_ENCODER_CACHE = {}


def load_problem_encoder(model_name_or_path: str):
    """
    Load SentenceTransformer encoder for problem embeddings.

    Uses a global cache to avoid loading the model multiple times
    across environment instances. This is critical because each
    environment worker creates multiple environment instances, and
    loading SentenceTransformer for each would cause memory exhaustion.

    Args:
        model_name_or_path: Model name or path

    Returns:
        SentenceTransformer model or None
    """
    global _ENCODER_CACHE

    if not model_name_or_path:
        return None

    # Check cache first
    if model_name_or_path in _ENCODER_CACHE:
        logger.debug(f"Using cached problem encoder: {model_name_or_path}")
        return _ENCODER_CACHE[model_name_or_path]

    try:
        from sentence_transformers import SentenceTransformer
        # Load on CPU to avoid GPU memory conflicts with vLLM
        encoder = SentenceTransformer(model_name_or_path, device='cpu')
        _ENCODER_CACHE[model_name_or_path] = encoder
        logger.info(f"Loaded problem encoder: {model_name_or_path} (cached, device=cpu)")
        return encoder
    except Exception as e:
        logger.warning(f"Failed to load problem encoder: {e}")
        return None


class MathReasoningBanditEnv(BaseEnv):
    """
    Math reasoning environment with bandit-based prompt selection.

    This environment integrates with a centralized BanditActor for prompt
    selection, enabling multi-armed bandit learning across distributed
    environment workers.

    Key feature: Uses Ray Named Actor pattern so that all distributed
    environment workers can access the same BanditActor.
    """

    def __init__(self, config: MathReasoningBanditConfig):
        """
        Initialize environment.

        Args:
            config: Environment configuration
        """
        super().__init__(config)
        self.config = config

        # Initialize bandit components
        self._init_bandit_components()

        # Initialize dataset
        self.dataset = MathDataset(
            dataset_name=config.dataset_name,
            split=config.dataset_split,
            seed=config.dataset_seed,
            dataset_path=config.dataset_path,
        )

        # Current episode state
        self.current_problem = None
        self.current_ground_truth = None
        self.current_embedding = None
        self.current_prompt_idx = None
        self.current_prompt_info = None
        self.current_prompt_name = None

        # Log initialization status
        self._log_init_status()

    def _init_bandit_components(self):
        """
        Initialize bandit components with multiple fallback strategies.

        Priority order:
        1. Use components passed in config (if any)
        2. Get BanditActor via Ray Named Actor
        3. Load prompt templates from config path
        4. Get EncoderActor via Ray Named Actor (preferred) or load locally
        """
        # 1. BanditActor
        self.bandit_actor = self.config.bandit_actor
        if self.bandit_actor is None:
            # Try Ray Named Actor pattern
            actor_name = getattr(self.config, 'bandit_actor_name', DEFAULT_BANDIT_ACTOR_NAME)
            self.bandit_actor = get_bandit_actor_by_name(actor_name)

        # 2. Prompt templates
        self.prompt_templates = self.config.prompt_templates
        if self.prompt_templates is None:
            # Load from config path
            config_path = getattr(self.config, 'prompt_config_path', None)
            preset_name = getattr(self.config, 'preset_name', 'diverse_5')
            if config_path:
                self.prompt_templates = load_prompt_templates(config_path, preset_name)

        # 3. Problem encoder - prefer EncoderActor (centralized) over local loading
        self.problem_encoder = self.config.problem_encoder
        self.encoder_actor = None  # Ray actor for centralized encoding

        if self.problem_encoder is None:
            # First try EncoderActor (centralized, avoids thread safety issues)
            try:
                from roll.algorithms.bandit.encoder_actor import get_encoder_actor_by_name
                self.encoder_actor = get_encoder_actor_by_name("encoder_actor_global")
                if self.encoder_actor is not None:
                    logger.info("Using centralized EncoderActor for problem encoding")
            except Exception as e:
                logger.debug(f"EncoderActor not available: {e}")

            # Fallback: load locally (not recommended for multi-threaded envs)
            if self.encoder_actor is None:
                encoder_model = getattr(self.config, 'encoder_model', None)
                if encoder_model:
                    logger.warning("Loading encoder locally (may cause issues in multi-threaded env)")
                    self.problem_encoder = load_problem_encoder(encoder_model)

        # Get context dimension for fallback embeddings
        self.context_dim = getattr(self.config, 'context_dim', 768)

    def _log_init_status(self):
        """Log the initialization status of bandit components."""
        status_parts = []

        if self.bandit_actor is not None:
            status_parts.append("BanditActor: ✓")
        else:
            status_parts.append("BanditActor: ✗ (random selection)")

        if self.prompt_templates is not None and len(self.prompt_templates) > 0:
            status_parts.append(f"Prompts: ✓ ({len(self.prompt_templates)} templates)")
        else:
            status_parts.append("Prompts: ✗ (default template)")

        if self.encoder_actor is not None:
            status_parts.append("Encoder: ✓ (EncoderActor)")
        elif self.problem_encoder is not None:
            status_parts.append("Encoder: ✓ (local)")
        else:
            status_parts.append("Encoder: ✗ (hash-based embedding)")

        logger.info(
            f"MathReasoningBanditEnv initialized: {', '.join(status_parts)} | "
            f"Dataset: {len(self.dataset)} problems"
        )

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Reset environment for a new episode.

        Steps:
        1. Sample a mathematical problem
        2. Encode problem to get context embedding
        3. Call BanditActor to select prompt
        4. Format observation with selected prompt

        Args:
            seed: Random seed (optional)

        Returns:
            observation: Formatted problem text
            info: Episode information
        """
        # 1. Sample problem from dataset
        problem_data = self.dataset.sample()
        self.current_problem = problem_data["problem"]
        self.current_ground_truth = problem_data["answer"]

        # 2. Encode problem to get context
        self.current_embedding = self._encode_problem(self.current_problem)

        # 3. Select prompt using BanditActor
        self._select_prompt()

        # 4. Format observation with selected prompt
        observation = self._format_observation()

        # 5. Prepare info
        info = {
            "prompt_idx": self.current_prompt_idx,
            "prompt_name": self.current_prompt_name,
            "ucb_value": self.current_prompt_info.get("ucb_value", 0.0),
            "predicted_reward": self.current_prompt_info.get("predicted_reward", 0.0),
            "confidence": self.current_prompt_info.get("confidence", 0.0),
            "problem_length": len(self.current_problem),
        }

        return observation, info

    def _encode_problem(self, problem: str) -> np.ndarray:
        """
        Encode problem text to embedding vector.

        Uses EncoderActor (centralized) if available, otherwise falls back
        to local encoder or hash-based embedding.

        Args:
            problem: Problem text

        Returns:
            Embedding vector
        """
        # Priority 1: Use centralized EncoderActor (preferred for distributed envs)
        if self.encoder_actor is not None:
            try:
                embedding = ray.get(self.encoder_actor.encode_numpy.remote(problem))
                return embedding.astype(np.float32)
            except Exception as e:
                logger.warning(f"EncoderActor encoding failed: {e}. Trying fallback.")

        # Priority 2: Use local encoder
        if self.problem_encoder is not None:
            try:
                embedding = self.problem_encoder.encode(problem)
                return embedding.astype(np.float32)
            except Exception as e:
                logger.warning(f"Local encoding failed: {e}. Using hash-based embedding.")

        # Fallback: hash-based pseudo-embedding
        hash_val = hash(problem)
        np.random.seed(abs(hash_val) % (2**32))
        return np.random.randn(self.context_dim).astype(np.float32)

    def _select_prompt(self):
        """Select prompt using BanditActor or fallback to random."""
        n_prompts = len(self.prompt_templates) if self.prompt_templates else 1

        if self.bandit_actor is not None and self.prompt_templates is not None:
            try:
                # Serialize embedding for Ray transmission
                embedding_bytes = pickle.dumps(self.current_embedding)

                # Remote call to BanditActor
                prompt_info = ray.get(
                    self.bandit_actor.select_arm.remote(embedding_bytes)
                )

                self.current_prompt_idx = prompt_info["arm_idx"]
                self.current_prompt_info = prompt_info
                self.current_prompt_name = self.prompt_templates[self.current_prompt_idx].name
                return

            except Exception as e:
                logger.warning(f"Bandit selection failed: {e}. Using random prompt.")

        # Fallback: random prompt selection
        if self.prompt_templates and len(self.prompt_templates) > 0:
            self.current_prompt_idx = np.random.randint(0, len(self.prompt_templates))
            self.current_prompt_name = self.prompt_templates[self.current_prompt_idx].name
        else:
            self.current_prompt_idx = 0
            self.current_prompt_name = "default"

        self.current_prompt_info = {
            "arm_idx": self.current_prompt_idx,
            "ucb_value": 0.0,
            "predicted_reward": 0.0,
            "confidence": 0.0,
        }

    def _format_observation(self) -> str:
        """
        Format observation using selected prompt template.

        Returns:
            Formatted problem text
        """
        if self.prompt_templates and self.current_prompt_idx < len(self.prompt_templates):
            prompt_template = self.prompt_templates[self.current_prompt_idx]
            # Use template's format method if available
            if hasattr(prompt_template, 'template'):
                try:
                    observation = prompt_template.template.format(problem=self.current_problem)
                except Exception as e:
                    logger.warning(f"Template formatting failed: {e}")
                    observation = f"{prompt_template.template}\n\n{self.current_problem}"
            else:
                observation = f"{prompt_template}\n\n{self.current_problem}"
        else:
            # Default format
            observation = f"{self.config.env_instruction}\n\n{self.current_problem}"

        return observation

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Execute action (LLM-generated answer).

        Steps:
        1. Extract answer from LLM output
        2. Verify answer against ground truth
        3. Compute reward
        4. Update BanditActor with reward
        5. Return results

        Args:
            action: LLM-generated text (full response)

        Returns:
            observation: Next observation (empty for terminal state)
            reward: Reward for this action
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # 1. Extract answer from LLM output
        extracted_answer = self._extract_answer(action)

        # 2. Verify answer and compute reward
        reward = self._verify_answer(extracted_answer, self.current_ground_truth)

        # 3. Update BanditActor (async)
        if self.bandit_actor is not None:
            try:
                embedding_bytes = pickle.dumps(self.current_embedding)
                ray.get(self.bandit_actor.update.remote(
                    self.current_prompt_idx,
                    embedding_bytes,
                    reward
                ))
            except Exception as e:
                logger.warning(f"Bandit update failed: {e}")

        # 4. Prepare metrics
        metrics = {
            "action_is_valid": 1.0,
            "action_is_effective": 1.0,
            "success": 1.0 if (reward >= self.config.reward_correct - 1e-6) else 0.0,
            "prompt_idx": float(self.current_prompt_idx),
            "reward": float(reward),
        }

        # 5. Math reasoning is single-step, always terminate
        terminated = True
        truncated = False

        # Next observation (not used since terminated=True)
        next_obs = ""

        info = {"metrics": metrics}

        return next_obs, reward, terminated, truncated, info

    def _extract_answer(self, text: str) -> str:
        """
        Extract answer from LLM-generated text.

        Tries multiple extraction patterns:
        1. \boxed{answer} (LaTeX)
        2. <answer>answer</answer> (XML)
        3. "Answer:" or "答案是："
        4. Last number in text

        Args:
            text: LLM-generated response

        Returns:
            Extracted answer string
        """
        # Try configured patterns
        for pattern in self.config.answer_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Clean up answer
                answer = self._clean_answer(answer)
                return answer

        # Fallback: extract last number
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]

        # Last resort: use last line
        lines = text.strip().split('\n')
        if lines:
            return self._clean_answer(lines[-1])

        return text.strip()

    def _clean_answer(self, answer: str) -> str:
        """
        Clean extracted answer.

        Args:
            answer: Raw extracted answer

        Returns:
            Cleaned answer
        """
        # Remove common formatting
        answer = answer.strip()
        answer = answer.replace('$', '')  # Remove dollar signs
        answer = answer.replace(',', '')  # Remove thousands separators
        answer = answer.replace(' ', '')  # Remove spaces

        return answer

    def _verify_answer(self, predicted: str, ground_truth: str) -> float:
        """
        Verify answer correctness.

        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            Reward value (0.0 to 1.0)
        """
        # Normalize both answers
        pred_norm = self._normalize_answer(predicted)
        gt_norm = self._normalize_answer(ground_truth)

        # Exact match
        if pred_norm == gt_norm:
            return self.config.reward_correct

        # Numerical matching (if enabled)
        if self.config.enable_numerical_matching:
            try:
                pred_num = float(pred_norm)
                gt_num = float(gt_norm)
                if abs(pred_num - gt_num) < self.config.numerical_tolerance:
                    return self.config.reward_correct
            except (ValueError, TypeError):
                pass

        # Partial matching (if enabled)
        if self.config.enable_partial_matching:
            if gt_norm in pred_norm or pred_norm in gt_norm:
                return self.config.reward_partial

        return self.config.reward_incorrect

    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for comparison.

        Args:
            answer: Answer string

        Returns:
            Normalized answer
        """
        answer = str(answer).strip().lower()
        answer = answer.replace(' ', '')
        answer = answer.replace(',', '')
        answer = answer.replace('$', '')
        return answer

    def render(self, mode: str = "text") -> str:
        """
        Render environment state.

        Args:
            mode: Rendering mode

        Returns:
            Rendered state string
        """
        if self.current_problem:
            return f"Problem: {self.current_problem}\nAnswer: {self.current_ground_truth}"
        return "No current problem"

    def parse_action(self, text: str) -> Dict[str, Any]:
        """
        Parse action from text.

        For math reasoning, we don't need special parsing since
        the LLM generates natural language answers.

        Args:
            text: Action text

        Returns:
            Parsed action dict
        """
        return {
            "action": text,
            "action_is_valid": True,
        }

    def get_all_actions(self):
        """
        Get all possible actions.

        For math reasoning, actions are open-ended (natural language).

        Returns:
            Empty list (open-ended action space)
        """
        return []
