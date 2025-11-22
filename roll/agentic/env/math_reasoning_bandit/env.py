"""
Math Reasoning environment with integrated Bandit prompt selection.

This environment:
1. Samples mathematical problems from dataset
2. Uses centralized BanditActor to select prompts
3. Formats observations with selected prompts
4. Verifies answers and computes rewards
5. Updates bandit statistics
"""

import re
import ray
import pickle
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

from roll.agentic.env.base import BaseEnv
from .config import MathReasoningBanditConfig
from .dataset import MathDataset

logger = logging.getLogger(__name__)


def get_global_bandit_actor():
    """
    Get global bandit actor from startup script.

    This allows environments to access the BanditActor even when it cannot
    be passed through config due to OmegaConf limitations.
    """
    try:
        import sys
        # Try to get from start_bandit_aime module
        if 'experiments.bandit_aime_reasoning.start_bandit_aime' in sys.modules:
            module = sys.modules['experiments.bandit_aime_reasoning.start_bandit_aime']
            return getattr(module, 'GLOBAL_BANDIT_ACTOR', None)
        # Try to get from start_bandit_math_reasoning module
        if 'experiments.bandit_math_reasoning.start_bandit_math_reasoning' in sys.modules:
            module = sys.modules['experiments.bandit_math_reasoning.start_bandit_math_reasoning']
            return getattr(module, 'GLOBAL_BANDIT_ACTOR', None)
    except Exception as e:
        logger.debug(f"Failed to get global bandit actor: {e}")
    return None


class MathReasoningBanditEnv(BaseEnv):
    """
    Math reasoning environment with bandit-based prompt selection.

    This environment integrates with a centralized BanditActor for prompt
    selection, enabling multi-armed bandit learning across distributed
    environment workers.
    """

    def __init__(self, config: MathReasoningBanditConfig):
        """
        Initialize environment.

        Args:
            config: Environment configuration
        """
        super().__init__(config)
        self.config = config

        # These will be injected by pipeline or from global registry
        self.bandit_actor = config.bandit_actor
        if self.bandit_actor is None:
            # Try to get from global registry
            self.bandit_actor = get_global_bandit_actor()

        self.prompt_templates = config.prompt_templates
        self.problem_encoder = config.problem_encoder

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

        # Check if bandit components are available
        if self.bandit_actor is None:
            logger.warning(
                "BanditActor not provided. Environment will use random prompt selection."
            )
        if self.prompt_templates is None or len(self.prompt_templates) == 0:
            logger.warning(
                "No prompt templates provided. Using default template."
            )
        if self.problem_encoder is None:
            logger.warning(
                "Problem encoder not provided. Using random embeddings."
            )

        logger.info(
            f"Initialized MathReasoningBanditEnv with {len(self.dataset)} problems"
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
        if self.problem_encoder is not None:
            try:
                self.current_embedding = self.problem_encoder.encode(self.current_problem)
            except Exception as e:
                logger.warning(f"Problem encoding failed: {e}. Using random embedding.")
                self.current_embedding = np.random.randn(384).astype(np.float32)
        else:
            # Fallback: use hash-based pseudo-embedding
            hash_val = hash(self.current_problem)
            np.random.seed(abs(hash_val) % (2**32))
            self.current_embedding = np.random.randn(384).astype(np.float32)

        # 3. Select prompt using BanditActor
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

            except Exception as e:
                logger.warning(f"Bandit selection failed: {e}. Using random prompt.")
                self.current_prompt_idx = np.random.randint(0, len(self.prompt_templates))
                self.current_prompt_name = self.prompt_templates[self.current_prompt_idx].name
                self.current_prompt_info = {
                    "arm_idx": self.current_prompt_idx,
                    "ucb_value": 0.0,
                    "predicted_reward": 0.0,
                    "confidence": 0.0,
                }
        else:
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

        # 4. Prepare metrics (只包含数值类型)
        metrics = {
            "action_is_valid": 1.0,  # 转换为float
            "action_is_effective": 1.0,  # 转换为float
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
