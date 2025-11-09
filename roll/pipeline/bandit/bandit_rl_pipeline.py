"""
Bandit-RL Pipeline for ROLL framework.

Integrates Bandit-REINFORCE++ with ROLL's training infrastructure
for mathematical reasoning tasks.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from roll.pipeline.base_pipeline import BasePipeline
from roll.algorithms.bandit import BanditReinforcePlusPlus, DEFAULT_PROMPT_TEMPLATES
from roll.algorithms.bandit.bandit_reinforce_plus import PromptTemplate
from roll.utils.logging import get_logger

logger = get_logger()


@dataclass
class BanditRLConfig:
    """Configuration for Bandit-RL pipeline."""

    # Model configuration
    model_name: str = "Qwen2.5-3B-Instruct"
    model_path: str = ""

    # Bandit configuration
    n_prompts: int = 5
    context_dim: int = 768  # Dimension of problem embeddings
    hidden_dims: List[int] = None  # Default [256, 128]
    exploration_param: float = 1.0

    # Training configuration
    num_trajectories: int = 8  # K in the paper
    num_episodes: int = 10000
    batch_size: int = 32
    learning_rate: float = 1e-4

    # NeuralUCB specific
    neural_ucb_buffer_size: int = 10000
    neural_ucb_update_freq: int = 10
    neural_ucb_reg_param: float = 1.0

    # Logging
    log_interval: int = 10
    checkpoint_interval: int = 100

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class BanditRLPipeline(BasePipeline):
    """
    Pipeline for Bandit-REINFORCE++ training on mathematical reasoning tasks.
    """

    def __init__(self, config: BanditRLConfig):
        super().__init__()
        self.config = config

        # Initialize prompt templates
        self.prompt_templates = self._init_prompt_templates()

        # Initialize Bandit-REINFORCE++
        neural_ucb_kwargs = {
            "buffer_size": config.neural_ucb_buffer_size,
            "update_freq": config.neural_ucb_update_freq,
            "reg_param": config.neural_ucb_reg_param,
            "learning_rate": config.learning_rate,
        }

        self.bandit_rl = BanditReinforcePlusPlus(
            prompt_templates=self.prompt_templates,
            context_dim=config.context_dim,
            hidden_dims=config.hidden_dims,
            exploration_param=config.exploration_param,
            neural_ucb_kwargs=neural_ucb_kwargs,
        )

        # Initialize problem encoder (for getting embeddings)
        self.problem_encoder = self._init_problem_encoder()

        # Statistics tracking
        self.episode_rewards = []
        self.prompt_selection_history = []

        logger.info(f"Initialized BanditRLPipeline with config: {config}")

    def _init_prompt_templates(self) -> List[PromptTemplate]:
        """Initialize prompt templates."""
        # Use default templates or customize based on config
        templates = DEFAULT_PROMPT_TEMPLATES[:self.config.n_prompts]

        # Could also load custom templates from config
        # if hasattr(self.config, 'custom_prompts'):
        #     templates = [PromptTemplate(**p) for p in self.config.custom_prompts]

        logger.info(f"Initialized {len(templates)} prompt templates")
        return templates

    def _init_problem_encoder(self):
        """
        Initialize problem encoder for getting context embeddings.

        This could be a pre-trained sentence encoder like BERT/Sentence-BERT.
        For now, using a simple placeholder.
        """
        # Placeholder - in practice, use a proper encoder
        class SimpleEncoder:
            def __init__(self, output_dim):
                self.output_dim = output_dim
                self.rng = np.random.RandomState(42)

            def encode(self, text):
                # Simple hash-based pseudo-embedding
                # In practice, use sentence-transformers or similar
                hash_val = hash(text)
                self.rng.seed(abs(hash_val) % (2**32))
                return self.rng.randn(self.output_dim).astype(np.float32)

        return SimpleEncoder(self.config.context_dim)

    def encode_problem(self, problem: str) -> np.ndarray:
        """
        Encode a mathematical problem into a context vector.

        Args:
            problem: Problem text

        Returns:
            Context embedding of shape (context_dim,)
        """
        return self.problem_encoder.encode(problem)

    def compute_reward(self, problem: str, solution: str, ground_truth: Optional[str] = None) -> float:
        """
        Compute reward for a solution.

        Args:
            problem: Problem text
            solution: Generated solution
            ground_truth: Optional ground truth answer

        Returns:
            Reward value (e.g., 1.0 for correct, 0.0 for incorrect)
        """
        # Placeholder - implement actual mathematical verification
        # Could use:
        # 1. Exact match with ground truth
        # 2. Symbolic math verification
        # 3. Numerical answer extraction and comparison

        if ground_truth:
            # Simple string matching for now
            # In practice, extract numerical answer and compare
            return 1.0 if ground_truth.lower() in solution.lower() else 0.0
        else:
            # Random reward for testing
            return np.random.binomial(1, 0.3)

    def train_episode(
        self,
        problem: str,
        ground_truth: Optional[str] = None,
        policy_model: Optional[Any] = None
    ) -> Dict:
        """
        Run one training episode.

        Args:
            problem: Mathematical problem text
            ground_truth: Optional ground truth answer
            policy_model: The LLM policy model

        Returns:
            Episode statistics
        """
        # Encode problem to get context
        problem_embedding = self.encode_problem(problem)

        # Run Bandit-REINFORCE++ episode
        episode_stats = self.bandit_rl.run_episode(
            problem=problem,
            problem_embedding=problem_embedding,
            policy_model=policy_model,
            num_trajectories=self.config.num_trajectories,
            compute_reward_fn=lambda p, s: self.compute_reward(p, s, ground_truth)
        )

        # Track statistics
        self.episode_rewards.append(episode_stats["mean_reward"])
        self.prompt_selection_history.append(episode_stats["prompt_idx"])

        return episode_stats

    def train(
        self,
        train_problems: List[Dict[str, str]],
        val_problems: Optional[List[Dict[str, str]]] = None,
        policy_model: Optional[Any] = None
    ):
        """
        Main training loop.

        Args:
            train_problems: List of training problems with 'problem' and 'answer' keys
            val_problems: Optional validation problems
            policy_model: The LLM policy model to train
        """
        logger.info(f"Starting training with {len(train_problems)} problems")

        for episode_idx in range(self.config.num_episodes):
            # Sample a problem
            problem_data = np.random.choice(train_problems)
            problem = problem_data["problem"]
            ground_truth = problem_data.get("answer", None)

            # Run training episode
            episode_stats = self.train_episode(
                problem=problem,
                ground_truth=ground_truth,
                policy_model=policy_model
            )

            # Logging
            if episode_idx % self.config.log_interval == 0:
                self._log_statistics(episode_idx, episode_stats)

            # Checkpointing
            if episode_idx % self.config.checkpoint_interval == 0:
                self._save_checkpoint(episode_idx)

            # Validation
            if val_problems and episode_idx % 100 == 0:
                val_stats = self.validate(val_problems, policy_model)
                logger.info(f"Validation at episode {episode_idx}: {val_stats}")

    def validate(
        self,
        val_problems: List[Dict[str, str]],
        policy_model: Optional[Any] = None
    ) -> Dict:
        """
        Run validation on a set of problems.

        Args:
            val_problems: Validation problems
            policy_model: The policy model

        Returns:
            Validation statistics
        """
        total_reward = 0.0
        prompt_counts = {i: 0 for i in range(self.config.n_prompts)}

        for problem_data in val_problems:
            problem = problem_data["problem"]
            ground_truth = problem_data.get("answer", None)
            problem_embedding = self.encode_problem(problem)

            # Select prompt (no exploration during validation)
            with torch.no_grad():
                prompt_idx, _ = self.bandit_rl.select_prompt(problem_embedding)

            prompt_counts[prompt_idx] += 1

            # Generate solution and compute reward
            # (simplified for now)
            reward = self.compute_reward(problem, "solution", ground_truth)
            total_reward += reward

        val_stats = {
            "mean_reward": total_reward / len(val_problems),
            "prompt_distribution": prompt_counts,
        }

        return val_stats

    def _log_statistics(self, episode_idx: int, episode_stats: Dict):
        """Log training statistics."""
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards

        logger.info(
            f"Episode {episode_idx}: "
            f"Reward={episode_stats['mean_reward']:.3f}, "
            f"Best={episode_stats['best_reward']:.3f}, "
            f"Prompt={episode_stats['prompt_name']}, "
            f"Avg(100)={np.mean(recent_rewards):.3f}"
        )

        # Log prompt usage distribution
        if episode_idx % (self.config.log_interval * 10) == 0:
            stats = self.bandit_rl.get_statistics()
            logger.info(f"Bandit statistics: {stats}")

    def _save_checkpoint(self, episode_idx: int):
        """Save checkpoint."""
        checkpoint = {
            "episode": episode_idx,
            "config": self.config,
            "bandit_state": self.bandit_rl.get_statistics(),
            "episode_rewards": self.episode_rewards,
            "prompt_history": self.prompt_selection_history,
        }

        # Save checkpoint (implement actual saving logic)
        logger.info(f"Checkpoint saved at episode {episode_idx}")

        return checkpoint