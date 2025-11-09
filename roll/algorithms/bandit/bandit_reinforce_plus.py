"""
Bandit-REINFORCE++: Dual-layer optimization framework combining
contextual bandits for prompt selection with REINFORCE++ for LLM training.

This implementation is designed for mathematical reasoning tasks where:
- Outer loop: NeuralUCB selects optimal prompts
- Inner loop: REINFORCE++ trains the LLM with selected prompts
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .neural_ucb import NeuralUCB
from roll.utils.logging import get_logger

logger = get_logger()


@dataclass
class PromptTemplate:
    """Represents a prompt template for mathematical reasoning."""

    name: str
    template: str
    description: str

    def format(self, problem: str) -> str:
        """Format the prompt template with the given problem."""
        return self.template.format(problem=problem)


class BanditReinforcePlusPlus:
    """
    Bandit-REINFORCE++ algorithm for LLM mathematical reasoning.

    Combines contextual bandit (NeuralUCB) for prompt selection with
    REINFORCE++ for policy optimization.
    """

    def __init__(
        self,
        prompt_templates: List[PromptTemplate],
        context_dim: int = 768,  # Dimension of problem embeddings
        hidden_dims: List[int] = [256, 128],
        exploration_param: float = 1.0,
        neural_ucb_kwargs: Optional[Dict] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
    ):
        """
        Initialize Bandit-REINFORCE++.

        Args:
            prompt_templates: List of prompt templates to choose from
            context_dim: Dimension of problem embeddings (context)
            hidden_dims: Hidden dimensions for NeuralUCB network
            exploration_param: Exploration parameter (alpha)
            neural_ucb_kwargs: Additional kwargs for NeuralUCB
            device: Device for computation
            seed: Random seed
        """
        self.prompt_templates = prompt_templates
        self.n_prompts = len(prompt_templates)
        self.context_dim = context_dim
        self.device = device

        # Initialize NeuralUCB for prompt selection
        ucb_kwargs = neural_ucb_kwargs or {}
        self.bandit = NeuralUCB(
            n_arms=self.n_prompts,
            context_dim=context_dim,
            hidden_dims=hidden_dims,
            exploration_param=exploration_param,
            device=device,
            seed=seed,
            **ucb_kwargs
        )

        # Statistics tracking
        self.episode_count = 0
        self.total_reward = 0.0
        self.prompt_usage_stats = {i: {"count": 0, "rewards": []} for i in range(self.n_prompts)}

        logger.info(
            f"Initialized Bandit-REINFORCE++ with {self.n_prompts} prompt templates, "
            f"context_dim={context_dim}, exploration_param={exploration_param}"
        )

    def select_prompt(self, problem_embedding: np.ndarray) -> Tuple[int, PromptTemplate]:
        """
        Select a prompt template using NeuralUCB.

        Args:
            problem_embedding: Embedding of the problem (context), shape (context_dim,)

        Returns:
            Tuple of (prompt_index, prompt_template)
        """
        # Ensure embedding has correct shape
        if problem_embedding.shape != (self.context_dim,):
            raise ValueError(
                f"Expected problem_embedding shape ({self.context_dim},), "
                f"got {problem_embedding.shape}"
            )

        # Select prompt using bandit
        prompt_idx = self.bandit.select_arm(problem_embedding)
        prompt_template = self.prompt_templates[prompt_idx]

        logger.debug(f"Selected prompt {prompt_idx}: {prompt_template.name}")

        return prompt_idx, prompt_template

    def update_bandit(
        self,
        prompt_idx: int,
        problem_embedding: np.ndarray,
        reward: float
    ) -> None:
        """
        Update the bandit with observed reward.

        Args:
            prompt_idx: Index of the prompt that was used
            problem_embedding: The problem embedding (context)
            reward: Observed reward (e.g., correctness of solution)
        """
        self.bandit.update(prompt_idx, problem_embedding, reward)

        # Update statistics
        self.prompt_usage_stats[prompt_idx]["count"] += 1
        self.prompt_usage_stats[prompt_idx]["rewards"].append(reward)
        self.total_reward += reward

        logger.debug(
            f"Updated bandit for prompt {prompt_idx} with reward {reward:.3f}"
        )

    def run_episode(
        self,
        problem: str,
        problem_embedding: np.ndarray,
        policy_model: Any,  # The LLM policy model
        num_trajectories: int = 8,
        compute_reward_fn: callable = None,
    ) -> Dict:
        """
        Run one episode of Bandit-REINFORCE++.

        1. Select prompt using NeuralUCB
        2. Generate trajectories with selected prompt
        3. Compute rewards
        4. Update policy with REINFORCE++
        5. Update bandit with observed reward

        Args:
            problem: The mathematical problem text
            problem_embedding: Embedding of the problem
            policy_model: The LLM policy model to train
            num_trajectories: Number of trajectories to sample
            compute_reward_fn: Function to compute reward for a solution

        Returns:
            Dictionary containing episode statistics
        """
        self.episode_count += 1

        # Step 1: Select prompt using NeuralUCB
        prompt_idx, prompt_template = self.select_prompt(problem_embedding)
        formatted_prompt = prompt_template.format(problem)

        # Step 2: Generate trajectories with selected prompt
        # This would interface with ROLL's generation pipeline
        trajectories = self._generate_trajectories(
            policy_model,
            formatted_prompt,
            num_trajectories
        )

        # Step 3: Compute rewards for each trajectory
        rewards = []
        for traj in trajectories:
            if compute_reward_fn:
                reward = compute_reward_fn(problem, traj)
            else:
                # Default: binary reward based on correctness
                reward = self._default_reward(problem, traj)
            rewards.append(reward)

        # Step 4: Update policy with REINFORCE++
        # Select trajectory with highest reward for policy update
        best_idx = np.argmax(rewards)
        best_trajectory = trajectories[best_idx]
        best_reward = rewards[best_idx]

        # This would interface with ROLL's training pipeline
        train_metrics = self._update_policy(
            policy_model,
            formatted_prompt,
            best_trajectory,
            best_reward
        )

        # Step 5: Update bandit with mean reward
        mean_reward = np.mean(rewards)
        self.update_bandit(prompt_idx, problem_embedding, mean_reward)

        # Prepare episode statistics
        episode_stats = {
            "episode": self.episode_count,
            "prompt_idx": prompt_idx,
            "prompt_name": prompt_template.name,
            "num_trajectories": num_trajectories,
            "rewards": rewards,
            "mean_reward": mean_reward,
            "best_reward": best_reward,
            "total_reward": self.total_reward,
            **train_metrics
        }

        return episode_stats

    def _generate_trajectories(
        self,
        policy_model: Any,
        prompt: str,
        num_trajectories: int
    ) -> List[str]:
        """
        Generate trajectories using the policy model.

        This should interface with ROLL's generation pipeline.
        """
        # Placeholder - actual implementation would use ROLL's generation
        trajectories = []
        for _ in range(num_trajectories):
            # Generate solution using policy_model
            # trajectory = policy_model.generate(prompt, ...)
            trajectory = f"Solution for: {prompt[:50]}..."  # Placeholder
            trajectories.append(trajectory)

        return trajectories

    def _update_policy(
        self,
        policy_model: Any,
        prompt: str,
        trajectory: str,
        reward: float
    ) -> Dict:
        """
        Update policy using REINFORCE++.

        This should interface with ROLL's training pipeline.
        """
        # Placeholder - actual implementation would use ROLL's training
        train_metrics = {
            "policy_loss": 0.0,  # Placeholder
            "gradient_norm": 0.0,  # Placeholder
        }

        return train_metrics

    def _default_reward(self, problem: str, solution: str) -> float:
        """
        Default reward function (binary correctness).

        In practice, this would check if the solution is correct.
        """
        # Placeholder - actual implementation would verify mathematical correctness
        return np.random.binomial(1, 0.5)  # Random binary reward for now

    def get_statistics(self) -> Dict:
        """Get current statistics of the algorithm."""
        stats = {
            "episode_count": self.episode_count,
            "total_reward": self.total_reward,
            "mean_reward": self.total_reward / max(self.episode_count, 1),
            "prompt_usage": {
                self.prompt_templates[i].name: {
                    "count": self.prompt_usage_stats[i]["count"],
                    "mean_reward": np.mean(self.prompt_usage_stats[i]["rewards"])
                    if self.prompt_usage_stats[i]["rewards"] else 0.0
                }
                for i in range(self.n_prompts)
            },
            "bandit_stats": self.bandit.get_statistics(),
        }

        return stats

    def reset(self) -> None:
        """Reset the algorithm to initial state."""
        self.bandit.reset()
        self.episode_count = 0
        self.total_reward = 0.0
        self.prompt_usage_stats = {i: {"count": 0, "rewards": []} for i in range(self.n_prompts)}

        logger.info("Reset Bandit-REINFORCE++ to initial state")


# Predefined prompt templates for mathematical reasoning
DEFAULT_PROMPT_TEMPLATES = [
    PromptTemplate(
        name="direct",
        template="Solve the following problem directly:\n{problem}\nSolution:",
        description="Direct problem solving without explicit reasoning steps"
    ),
    PromptTemplate(
        name="step_by_step",
        template="Solve step by step:\n{problem}\nLet's think step by step.\nSolution:",
        description="Step-by-step reasoning approach"
    ),
    PromptTemplate(
        name="detailed",
        template="Provide a detailed solution with explanations:\n{problem}\nDetailed solution with reasoning:",
        description="Detailed solution with comprehensive explanations"
    ),
    PromptTemplate(
        name="verify",
        template="Solve and verify your answer:\n{problem}\nSolution (show work and verification):",
        description="Solution with verification step"
    ),
    PromptTemplate(
        name="formal",
        template="Provide a formal mathematical solution:\n{problem}\nFormal solution:",
        description="Formal mathematical approach"
    ),
]