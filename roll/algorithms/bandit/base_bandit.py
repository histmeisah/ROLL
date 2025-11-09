"""
Base class for contextual bandit algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch


class BaseContextualBandit(ABC):
    """
    Abstract base class for contextual bandit algorithms.

    Used for prompt selection in the outer loop of Bandit-REINFORCE++.
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        exploration_param: float = 1.0,
        device: str = "cuda",
        seed: int = 42,
    ):
        """
        Initialize contextual bandit.

        Args:
            n_arms: Number of arms (prompt templates)
            context_dim: Dimension of context features (problem embedding)
            exploration_param: Exploration parameter (alpha in UCB)
            device: Device for computation
            seed: Random seed
        """
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.exploration_param = exploration_param
        self.device = device
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        self.t = 0  # Time step counter
        self.arm_counts = np.zeros(n_arms)  # Number of times each arm is pulled
        self.rewards_history = [[] for _ in range(n_arms)]  # Reward history per arm

    @abstractmethod
    def select_arm(self, context: np.ndarray) -> int:
        """
        Select an arm based on the current context.

        Args:
            context: Context vector (problem embedding) of shape (context_dim,)

        Returns:
            Selected arm index
        """
        pass

    @abstractmethod
    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """
        Update the bandit with observed reward.

        Args:
            arm: The arm that was pulled
            context: The context when the arm was pulled
            reward: Observed reward
        """
        pass

    def reset(self) -> None:
        """Reset the bandit to initial state."""
        self.t = 0
        self.arm_counts = np.zeros(self.n_arms)
        self.rewards_history = [[] for _ in range(self.n_arms)]

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics of the bandit."""
        return {
            "total_pulls": self.t,
            "arm_counts": self.arm_counts.tolist(),
            "mean_rewards": [
                np.mean(rewards) if rewards else 0.0
                for rewards in self.rewards_history
            ],
            "exploration_param": self.exploration_param,
        }