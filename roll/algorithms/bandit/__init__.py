"""
Contextual Bandit algorithms for ROLL framework.

This module implements NeuralUCB and other contextual bandit algorithms
for prompt selection in LLM training, particularly for mathematical reasoning tasks.
"""

from .neural_ucb import NeuralUCB
from .base_bandit import BaseContextualBandit
from .bandit_reinforce_plus import BanditReinforcePlusPlus

__all__ = [
    "BaseContextualBandit",
    "NeuralUCB",
    "BanditReinforcePlusPlus",
]