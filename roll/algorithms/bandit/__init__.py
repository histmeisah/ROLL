"""
Contextual Bandit algorithms for ROLL framework.

This module implements NeuralUCB and other contextual bandit algorithms
for prompt selection in LLM training, particularly for mathematical reasoning tasks.
"""

from .neural_ucb import NeuralUCB
from .base_bandit import BaseContextualBandit
from .bandit_reinforce_plus import (
    BanditReinforcePlusPlus,
    create_bandit_reinforce_from_preset,
    create_bandit_reinforce_from_names,
)
from .prompt_loader import (
    PromptTemplate,
    PromptLoader,
    load_prompts,
    load_preset,
    load_prompt_by_name,
)
from .prompt_monitor import PromptMonitor, PromptStats, get_wandb_metrics

__all__ = [
    "BaseContextualBandit",
    "NeuralUCB",
    "BanditReinforcePlusPlus",
    "create_bandit_reinforce_from_preset",
    "create_bandit_reinforce_from_names",
    "PromptTemplate",
    "PromptLoader",
    "load_prompts",
    "load_preset",
    "load_prompt_by_name",
    "PromptMonitor",
    "PromptStats",
    "get_wandb_metrics",
]