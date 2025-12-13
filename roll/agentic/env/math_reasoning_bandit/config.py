"""
Configuration for Math Reasoning Bandit environment.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from roll.agentic.env.base import BaseEnvConfig


# Default bandit actor name for Ray Named Actor pattern
DEFAULT_BANDIT_ACTOR_NAME = "bandit_actor_global"


@dataclass
class MathReasoningBanditConfig(BaseEnvConfig):
    """Configuration for Math Reasoning Bandit environment."""

    # Environment settings
    max_steps: int = 1  # Single-step task (question -> answer -> reward)

    # Dataset settings
    dataset_name: str = "gsm8k"  # Options: gsm8k, math, aime
    dataset_split: str = "train"
    dataset_seed: int = 42
    dataset_path: Optional[str] = None  # Path to local dataset file (JSON/JSONL)

    # Answer extraction settings
    answer_patterns: List[str] = field(default_factory=lambda: [
        r'\\boxed\{(.+?)\}',  # LaTeX boxed format
        r'<answer>(.+?)</answer>',  # XML tag format
        r'(?:答案是|Answer|answer)[:：]\s*(.+)',  # "Answer:" format
    ])

    # Reward settings
    reward_correct: float = 1.0
    reward_incorrect: float = 0.0
    reward_partial: float = 0.5  # For partial matches

    # Verification settings
    enable_numerical_matching: bool = True
    numerical_tolerance: float = 1e-3
    enable_partial_matching: bool = True

    # Instruction template
    env_instruction: str = "Solve the following mathematical problem step by step and provide the final answer."

    # Bandit configuration - for loading prompts and encoder in distributed workers
    bandit_actor_name: str = DEFAULT_BANDIT_ACTOR_NAME  # Ray Named Actor name
    prompt_config_path: Optional[str] = None  # Path to prompt YAML config
    preset_name: str = "diverse_5"  # Prompt preset name
    encoder_model: Optional[str] = None  # SentenceTransformer model name/path
    context_dim: int = 768  # Embedding dimension

    # Bandit-related (will be set by pipeline or loaded dynamically)
    bandit_actor: Optional[object] = None  # Ray actor handle (not serializable)
    prompt_templates: Optional[List] = None  # PromptTemplate list (not serializable)
    problem_encoder: Optional[object] = None  # SentenceTransformer (not serializable)
