from dataclasses import dataclass
from typing import Optional

from roll.agentic.env.base import BaseEnvConfig


@dataclass
class MathEnvConfig(BaseEnvConfig):
    """Configuration for Math problem solving environment."""

    # Dataset config
    dataset_name: str = "gsm8k"         # aime / math / math_all / gsm8k
    dataset_split: str = "train"
    cache_dir: Optional[str] = None
    dataset_path: Optional[str] = None  # Local parquet file path (overrides HuggingFace loading)
    question_key: str = "problem"
    answer_key: str = "answer"

    # Environment parameters (max_steps, action_pattern inherited from BaseEnvConfig)
    max_steps: int = 3                   # 3 attempts for off-policy training
    correct_reward: float = 1.0
    action_pattern: str = r"<answer>(.*?)</answer>"

    # Answer verification
    use_math_verify: bool = True         # Use math-verify library for robust checking
    verify_timeout: float = 5.0          # Timeout for answer verification (seconds)
