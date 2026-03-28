from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MathEnvConfig:
    """Configuration for Math problem solving environment."""

    # Dataset config
    dataset_name: str = "gsm8k"
    dataset_split: str = "train"
    cache_dir: Optional[str] = None
    dataset_path: Optional[str] = None
    question_key: str = "problem"
    answer_key: str = "answer"

    # Environment parameters
    max_steps: int = 3
    correct_reward: float = 1.0
    action_pattern: str = r"<answer>(.*?)</answer>"

    # Answer verification
    use_math_verify: bool = True
    verify_timeout: float = 5.0


@dataclass
class RollMathEnvConfig:
    """Configuration for gem-based MathEnv (roll_math)."""

    dataset_name: str = ""
    split: str = "train"
    question_key: str = "prompt"
    answer_key: str = "solution"
    max_steps: int = 1


@dataclass
class RollMathBanditEnvConfig:
    """Configuration for gem-based MathBanditEnv (roll_math_bandit)."""

    dataset_name: str = ""
    split: str = "train"
    question_key: str = "prompt"
    answer_key: str = "solution"
    bandit_config: Dict[str, Any] = field(default_factory=dict)
    max_steps: int = 1
