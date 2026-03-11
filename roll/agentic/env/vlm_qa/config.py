from dataclasses import dataclass, field
from typing import List, Optional

from roll.agentic.env.base import BaseEnvConfig


@dataclass
class VLMQAEnvConfig(BaseEnvConfig):
    """Configuration for the universal VLM QA environment.

    Supports GeoQA, MathVista, TabMWP and similar visual question-answering
    datasets through column mapping configuration.
    """

    # Dataset
    dataset_name: str = ""
    dataset_split: str = "train"
    dataset_subset: str = ""
    dataset_cache_dir: Optional[str] = None

    # Column mapping (adapt to different datasets)
    image_column: str = "image"
    question_column: str = "question"
    answer_column: str = "answer"
    answer_type_column: str = ""       # optional, e.g. "question_type" for MathVista
    choices_column: str = ""           # optional, e.g. "choices" for MC datasets

    # Table rendering (for TabMWP which has no image)
    table_column: str = ""             # non-empty -> render table text as image
    table_title_column: str = ""

    # Answer pre-processing
    answer_extract_pattern: str = ""   # e.g. "<answer>(.*?)</answer>" for GeoQA

    # Rendering / scoring
    render_mode: str = "rgb_array"
    image_resolution: int = 256
    answer_tolerance: float = 0.01
    correct_reward: float = 2.0
    wrong_reward: float = 0.5
    invalid_reward: float = 0.0

    # Action parsing
    action_pattern: str = r"<answer>(.*?)</answer>"
    special_token_list: Optional[List[str]] = field(
        default_factory=lambda: [
            "<think>", "</think>", "<|im_start|>", "<|im_end|>",
        ]
    )
    max_steps: int = 3
    max_tokens_per_step: int = 512

    # Instruction (will be dynamically updated in reset)
    env_instruction: str = (
        "You are a visual question-answering agent. You will be shown an image "
        "and a question about it. Analyze the image carefully and provide your "
        "answer inside <answer>YOUR_ANSWER</answer> tags.\n"
        "For multiple-choice questions, answer with the letter (e.g. A, B, C, D).\n"
        "For numerical questions, answer with just the number.\n"
        "For free-form questions, answer concisely."
    )
