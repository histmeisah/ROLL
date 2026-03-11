from dataclasses import dataclass, field
from typing import Optional, List

from roll.agentic.env.base import BaseEnvConfig


@dataclass
class SVGReconstructionEnvConfig(BaseEnvConfig):
    """Configuration for SVG Reconstruction environment.

    The agent sees a target image and must generate SVG code to reconstruct it.
    Supports multi-turn interaction: after each attempt the agent sees a
    side-by-side comparison and can refine its output.
    """

    # Dataset
    dataset_name: str = "starvector/svg-icons-simple"
    dataset_split: str = "train"
    dataset_cache_dir: Optional[str] = None

    # Rendering
    render_mode: str = "rgb_array"
    image_resolution: int = 256

    # Environment
    max_steps: int = 3

    # Scoring weights (CPU-based, no GPU model needed)
    structural_weight: float = 0.4
    pixel_mse_weight: float = 0.3
    ssim_weight: float = 0.3

    # SVG processing
    svg_timeout: float = 5.0  # seconds for SVG cleaning / rasterization

    # Action parsing
    action_pattern: str = r"<answer>(.*?)</answer>"
    special_token_list: Optional[List[str]] = field(
        default_factory=lambda: [
            "<think>", "</think>", "<|im_start|>", "<|im_end|>",
        ]
    )

    # Instruction
    env_instruction: str = (
        "You are an SVG reconstruction agent. You will be shown a target image. "
        "Your goal is to write SVG code that recreates the target image as closely as possible. "
        "Output your SVG code inside <answer>YOUR_SVG_CODE</answer> tags. "
        "The SVG must be a valid SVG document starting with <svg and ending with </svg>. "
        "After each attempt you will see a side-by-side comparison (target on the left, "
        "your result on the right) so you can refine your output."
    )
