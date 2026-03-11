from dataclasses import dataclass, field
from typing import Optional, List, Dict

from roll.agentic.env.base import BaseEnvConfig


@dataclass
class Game2048EnvConfig(BaseEnvConfig):
    """Configuration for the 2048 game environment.

    The agent views an image of the 2048 board and decides which direction
    to slide the tiles (Up / Right / Down / Left).
    """

    # Board
    board_size: int = 4

    # Rendering
    render_mode: str = "rgb_array"
    image_resolution: int = 256

    # Environment
    max_steps: int = 200

    # Action parsing
    action_pattern: str = r"<answer>(.*?)</answer>"
    action_lookup: Dict[int, str] = field(
        default_factory=lambda: {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
    )
    special_token_list: Optional[List[str]] = field(
        default_factory=lambda: [
            "<think>", "</think>", "<|im_start|>", "<|im_end|>",
        ]
    )

    # Instruction
    env_instruction: str = (
        "You are playing the 2048 game. Slide numbered tiles on a 4x4 grid. "
        "Tiles with the same number merge when they collide. After each move, "
        "a new tile (2 or 4) appears randomly. The goal is to create high-value tiles. "
        "Available actions: Up, Down, Left, Right. "
        "Format your answer as: <answer>Right</answer>"
    )
