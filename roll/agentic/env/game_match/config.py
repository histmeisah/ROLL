from dataclasses import dataclass, field
from typing import Optional, List

from roll.agentic.env.base import BaseEnvConfig


@dataclass
class GameMatchEnvConfig(BaseEnvConfig):
    """Configuration for the Shisen-Sho (Match Game) environment.

    The agent views an image of a grid with colored shape tiles and
    selects pairs of matching tiles to remove.  Two tiles match if they
    are the same type and can be connected by a path with at most 2 turns
    that does not pass through other tiles.
    """

    # Board
    grid_rows: int = 6
    grid_cols: int = 6
    num_colors: int = 3
    num_shapes: int = 3

    # Rendering
    render_mode: str = "rgb_array"
    image_resolution: int = 256

    # Environment
    max_steps: int = 50

    # Action parsing
    action_pattern: str = r"<answer>(.*?)</answer>"
    special_token_list: Optional[List[str]] = field(
        default_factory=lambda: [
            "<think>", "</think>", "<|im_start|>", "<|im_end|>",
        ]
    )

    # Instruction
    env_instruction: str = (
        "You are playing Shisen-Sho, a tile matching puzzle game. "
        "A 6x6 grid contains colored shape tiles. Each tile type appears 4 times. "
        "Match two identical tiles that can be connected by a path with at most 2 turns "
        "(the path uses only horizontal and vertical segments and cannot pass through "
        "other tiles, but can go around the board edges). "
        "When matched, both tiles are removed. Clear all tiles to win. "
        "Specify two tile positions using row and column coordinates (0-indexed). "
        "Format your answer as: <answer>(row1, col1) (row2, col2)</answer>"
    )
