from typing import Optional, List, Dict
from dataclasses import dataclass, field

from roll.agentic.env.base import BaseEnvConfig


@dataclass
class CliffWalkingEnvConfig(BaseEnvConfig):
    """Configuration for CliffWalking environment"""

    # Grid is fixed at 4 rows x 12 columns (standard CliffWalking)
    nrow: int = 4
    ncol: int = 12
    render_mode: str = "text"

    # Mappings
    # 0: Up, 1: Right, 2: Down, 3: Left (gymnasium standard)
    action_lookup: Dict[int, str] = field(default_factory=lambda: {0: "Up", 1: "Right", 2: "Down", 3: "Left"})
    grid_lookup: Dict[int, str] = field(
        default_factory=lambda: {0: "P", 1: "_", 2: "C", 3: "S", 4: "G", 5: "X"}
    )
    # P: player, _: empty, C: cliff, S: start, G: goal, X: player on cliff (fell)
    grid_vocab: Dict[str, str] = field(
        default_factory=lambda: {
            "P": "player",
            "_": "empty",
            "C": "cliff",
            "S": "start",
            "G": "goal",
            "X": "player fell off cliff",
        }
    )

    max_steps: int = 200
    env_instruction: str = (
        "You are navigating the CliffWalking grid (4 rows x 12 columns). "
        "You start at the bottom-left corner and must reach the goal at the bottom-right corner. "
        "The bottom edge between start and goal is a cliff. "
        "Stepping on the cliff gives -100 reward and sends you back to start. "
        "Each normal step gives -1 reward. Reaching the goal ends the episode. "
        "The answer must be one of action in a turn, format is <answer>Right</answer>"
    )
    action_pattern: str = r"<answer>(.*?)</answer>"
    special_token_list: Optional[List[str]] = field(
        default_factory=lambda: ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]
    )

    def __post_init__(self):
        grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join(
            [f"{k}: {v}" for k, v in self.grid_vocab.items()])
        action_lookup_str = "\nYour available actions are:\n" + ", ".join(
            [f"{v}" for k, v in self.action_lookup.items()])
        self.env_instruction = self.env_instruction + grid_vocab_str + action_lookup_str
