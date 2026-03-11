"""2048 game environment for ROLL.

The agent views an image of the 4x4 board and chooses a slide direction.
Designed for VLM training via ``VLTrajEnvManager``.

Ported from G1/VLM-Gym with PIL rendering (no pygame dependency).
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from roll.agentic.env.base import BaseEnv
from roll.agentic.env.parse_action_utils import default_parser_action_func
from roll.agentic.utils import all_seed

from .config import Game2048EnvConfig
from .game_logic import Game2048Logic, render_board

logger = logging.getLogger(__name__)


class Game2048Env(BaseEnv):
    """2048 game environment.

    Observations are numpy uint8 ``(H, W, 3)`` arrays showing the board.
    Actions are discrete directions: Up / Right / Down / Left.
    """

    def __init__(self, config: Game2048EnvConfig = Game2048EnvConfig()):
        super().__init__(config)
        self.config: Game2048EnvConfig = config
        self.render_mode: str = config.render_mode

        self._game = Game2048Logic(size=config.board_size)
        self._rng = random.Random()
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, dict]:
        self._step_count = 0
        with all_seed(seed):
            self._rng = random.Random(seed)
            self._game.reset(self._rng)
        return self.render(), {}

    def step(self, action: str) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step_count += 1
        action_info = self.parse_action(action)
        act = action_info["action"]

        if act is None:
            metrics = {
                "action_is_valid": False,
                "action_is_effective": False,
                "success": False,
                "score": self._game.score,
                "max_tile": int(self._game.board.max()),
            }
            info: Dict[str, Any] = {"metrics": metrics}
            info.update(action_info)
            truncated = self._step_count >= self.config.max_steps
            return self.render(), -1.0, False, truncated, info

        old_board = self._game.board.copy()
        _, reward, done = self._game.step(act, self._rng)
        effective = not np.array_equal(old_board, self._game.board)

        max_tile = int(self._game.board.max())
        metrics = {
            "action_is_valid": True,
            "action_is_effective": effective,
            "success": max_tile >= 2048,
            "score": self._game.score,
            "max_tile": max_tile,
        }
        info = {"metrics": metrics}
        info.update(action_info)

        terminated = done
        truncated = not done and self._step_count >= self.config.max_steps

        return self.render(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    def parse_action(self, text: str) -> Dict[str, Any]:
        return default_parser_action_func(
            text,
            self.config.action_pattern,
            self.config.action_lookup,
            self.config.special_token_list,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, mode: Optional[str] = None) -> np.ndarray:
        if mode is None:
            mode = self.render_mode
        if mode == "rgb_array":
            return render_board(self._game.board, self.config.image_resolution)
        raise ValueError(f"Unsupported render mode: {mode}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_all_actions(self) -> List[str]:
        return list(self.config.action_lookup.values())

    def close(self):
        pass


if __name__ == "__main__":
    config = Game2048EnvConfig(image_resolution=256, max_steps=20)
    env = Game2048Env(config)

    print("=== Game2048 Environment Test ===\n")

    obs, info = env.reset(seed=42)
    print(f"Reset: shape={obs.shape}, dtype={obs.dtype}")
    print(f"Board:\n{env._game.board}\n")

    for i in range(5):
        obs, reward, terminated, truncated, info = env.step("<answer>Right</answer>")
        print(f"Step {i+1}: reward={reward}, terminated={terminated}, "
              f"truncated={truncated}, max_tile={info['metrics']['max_tile']}")

    env.close()
    print("\nTest complete.")
