"""Shisen-Sho (tile matching) game environment for ROLL.

The agent views an image of the tile grid and selects pairs of matching
tiles to remove.  Designed for VLM training via ``VLTrajEnvManager``.

Ported from G1/VLM-Gym with PIL rendering (no pygame dependency).
"""

import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from roll.agentic.env.base import BaseEnv
from roll.agentic.utils import all_seed

from .config import GameMatchEnvConfig
from .game_logic import GameMatchLogic, render_board

logger = logging.getLogger(__name__)

# Pattern: (row1, col1) (row2, col2) — tolerant of whitespace
_COORD_PATTERN = re.compile(
    r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
)


class GameMatchEnv(BaseEnv):
    """Shisen-Sho tile matching environment.

    Observations are numpy uint8 ``(H, W, 3)`` arrays showing the board.
    Actions are coordinate pairs selecting two tiles to match.
    """

    def __init__(self, config: GameMatchEnvConfig = GameMatchEnvConfig()):
        super().__init__(config)
        self.config: GameMatchEnvConfig = config
        self.render_mode: str = config.render_mode

        self._game = GameMatchLogic(
            rows=config.grid_rows,
            cols=config.grid_cols,
            num_colors=config.num_colors,
            num_shapes=config.num_shapes,
        )
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
        coords = action_info.get("action")

        if coords is None:
            metrics = {
                "action_is_valid": False,
                "action_is_effective": False,
                "success": False,
                "score": self._game.score,
                "matches_made": self._game.matches_made,
                "tiles_remaining": self._game.tiles_remaining,
            }
            info: Dict[str, Any] = {"metrics": metrics}
            info.update(action_info)
            truncated = self._step_count >= self.config.max_steps
            return self.render(), 0.0, False, truncated, info

        r1, c1, r2, c2 = coords
        old_board = self._game.board.copy()
        _, reward, done = self._game.step(r1, c1, r2, c2)
        effective = not np.array_equal(old_board, self._game.board)

        metrics = {
            "action_is_valid": True,
            "action_is_effective": effective,
            "success": self._game.tiles_remaining == 0,
            "score": self._game.score,
            "matches_made": self._game.matches_made,
            "tiles_remaining": self._game.tiles_remaining,
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
        """Parse coordinate-pair action from ``<answer>`` tags.

        Expected format inside tags: ``(row1, col1) (row2, col2)``
        """
        # Extract content from <answer> tags
        match = re.search(self.config.action_pattern, text, re.DOTALL)
        if not match:
            return {"action": None, "action_content": "", "think_content": ""}

        content = match.group(1).strip()

        # Strip special tokens
        if self.config.special_token_list:
            for token in self.config.special_token_list:
                content = content.replace(token, "").strip()

        # Parse coordinate pairs
        coord_match = _COORD_PATTERN.search(content)
        if not coord_match:
            return {"action": None, "action_content": content, "think_content": ""}

        try:
            r1 = int(coord_match.group(1))
            c1 = int(coord_match.group(2))
            r2 = int(coord_match.group(3))
            c2 = int(coord_match.group(4))
        except (ValueError, IndexError):
            return {"action": None, "action_content": content, "think_content": ""}

        return {
            "action": (r1, c1, r2, c2),
            "action_content": content,
            "think_content": "",
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, mode: Optional[str] = None) -> np.ndarray:
        if mode is None:
            mode = self.render_mode
        if mode == "rgb_array":
            return render_board(
                self._game.board,
                self.config.num_colors,
                self.config.num_shapes,
                self.config.image_resolution,
            )
        raise ValueError(f"Unsupported render mode: {mode}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_all_actions(self) -> List[str]:
        return []

    def close(self):
        pass


if __name__ == "__main__":
    config = GameMatchEnvConfig(image_resolution=256, max_steps=20)
    env = GameMatchEnv(config)

    print("=== GameMatch (Shisen-Sho) Environment Test ===\n")

    obs, info = env.reset(seed=42)
    print(f"Reset: shape={obs.shape}, dtype={obs.dtype}")
    print(f"Board:\n{env._game.board}\n")
    print(f"Tiles remaining: {env._game.tiles_remaining}")

    # Find a valid match for testing
    game = env._game
    found = False
    ext = game._extended_board()
    from .game_logic import can_connect
    for r1 in range(game.rows):
        for c1 in range(game.cols):
            if game.board[r1, c1] == 0:
                continue
            for r2 in range(game.rows):
                for c2 in range(game.cols):
                    if (r1, c1) >= (r2, c2):
                        continue
                    if game.board[r1, c1] == game.board[r2, c2]:
                        if can_connect(ext, r1 + 1, c1 + 1, r2 + 1, c2 + 1):
                            action_str = f"<answer>({r1}, {c1}) ({r2}, {c2})</answer>"
                            obs, reward, term, trunc, info = env.step(action_str)
                            print(f"Valid match ({r1},{c1})->({r2},{c2}): "
                                  f"reward={reward}, tiles_left="
                                  f"{info['metrics']['tiles_remaining']}")
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if found:
            break

    # Test invalid action
    obs, reward, term, trunc, info = env.step("<answer>invalid</answer>")
    print(f"Invalid action: reward={reward}, valid={info['metrics']['action_is_valid']}")

    env.close()
    print("\nTest complete.")
