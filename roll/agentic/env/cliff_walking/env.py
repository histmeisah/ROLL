import numpy as np
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv as GymCliffWalkingEnv

from roll.agentic.env.base import BaseEnv
from roll.agentic.env.parse_action_utils import default_parser_action_func
from .config import CliffWalkingEnvConfig
from roll.agentic.utils import all_seed


class CliffWalkingEnv(BaseEnv, GymCliffWalkingEnv):
    def __init__(self, config: CliffWalkingEnvConfig = CliffWalkingEnvConfig()):
        BaseEnv.__init__(self, config)
        self.config = config
        self.GRID_LOOKUP = config.grid_lookup
        self.ACTION_LOOKUP = config.action_lookup
        self.render_mode = config.render_mode
        GymCliffWalkingEnv.__init__(self, render_mode=config.render_mode)
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        with all_seed(seed):
            GymCliffWalkingEnv.reset(self, seed=seed)
            return self.render(), {}

    def step(self, action: str):
        action_info = self.parse_action(action)
        if action_info["action"] is None:
            metrics = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
            }
            info = {"metrics": metrics}
            info.update(action_info)
            self.step_count += 1
            return self.render(), 0, False, False, info

        prev_pos = int(self.s)
        _, reward, terminated, truncated, _ = GymCliffWalkingEnv.step(self, action_info["action"])
        self.step_count += 1
        next_obs = self.render()
        metrics = {
            "action_is_effective": prev_pos != int(self.s),
            "action_is_valid": True,
            "success": tuple(np.unravel_index(int(self.s), self.shape)) == (self.shape[0] - 1, self.shape[1] - 1),
        }
        info = {"metrics": metrics}
        info.update(action_info)
        if not terminated and self.step_count >= self.config.max_steps:
            truncated = True
        return next_obs, reward, terminated, truncated, info

    def parse_action(self, text):
        return default_parser_action_func(text, self.config.action_pattern, self.config.action_lookup, self.config.special_token_list)

    def render(self, mode=None):
        if not mode:
            mode = self.render_mode
        if mode == "text":
            nrow, ncol = self.shape
            grid = np.full(self.shape, 1, dtype=int)  # 1 = empty (_)
            # Mark cliff cells
            for r in range(nrow):
                for c in range(ncol):
                    if self._cliff[r, c]:
                        grid[r, c] = 2  # C = cliff
            # Mark start
            start_pos = np.unravel_index(self.start_state_index, self.shape)
            grid[start_pos] = 3  # S = start
            # Mark goal
            grid[nrow - 1, ncol - 1] = 4  # G = goal
            # Mark player position
            player_row, player_col = self.player_pos
            if self._cliff[player_row, player_col]:
                grid[player_row, player_col] = 5  # X = player on cliff
            elif (player_row, player_col) == (nrow - 1, ncol - 1):
                # player on goal - show as P (goal reached)
                grid[player_row, player_col] = 0  # P = player
            else:
                grid[player_row, player_col] = 0  # P = player
            return "\n".join("".join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in grid)
        elif mode == "rgb_array":
            return None
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def get_all_actions(self):
        return list(self.ACTION_LOOKUP.values())

    @property
    def player_pos(self):
        return tuple(np.unravel_index(int(self.s), self.shape))  # (row, col)

    def close(self):
        super(CliffWalkingEnv, self).close()


if __name__ == "__main__":
    config = CliffWalkingEnvConfig()
    env = CliffWalkingEnv(config)
    obs, _ = env.reset(seed=42)
    print("Initial state:")
    print(obs)
    print()
    while True:
        keyboard = input("Enter action (0=Up, 1=Right, 2=Down, 3=Left, q=quit): ")
        if keyboard == "q":
            break
        action_id = int(keyboard)
        assert action_id in env.ACTION_LOOKUP, f"Invalid action: {action_id}"
        action_text = f"<answer>{env.ACTION_LOOKUP[action_id]}</answer>"
        obs, reward, terminated, truncated, info = env.step(action_text)
        print(obs, f"reward={reward}", f"terminated={terminated}", f"truncated={truncated}")
        print()
        if terminated or truncated:
            break
