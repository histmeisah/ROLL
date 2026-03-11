"""2048 game environment for VLM RL training."""

from .config import Game2048EnvConfig
from .env import Game2048Env

__all__ = ["Game2048Env", "Game2048EnvConfig"]
