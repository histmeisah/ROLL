"""
CliffWalking Environment for ROLL Framework

## Description
A standard CliffWalking environment (4x12 grid) where the agent starts at
the bottom-left and must reach the goal at the bottom-right while avoiding
the cliff along the bottom edge.

## Action Space
- 0: Up
- 1: Right
- 2: Down
- 3: Left

## Grid Layout
The bottom row contains: Start (S), Cliff (C)...(C), Goal (G)
Stepping on the cliff returns the agent to the start with -100 reward.
Each normal step gives -1 reward. Reaching the goal terminates the episode.

## Example
_ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _
P C C C C C C C C C C G
"""

from .env import CliffWalkingEnv
from .config import CliffWalkingEnvConfig

__all__ = ["CliffWalkingEnv", "CliffWalkingEnvConfig"]
