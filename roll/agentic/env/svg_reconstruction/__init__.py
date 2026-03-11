"""SVG Reconstruction environment for VLM RL training.

The agent sees a target image and generates SVG code to reconstruct it,
with multi-turn refinement support.
"""

from .config import SVGReconstructionEnvConfig
from .env import SVGReconstructionEnv

__all__ = ["SVGReconstructionEnv", "SVGReconstructionEnvConfig"]
