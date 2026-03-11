"""Universal VLM QA environment for visual question-answering RL training.

Supports GeoQA, MathVista, TabMWP and similar VQA datasets through
configurable column mappings.
"""

from .config import VLMQAEnvConfig
from .env import VLMQAEnv

__all__ = ["VLMQAEnv", "VLMQAEnvConfig"]
