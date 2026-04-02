"""
MathBanditEnv: Math environment with bandit-based prompt selection.

Extends MathEnv to use NeuralLinearUCB contextual bandit for adaptive
prompt selection. The bandit selects optimal reasoning prompts based on
problem embeddings, and updates its model using reward feedback.

Supports multimodal observations: when the observation is a Dict
(e.g. {"prompt": str, "image": PIL.Image}), it is forwarded to
EncoderActor as-is for unified multimodal encoding.
"""

import logging
import pickle
from typing import Any, Dict, Optional, SupportsFloat, Tuple, Union

import numpy as np
import ray

from roll.pipeline.agentic.env.gem.math_env import MathEnv
from roll.utils.constants import RAY_NAMESPACE

logger = logging.getLogger(__name__)

DEFAULT_BANDIT_ACTOR_NAME = "bandit_actor_global"
DEFAULT_ENCODER_ACTOR_NAME = "encoder_actor_global"


def _obs_to_encoder_input(obs: Any) -> Union[str, Dict[str, Any]]:
    """
    Convert environment observation to EncoderActor input format.

    Handles:
    - str: Pure text problem → {"text": obs}
    - np.ndarray: Rendered image → {"image": obs}
    - dict: Multimodal obs (prompt/image/video keys) → mapped to encoder format
    - other: str() fallback
    """
    if isinstance(obs, str):
        return {"text": obs}
    if isinstance(obs, np.ndarray):
        return {"image": obs}
    if isinstance(obs, dict):
        encoder_input = {}
        # Map common env observation keys to encoder keys
        if "prompt" in obs:
            encoder_input["text"] = obs["prompt"]
        if "text" in obs:
            encoder_input["text"] = obs["text"]
        if "image" in obs:
            encoder_input["image"] = obs["image"]
        if "video" in obs:
            encoder_input["video"] = obs["video"]
        return encoder_input if encoder_input else {"text": str(obs)}
    return {"text": str(obs)}


class MathBanditEnv(MathEnv):
    """
    Math environment with bandit-based prompt selection.

    Uses a centralized BanditActor (Ray Named Actor) to select optimal
    prompt templates via NeuralLinearUCB, and an EncoderActor to encode
    problems into context vectors.

    Supports both text-only and multimodal observations. The selected
    prompt is injected via ``env_instruction`` in the reset info dict,
    which TrajEnvManager prepends to the first user message.
    """

    def __init__(
        self,
        bandit_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._bandit_config = bandit_config or {}
        self._bandit_actor_name = self._bandit_config.get("bandit_actor_name", DEFAULT_BANDIT_ACTOR_NAME)
        self._encoder_actor_name = self._bandit_config.get("encoder_actor_name", DEFAULT_ENCODER_ACTOR_NAME)

        # Lazy-resolved actor handles
        self._bandit_actor = None
        self._encoder_actor = None

        # Per-episode state for reward feedback
        self._current_arm_idx: Optional[int] = None
        self._current_context_bytes: Optional[bytes] = None

    @property
    def bandit_actor(self):
        """Lazy resolution of BanditActor via Ray Named Actor."""
        if self._bandit_actor is None:
            try:
                self._bandit_actor = ray.get_actor(self._bandit_actor_name, namespace=RAY_NAMESPACE)
                logger.info(f"[MathBanditEnv] Connected to BanditActor: {self._bandit_actor_name}")
            except ValueError:
                logger.error(f"[MathBanditEnv] BanditActor '{self._bandit_actor_name}' not found")
                raise
        return self._bandit_actor

    @property
    def encoder_actor(self):
        """Lazy resolution of EncoderActor via Ray Named Actor."""
        if self._encoder_actor is None:
            try:
                self._encoder_actor = ray.get_actor(self._encoder_actor_name, namespace=RAY_NAMESPACE)
                logger.info(f"[MathBanditEnv] Connected to EncoderActor: {self._encoder_actor_name}")
            except ValueError:
                logger.error(f"[MathBanditEnv] EncoderActor '{self._encoder_actor_name}' not found")
                raise
        return self._encoder_actor

    def reset(self, seed: Optional[None] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment: sample question, encode it, select prompt via bandit."""
        obs, info = super().reset(seed=seed)

        # Handle end-of-dataset
        if obs is None:
            return None, None

        # Convert observation to encoder input (supports text, image, dict)
        encoder_input = _obs_to_encoder_input(obs)
        context_bytes = ray.get(self.encoder_actor.encode.remote(encoder_input))
        self._current_context_bytes = context_bytes
        self._current_problem_text = obs if isinstance(obs, str) else str(obs)[:300]

        # Select prompt via bandit
        selection = ray.get(self.bandit_actor.select_arm.remote(context_bytes))
        self._current_arm_idx = selection["arm_idx"]

        # Get prompt templates from bandit config
        prompt_templates = self._bandit_config.get("prompt_templates", [])
        if self._current_arm_idx < len(prompt_templates):
            prompt_text = prompt_templates[self._current_arm_idx]
        else:
            prompt_text = ""
            logger.warning(
                f"[MathBanditEnv] arm_idx={self._current_arm_idx} out of range "
                f"(n_templates={len(prompt_templates)}), using empty prompt"
            )

        # Inject prompt via env_instruction
        info["env_instruction"] = prompt_text

        logger.debug(
            f"[MathBanditEnv] Selected prompt {self._current_arm_idx} "
            f"(UCB={selection['ucb_value']:.3f})"
        )

        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Step: compute reward and feed back to bandit."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Update bandit with reward feedback
        if self._current_arm_idx is not None and self._current_context_bytes is not None:
            try:
                ray.get(
                    self.bandit_actor.update.remote(
                        self._current_arm_idx,
                        self._current_context_bytes,
                        float(reward),
                        problem_text=getattr(self, "_current_problem_text", ""),
                    )
                )
                # Add bandit metrics to info
                info.setdefault("metrics", {})
                info["metrics"]["bandit_arm_idx"] = self._current_arm_idx
                info["metrics"]["bandit_reward"] = float(reward)

                info.setdefault("metrics_agg_mode", {})
                info["metrics_agg_mode"]["bandit_arm_idx"] = "last"
                info["metrics_agg_mode"]["bandit_reward"] = "mean"
            except Exception as e:
                logger.warning(f"[MathBanditEnv] Failed to update bandit: {e}")

        # Reset per-episode state
        self._current_arm_idx = None
        self._current_context_bytes = None

        return obs, reward, terminated, truncated, info
