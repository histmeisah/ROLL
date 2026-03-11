"""SVG Reconstruction environment for ROLL.

The agent sees a target image and must generate SVG code to reconstruct it.
Supports multi-turn interaction: after each attempt the agent receives a
side-by-side comparison image (ground-truth left, generated right) so it can
iteratively refine its output.

Designed for VLM training via ``VLTrajEnvManager``.
"""

import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from roll.agentic.env.base import BaseEnv
from roll.agentic.utils import all_seed

from .config import SVGReconstructionEnvConfig
from .scoring import ScoreWeights, calculate_total_score
from .svg_utils import (
    create_side_by_side,
    extract_svg_from_text,
    load_svg_dataset,
    pil_to_numpy,
    process_and_rasterize_svg,
)

logger = logging.getLogger(__name__)


class SVGReconstructionEnv(BaseEnv):
    """SVG Reconstruction environment.

    The observation is a numpy uint8 array of shape ``(H, W, 3)`` or
    ``(H, 2*W+gap, 3)`` (side-by-side comparison after a step).

    This environment is intended to be used with ``VLTrajEnvManager``
    which converts the numpy array to a PIL Image for the VLM.
    """

    # ---- Class-level dataset cache (shared across instances) ----
    _dataset_cache: Dict[str, Any] = {}
    _dataset_lock = threading.Lock()

    def __init__(self, config: SVGReconstructionEnvConfig = SVGReconstructionEnvConfig()):
        super().__init__(config)
        self.config: SVGReconstructionEnvConfig = config
        self.render_mode: str = config.render_mode

        # Per-instance state (set in reset)
        self._gt_svg: Optional[str] = None
        self._gt_image: Optional[np.ndarray] = None  # (H, W, 3) uint8
        self._gen_image: Optional[np.ndarray] = None  # latest generated image
        self._step_count: int = 0
        self._best_score: float = 0.0

        # Scoring weights
        self._score_weights = ScoreWeights(
            structural=config.structural_weight,
            pixel_mse=config.pixel_mse_weight,
            ssim=config.ssim_weight,
        )

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------

    def _get_dataset(self):
        """Load dataset with class-level caching."""
        cache_key = f"{self.config.dataset_name}|{self.config.dataset_split}"
        if cache_key not in self._dataset_cache:
            with self._dataset_lock:
                # Double-check after acquiring lock
                if cache_key not in self._dataset_cache:
                    logger.info(
                        "Loading SVG dataset: %s (split=%s)",
                        self.config.dataset_name,
                        self.config.dataset_split,
                    )
                    ds = load_svg_dataset(
                        self.config.dataset_name,
                        split=self.config.dataset_split,
                        cache_dir=self.config.dataset_cache_dir,
                    )
                    self._dataset_cache[cache_key] = ds
        return self._dataset_cache[cache_key]

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset the environment with a new target SVG.

        Uses ``all_seed(seed)`` for deterministic sample selection.
        Returns the rendered target image as a numpy array.
        """
        self._step_count = 0
        self._gen_image = None
        self._best_score = 0.0

        dataset = self._get_dataset()

        with all_seed(seed):
            idx = np.random.randint(0, len(dataset))

        item = dataset[idx]

        # The dataset may provide SVG as text and/or a pre-rendered image.
        # Try to get the SVG source first; fall back to the image column.
        self._gt_svg = item.get("svg", item.get("svg_code", None))

        # Rasterize ground-truth
        gt_pil: Optional[Image.Image] = None
        if self._gt_svg is not None:
            gt_pil = process_and_rasterize_svg(
                self._gt_svg,
                resolution=self.config.image_resolution,
                timeout=self.config.svg_timeout,
            )

        # Fallback: use image column from dataset directly
        if gt_pil is None and "image" in item:
            raw_img = item["image"]
            if isinstance(raw_img, Image.Image):
                gt_pil = raw_img.convert("RGB").resize(
                    (self.config.image_resolution, self.config.image_resolution),
                    Image.LANCZOS,
                )
            elif isinstance(raw_img, np.ndarray):
                gt_pil = Image.fromarray(raw_img).convert("RGB").resize(
                    (self.config.image_resolution, self.config.image_resolution),
                    Image.LANCZOS,
                )

        if gt_pil is None:
            # Last resort: blank white image
            logger.warning("Could not rasterize ground-truth SVG for idx=%d, using blank", idx)
            gt_pil = Image.new("RGB", (self.config.image_resolution, self.config.image_resolution), (255, 255, 255))

        self._gt_image = pil_to_numpy(gt_pil, self.config.image_resolution)

        return self.render(), {}

    def step(self, action: str) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step: parse SVG from the action, rasterize, and score.

        Args:
            action: Raw LLM response text.

        Returns:
            observation, reward, terminated, truncated, info
        """
        self._step_count += 1
        action_info = self.parse_action(action)
        svg_code = action_info["action"]

        # Default: invalid action
        if svg_code is None:
            self._gen_image = self._blank_image()
            metrics = {
                "action_is_valid": False,
                "action_is_effective": False,
                "success": False,
                "structural_score": 0.0,
                "pixel_score": 0.0,
                "ssim_score": 0.0,
                "total_score": 0.0,
            }
            info: Dict[str, Any] = {"metrics": metrics}
            info.update(action_info)
            terminated = False
            truncated = self._step_count >= self.config.max_steps
            return self.render(), 0.0, terminated, truncated, info

        # Try to rasterize
        gen_pil = process_and_rasterize_svg(
            svg_code,
            resolution=self.config.image_resolution,
            timeout=self.config.svg_timeout,
        )

        if gen_pil is None:
            # Valid SVG syntax but rasterization failed
            self._gen_image = self._blank_image()
            metrics = {
                "action_is_valid": True,
                "action_is_effective": False,
                "success": False,
                "structural_score": 0.0,
                "pixel_score": 0.0,
                "ssim_score": 0.0,
                "total_score": 0.0,
            }
            info = {"metrics": metrics}
            info.update(action_info)
            terminated = False
            truncated = self._step_count >= self.config.max_steps
            return self.render(), 0.0, terminated, truncated, info

        # Rasterization succeeded – compute scores
        gen_np = pil_to_numpy(gen_pil, self.config.image_resolution)
        self._gen_image = gen_np

        scores = calculate_total_score(self._gt_image, gen_np, self._score_weights)
        reward = scores["total_score"]
        self._best_score = max(self._best_score, reward)

        metrics = {
            "action_is_valid": True,
            "action_is_effective": True,
            "success": reward >= 0.8,
            "structural_score": scores["structural_score"],
            "pixel_score": scores["pixel_score"],
            "ssim_score": scores["ssim_score"],
            "total_score": scores["total_score"],
        }
        info = {"metrics": metrics}
        info.update(action_info)

        # Multi-turn: never terminates early, rely on max_steps for truncation
        terminated = False
        truncated = self._step_count >= self.config.max_steps

        return self.render(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    def parse_action(self, text: str) -> Dict[str, Any]:
        """Parse LLM output to extract SVG code.

        Two-stage extraction:
        1. Extract content from ``<answer>...</answer>`` (with ``re.DOTALL``).
        2. Extract ``<svg>...</svg>`` from the answer content.

        This avoids issues with SVG angle brackets interfering with the
        answer tag matching.
        """
        if text is None:
            return {"action": None, "action_content": "", "think_content": ""}

        # Stage 1: extract <answer>...</answer>
        answer_match = re.search(self.config.action_pattern, text, re.DOTALL)
        if not answer_match:
            return {"action": None, "action_content": "", "think_content": ""}

        answer_content = answer_match.group(1).strip()

        # Remove special tokens
        if self.config.special_token_list:
            for token in self.config.special_token_list:
                answer_content = answer_content.replace(token, "").strip()

        # Stage 2: extract <svg>...</svg> from answer content
        svg_code = extract_svg_from_text(answer_content)

        # If no <svg> tag found, try to use raw answer content
        # (the agent might have output raw SVG without wrapping in <svg> tags)
        if svg_code is None and answer_content.strip().startswith("<"):
            svg_code = answer_content

        return {
            "action": svg_code,
            "action_content": answer_content,
            "think_content": "",
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, mode: Optional[str] = None) -> np.ndarray:
        """Render the current environment state as a numpy (H, W, 3) uint8 array.

        - After ``reset()``: returns the target image only.
        - After ``step()``: returns a side-by-side image (target | generated).
        """
        if mode is None:
            mode = self.render_mode

        if mode == "rgb_array":
            if self._gen_image is None:
                # Only target available
                return self._gt_image.copy()
            # Side-by-side comparison
            return create_side_by_side(self._gt_image, self._gen_image)
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _blank_image(self) -> np.ndarray:
        """Return a blank (white) image at the configured resolution."""
        res = self.config.image_resolution
        return np.full((res, res, 3), 255, dtype=np.uint8)

    def get_all_actions(self) -> List[str]:
        """SVG is free-form; no discrete action set."""
        return []

    def close(self):
        """Release per-instance state."""
        self._gt_image = None
        self._gen_image = None
        self._gt_svg = None


# ---------------------------------------------------------------------------
# Interactive testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    config = SVGReconstructionEnvConfig(
        image_resolution=128,
        max_steps=2,
    )
    env = SVGReconstructionEnv(config)

    print("=== SVG Reconstruction Environment Test ===\n")

    # Reset
    obs, info = env.reset(seed=42)
    print(f"Reset: observation shape = {obs.shape}, dtype = {obs.dtype}")

    # Step with a simple SVG
    test_svg = (
        '<answer>'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<circle cx="50" cy="50" r="40" fill="blue"/>'
        '</svg>'
        '</answer>'
    )
    obs, reward, terminated, truncated, info = env.step(test_svg)
    print(f"Step 1: shape={obs.shape}, reward={reward:.4f}, "
          f"terminated={terminated}, truncated={truncated}")
    print(f"  Metrics: {info['metrics']}")

    # Step with invalid action
    obs, reward, terminated, truncated, info = env.step("no svg here")
    print(f"Step 2 (invalid): shape={obs.shape}, reward={reward:.4f}, "
          f"terminated={terminated}, truncated={truncated}")
    print(f"  Metrics: {info['metrics']}")

    env.close()
    print("\nTest complete.")
