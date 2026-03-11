"""Lightweight CPU-based scoring for SVG reconstruction.

All metrics operate on numpy uint8 arrays of shape (H, W, 3).
No GPU or deep-learning models required.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def calculate_structural_accuracy(gt: np.ndarray, gen: np.ndarray) -> float:
    """Canny-edge IoU between ground-truth and generated images.

    Ported from VAGEN ``score.py``.  Uses OpenCV Canny edge detection and
    computes intersection-over-union of the edge maps.

    Returns a float in [0, 1].
    """
    try:
        import cv2
    except ImportError:
        return 0.0

    gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
    gen_gray = cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY)

    gt_edges = cv2.Canny(gt_gray, 50, 150)
    gen_edges = cv2.Canny(gen_gray, 50, 150)

    intersection = np.logical_and(gt_edges > 0, gen_edges > 0).sum()
    union = np.logical_or(gt_edges > 0, gen_edges > 0).sum()

    if union == 0:
        # Both images have no edges – consider them structurally identical.
        return 1.0
    return float(intersection / union)


def calculate_pixel_similarity(gt: np.ndarray, gen: np.ndarray) -> float:
    """Pixel-level similarity: ``1 - normalised_MSE``.

    MSE is normalised to [0, 1] by dividing by 255**2.
    Returns a float in [0, 1] where 1 means identical.
    """
    mse = np.mean((gt.astype(np.float64) - gen.astype(np.float64)) ** 2)
    normalised_mse = mse / (255.0 ** 2)
    return float(1.0 - normalised_mse)


def calculate_ssim_score(gt: np.ndarray, gen: np.ndarray) -> float:
    """Structural Similarity Index (SSIM) via scikit-image.

    Operates on the full RGB image (``channel_axis=2``).
    Returns a float in [-1, 1] (typically [0, 1] for natural images).
    Falls back to pixel similarity if scikit-image is unavailable.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return calculate_pixel_similarity(gt, gen)

    score = ssim(gt, gen, channel_axis=2, data_range=255)
    return float(score)


# ---------------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------------

@dataclass
class ScoreWeights:
    """Weights for the three scoring components."""
    structural: float = 0.4
    pixel_mse: float = 0.3
    ssim: float = 0.3


def calculate_total_score(
    gt: np.ndarray,
    gen: np.ndarray,
    weights: Optional[ScoreWeights] = None,
) -> dict:
    """Compute all metrics and return a dict with individual + total scores.

    Args:
        gt: Ground-truth image, (H, W, 3) uint8.
        gen: Generated image, (H, W, 3) uint8.
        weights: Scoring weights.  Defaults to equal-ish split.

    Returns:
        Dictionary with keys ``structural_score``, ``pixel_score``,
        ``ssim_score``, and ``total_score``.
    """
    if weights is None:
        weights = ScoreWeights()

    structural = calculate_structural_accuracy(gt, gen)
    pixel = calculate_pixel_similarity(gt, gen)
    ssim_val = calculate_ssim_score(gt, gen)

    total = (
        weights.structural * structural
        + weights.pixel_mse * pixel
        + weights.ssim * ssim_val
    )

    return {
        "structural_score": structural,
        "pixel_score": pixel,
        "ssim_score": ssim_val,
        "total_score": total,
    }
