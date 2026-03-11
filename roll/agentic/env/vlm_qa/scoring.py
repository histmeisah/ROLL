"""Answer matching utilities for VLM QA environments.

Supports exact string matching, numeric tolerance matching, and
multiple-choice letter matching.
"""

import re
from typing import Optional


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison.

    Strips whitespace, lowercases, and removes common prefixes like
    "the answer is", trailing punctuation, dollar signs, percent signs, etc.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.strip().lower()

    # Remove common answer prefixes
    for prefix in [
        "the answer is",
        "answer:",
        "answer is",
        "the final answer is",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Strip surrounding quotes
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()

    # Remove trailing period
    if text.endswith("."):
        text = text[:-1].strip()

    # Remove $, %, degree symbols (keep the numeric content)
    text = text.replace("$", "").replace("%", "").replace("\\%", "")
    text = text.replace("\u00b0", "").replace("\\degree", "")

    # Remove leading/trailing whitespace again
    text = text.strip()
    return text


def extract_number(text: str) -> Optional[float]:
    """Try to extract a numeric value from text.

    Handles integers, decimals, negative numbers, and fractions.
    Returns None if no number can be extracted.
    """
    text = normalize_answer(text)

    # Handle fractions like "3/4"
    frac_match = re.match(r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$", text)
    if frac_match:
        numer = float(frac_match.group(1))
        denom = float(frac_match.group(2))
        if denom != 0:
            return numer / denom
        return None

    # Try direct float conversion
    try:
        return float(text)
    except ValueError:
        pass

    # Try to find a number in the text
    num_match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if num_match:
        num_str = num_match.group(0).replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            pass

    return None


def match_answer(
    prediction: str,
    ground_truth: str,
    tolerance: float = 0.01,
    answer_type: str = "auto",
) -> bool:
    """Check if prediction matches ground truth.

    Matching strategies (tried in order for ``answer_type="auto"``):
    1. Exact string match (case-insensitive, normalized)
    2. Multiple-choice letter match (A/B/C/D/E)
    3. Numeric tolerance match (relative error < tolerance)

    Args:
        prediction: Model's predicted answer.
        ground_truth: Ground truth answer.
        tolerance: Relative tolerance for numeric comparison.
        answer_type: One of "auto", "multi_choice", "free_form".
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    if not pred_norm or not gt_norm:
        return False

    # 1. Exact string match
    if pred_norm == gt_norm:
        return True

    # 2. MC letter match  (e.g. "a" == "a", also "a)" == "a")
    if answer_type in ("auto", "multi_choice"):
        pred_letter = re.match(r"^([a-e])\)?$", pred_norm)
        gt_letter = re.match(r"^([a-e])\)?$", gt_norm)
        if pred_letter and gt_letter:
            return pred_letter.group(1) == gt_letter.group(1)

    # 3. Numeric tolerance match
    if answer_type in ("auto", "free_form"):
        pred_num = extract_number(prediction)
        gt_num = extract_number(ground_truth)
        if pred_num is not None and gt_num is not None:
            if gt_num == 0:
                return abs(pred_num) < tolerance
            return abs(pred_num - gt_num) / abs(gt_num) < tolerance

    return False
