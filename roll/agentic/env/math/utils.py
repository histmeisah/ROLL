import logging
import multiprocessing
import re
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Registry
# =============================================================================

REGISTERED_MATH_DATASETS: Dict[str, Dict[str, Any]] = {
    "aime": {
        "config": {"path": "AI-MO/aimo-validation-aime"},
        "question_key": "problem",
        "answer_key": "answer",
        "extract_answer": False,
    },
    "math": {
        "config": {"path": "EleutherAI/hendrycks_math", "name": "algebra"},
        "question_key": "problem",
        "answer_key": "solution",
        "extract_answer": True,          # Need to extract from \boxed{}
    },
    "math_all": {
        # Load all 7 subjects and concatenate - use "math" for single subject
        "config": {"path": "EleutherAI/hendrycks_math"},
        "subjects": ["algebra", "counting_and_probability", "geometry",
                      "intermediate_algebra", "number_theory", "prealgebra", "precalculus"],
        "question_key": "problem",
        "answer_key": "solution",
        "extract_answer": True,
    },
    "gsm8k": {
        "config": {"path": "openai/gsm8k", "name": "main"},
        "question_key": "question",
        "answer_key": "answer",
        "extract_answer": True,          # Need to extract from "#### xxx"
    },
}


# =============================================================================
# Answer Extraction
# =============================================================================

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...}, handling nested braces.

    Adapted from math_rule_reward_worker.check_and_extract_within_boxed.
    """
    boxed_starts = ["\\boxed{", "\\boxed\\{"]
    last_index = -1
    content_start = 0

    for prefix in boxed_starts:
        idx = text.rfind(prefix)
        if idx != -1 and idx > last_index:
            last_index = idx
            content_start = idx + len(prefix)

    if last_index == -1:
        return None

    # Count braces to find matching closing brace
    depth = 0
    in_quotes = False
    for i in range(content_start, len(text)):
        ch = text[i]
        if ch == '"':
            in_quotes = not in_quotes
        elif not in_quotes and ch == '{':
            depth += 1
        elif not in_quotes and ch == '}':
            if depth == 0:
                return text[content_start:i].strip()
            depth -= 1

    # No matching brace found, return everything after \boxed{
    return text[content_start:].strip()


def extract_answer_tag(text: str, pattern: str = r"<answer>(.*?)</answer>") -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract numeric answer from GSM8K format '#### xxx'."""
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip()
    return None


def extract_ground_truth(raw_answer: str, dataset_name: str) -> str:
    """Extract clean ground truth from dataset-specific answer format."""
    dataset_info = REGISTERED_MATH_DATASETS.get(dataset_name, {})

    if not dataset_info.get("extract_answer", False):
        return raw_answer.strip()

    if dataset_name == "math":
        # Hendrycks MATH: answer is in \boxed{} within the solution field
        extracted = extract_boxed_answer(raw_answer)
        return extracted if extracted else raw_answer.strip()

    if dataset_name == "gsm8k":
        # GSM8K: answer follows "#### " marker
        extracted = extract_gsm8k_answer(raw_answer)
        return extracted if extracted else raw_answer.strip()

    return raw_answer.strip()


# =============================================================================
# Answer Verification
# =============================================================================

def _normalize_answer(text: str) -> str:
    """Normalize answer string for comparison."""
    text = text.strip()
    # Remove trailing period
    if text.endswith('.'):
        text = text[:-1]
    # Remove dollar signs and whitespace
    text = text.replace('$', '').replace(' ', '')
    # Remove leading zeros in numbers (but keep "0" itself)
    text = re.sub(r'^0+(\d)', r'\1', text)
    # Normalize fractions: remove spaces around /
    text = re.sub(r'\s*/\s*', '/', text)
    return text.lower()


def _check_numeric_equal(pred: str, gt: str) -> bool:
    """Check if two strings represent the same number."""
    try:
        # Handle percentages
        pred_clean = pred.replace('%', '').replace(',', '')
        gt_clean = gt.replace('%', '').replace(',', '')
        return abs(float(pred_clean) - float(gt_clean)) < 1e-6
    except (ValueError, TypeError):
        return False


def _verify_with_math_verify(prediction: str, ground_truth: str, result_list: list) -> None:
    """Verify using math-verify library (run in subprocess for timeout safety).

    Adapted from math_rule_reward_worker._hf_verify_math_sample.
    """
    try:
        from math_verify import parse, verify

        parsed_pred = parse(f"${prediction}$", fallback_mode="no_fallback")
        parsed_gt = parse(f"${ground_truth}$")

        if not parsed_pred or not parsed_gt:
            result_list.append(False)
            return

        is_correct = verify(parsed_gt[0], parsed_pred[0])
        result_list.append(bool(is_correct))
    except Exception:
        result_list.append(False)


def check_math_answer(
    prediction: str,
    ground_truth: str,
    use_math_verify: bool = True,
    timeout: float = 5.0,
) -> bool:
    """Verify a math answer against ground truth.

    Strategy:
    1. Numeric comparison
    2. Normalized string comparison
    3. math-verify library (if enabled, with timeout)
    """
    pred_norm = _normalize_answer(prediction)
    gt_norm = _normalize_answer(ground_truth)

    # 1. Exact normalized match
    if pred_norm == gt_norm:
        return True

    # 2. Numeric comparison
    if _check_numeric_equal(pred_norm, gt_norm):
        return True

    # 3. math-verify with subprocess timeout
    if use_math_verify:
        try:
            with multiprocessing.Manager() as manager:
                result_list = manager.list()
                p = multiprocessing.Process(
                    target=_verify_with_math_verify,
                    args=(prediction, ground_truth, result_list),
                )
                p.start()
                p.join(timeout=timeout)

                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2)
                    if p.is_alive():
                        p.kill()
                    p.join(timeout=2)
                    return False

                if result_list:
                    return result_list[0]
        except Exception:
            pass

    return False
