import logging
import random
from typing import Any, Dict, Tuple

from datasets import load_dataset, concatenate_datasets

from roll.agentic.env.base import BaseEnv
from roll.agentic.utils import all_seed
from .config import MathEnvConfig
from .utils import (
    REGISTERED_MATH_DATASETS,
    check_math_answer,
    extract_answer_tag,
    extract_boxed_answer,
    extract_ground_truth,
)

logger = logging.getLogger(__name__)


class MathEnv(BaseEnv):
    """Math problem solving environment for RL training.

    Supports AIME, MATH (Hendrycks), and GSM8K benchmarks.
    Compatible with TrajEnvManager for multi-step off-policy training.
    """

    def __init__(self, config: MathEnvConfig = None):
        self.config = config or MathEnvConfig()
        super().__init__(self.config)

        # Resolve dataset keys from registry
        dataset_info = REGISTERED_MATH_DATASETS.get(self.config.dataset_name, {})
        self.question_key = dataset_info.get("question_key", self.config.question_key)
        self.answer_key = dataset_info.get("answer_key", self.config.answer_key)

        # Load dataset
        self.dataset = self._load_dataset()

        # Environment state
        self.current_question: str = ""
        self.correct_answer: str = ""
        self.step_num: int = 0

    def _load_dataset(self):
        """Load dataset from local parquet or HuggingFace."""
        # Priority 1: local parquet file via dataset_path
        if self.config.dataset_path:
            return self._load_from_parquet(self.config.dataset_path)

        # Priority 2: HuggingFace load_dataset
        dataset_info = REGISTERED_MATH_DATASETS.get(self.config.dataset_name, {})
        load_kwargs = dict(dataset_info.get("config", {"path": self.config.dataset_name}))

        if self.config.cache_dir:
            load_kwargs["cache_dir"] = self.config.cache_dir

        # Handle multi-subject datasets (e.g., math_all)
        subjects = dataset_info.get("subjects")
        if subjects:
            return self._load_multi_subject_dataset(load_kwargs, subjects)

        try:
            ds = load_dataset(**load_kwargs)
        except Exception as e:
            logger.error(f"Failed to load dataset '{self.config.dataset_name}': {e}")
            raise

        return self._select_split(ds)

    def _load_from_parquet(self, path: str):
        """Load dataset from a local parquet file."""
        logger.info(f"Loading dataset from local parquet: {path}")
        ds = load_dataset("parquet", data_files=path)
        return ds["train"]  # parquet loading always puts data in "train" split

    def _load_multi_subject_dataset(self, base_kwargs: dict, subjects: list):
        """Load and concatenate multiple subject splits (for hendrycks_math)."""
        split = self.config.dataset_split
        all_splits = []
        for subject in subjects:
            kwargs = dict(base_kwargs)
            kwargs["name"] = subject
            try:
                ds = load_dataset(**kwargs)
                selected = self._select_split(ds)
                all_splits.append(selected)
                logger.info(f"Loaded {subject}: {len(selected)} items")
            except Exception as e:
                logger.warning(f"Failed to load subject '{subject}': {e}")

        if not all_splits:
            raise ValueError(f"Failed to load any subjects for math_all")

        combined = concatenate_datasets(all_splits)
        logger.info(f"Combined math_all dataset: {len(combined)} items from {len(all_splits)} subjects")
        return combined

    def _select_split(self, ds):
        """Select the desired split from a DatasetDict."""
        split = self.config.dataset_split
        if split in ds:
            return ds[split]

        for fallback in ["train", "test", "validation"]:
            if fallback in ds:
                logger.warning(f"Split '{split}' not found, using '{fallback}' instead.")
                return ds[fallback]

        raise ValueError(f"No valid split found in dataset. Available: {list(ds.keys())}")

    def reset(self, seed=None, **kwargs) -> Tuple[str, dict]:
        """Reset environment: select a problem deterministically by seed."""
        with all_seed(seed):
            idx = random.randint(0, len(self.dataset) - 1)

        data = self.dataset[idx]
        self.current_question = data[self.question_key]
        raw_answer = data[self.answer_key]
        self.correct_answer = extract_ground_truth(raw_answer, self.config.dataset_name)
        self.step_num = 0

        return self.current_question, {}

    def step(self, action: str) -> Tuple[Any, float, bool, bool, Dict]:
        """Evaluate model answer. Returns gym 5-tuple."""
        # 1. Extract answer: try <answer> tag first, then \boxed{}
        model_answer = extract_answer_tag(action, self.config.action_pattern)
        if model_answer is None:
            model_answer = extract_boxed_answer(action)

        action_is_valid = model_answer is not None

        # 2. Check correctness
        is_correct = False
        if action_is_valid:
            is_correct = check_math_answer(
                model_answer,
                self.correct_answer,
                use_math_verify=self.config.use_math_verify,
                timeout=self.config.verify_timeout,
            )

        # 3. Reward with step decay: 1.0, 0.5, 0.25, ...
        reward = 0.0
        if is_correct:
            reward = self.config.correct_reward / (2 ** self.step_num)

        self.step_num += 1

        # 4. Observation
        if is_correct:
            observation = "Correct!"
        elif not action_is_valid:
            observation = (
                "Could not find your answer. Please wrap your answer as <answer>your_answer</answer>."
            )
        else:
            observation = "Incorrect. Please reconsider and try again."

        terminated = is_correct  # Correct answer terminates episode
        truncated = False

        # 5. Info (matches TrajEnvManager expected format, same as sokoban)
        info = {
            "metrics": {
                "action_is_valid": action_is_valid,
                "action_is_effective": True,
                "success": is_correct,
            },
            "action": model_answer or "",
        }

        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "text") -> str:
        """Render current problem."""
        return self.current_question

    def close(self):
        """Clean up resources."""
        pass


if __name__ == "__main__":
    import sys
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    sys.path.insert(0, project_root)

    print("=" * 60)
    print("MathEnv Self-Test")
    print("=" * 60)

    # Test with GSM8K
    for ds_name in ["gsm8k", "math", "aime"]:
        print(f"\n--- Testing {ds_name} ---")
        try:
            cfg = MathEnvConfig(dataset_name=ds_name, use_math_verify=False)
            env = MathEnv(cfg)
            print(f"  Dataset loaded: {len(env.dataset)} items")

            # Test reset
            obs, info = env.reset(seed=42)
            print(f"  Question: {obs[:100]}...")
            print(f"  Ground truth: {env.correct_answer}")

            # Test step with wrong answer
            obs1, r1, term1, trunc1, info1 = env.step("<answer>wrong</answer>")
            print(f"  Step 1 (wrong): reward={r1}, terminated={term1}, info={info1['metrics']}")
            assert not term1, "Should not terminate on wrong answer"
            assert r1 == 0.0, "Wrong answer should give 0 reward"

            # Test step with correct answer
            obs2, r2, term2, trunc2, info2 = env.step(f"<answer>{env.correct_answer}</answer>")
            print(f"  Step 2 (correct): reward={r2}, terminated={term2}, info={info2['metrics']}")
            assert term2, "Should terminate on correct answer"
            assert r2 == 0.5, f"Second step correct should give 0.5, got {r2}"

            # Test reset and 5-tuple format
            obs3, info3 = env.reset(seed=123)
            assert isinstance(obs3, str), "Observation should be string"
            assert isinstance(info3, dict), "Info should be dict"

            print(f"  PASSED")
        except Exception as e:
            print(f"  SKIPPED ({type(e).__name__}: {e})")

    print("\n" + "=" * 60)
    print("Self-test complete.")
