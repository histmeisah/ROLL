"""
Dataset loader for mathematical reasoning tasks.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MathDataset:
    """
    Mathematical reasoning dataset loader.

    Supports:
    - GSM8K: Grade school math problems
    - MATH: Competition-level math problems
    - Custom datasets
    """

    def __init__(
        self,
        dataset_name: str = "gsm8k",
        split: str = "train",
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            dataset_name: Name of dataset (gsm8k, math, etc.)
            split: Data split (train, test, validation)
            seed: Random seed for sampling
            max_samples: Maximum number of samples to load (None = all)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.rng = np.random.RandomState(seed)
        self.problems = []

        # Load dataset
        self._load_dataset(max_samples)

        logger.info(
            f"Loaded {len(self.problems)} problems from "
            f"{dataset_name}/{split}"
        )

    def _load_dataset(self, max_samples: Optional[int] = None):
        """Load dataset from HuggingFace or local source."""
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error(
                "datasets library not installed. "
                "Install with: pip install datasets"
            )
            raise

        if self.dataset_name == "gsm8k":
            self._load_gsm8k(max_samples)
        elif self.dataset_name == "math":
            self._load_math(max_samples)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _load_gsm8k(self, max_samples: Optional[int] = None):
        """Load GSM8K dataset."""
        from datasets import load_dataset

        try:
            dataset = load_dataset("gsm8k", "main", split=self.split)
        except Exception as e:
            logger.warning(f"Failed to load gsm8k: {e}. Using fallback data.")
            self._load_fallback_data()
            return

        for idx, example in enumerate(dataset):
            if max_samples and idx >= max_samples:
                break

            # GSM8K format: answer contains "#### FINAL_ANSWER"
            answer_text = example["answer"]
            if "####" in answer_text:
                final_answer = answer_text.split("####")[-1].strip()
            else:
                final_answer = answer_text.strip()

            self.problems.append({
                "problem": example["question"],
                "answer": final_answer,
                "full_solution": answer_text,
                "dataset": "gsm8k",
            })

    def _load_math(self, max_samples: Optional[int] = None):
        """Load MATH dataset."""
        from datasets import load_dataset

        try:
            dataset = load_dataset("hendrycks/math", "all", split=self.split)
        except Exception as e:
            logger.warning(f"Failed to load MATH dataset: {e}. Using fallback data.")
            self._load_fallback_data()
            return

        for idx, example in enumerate(dataset):
            if max_samples and idx >= max_samples:
                break

            self.problems.append({
                "problem": example["problem"],
                "answer": example["solution"],
                "full_solution": example["solution"],
                "dataset": "math",
                "level": example.get("level", "unknown"),
                "type": example.get("type", "unknown"),
            })

    def _load_fallback_data(self):
        """Load fallback data for testing when datasets unavailable."""
        self.problems = [
            {
                "problem": "John has 5 apples. He gives 2 to Mary. How many apples does John have left?",
                "answer": "3",
                "full_solution": "John starts with 5 apples.\nHe gives away 2 apples.\n5 - 2 = 3\n#### 3",
                "dataset": "fallback",
            },
            {
                "problem": "A store sells pencils for $0.50 each. How much do 8 pencils cost?",
                "answer": "4",
                "full_solution": "Each pencil costs $0.50.\nWe need 8 pencils.\n0.50 * 8 = 4\n#### 4",
                "dataset": "fallback",
            },
            {
                "problem": "If x + 5 = 12, what is x?",
                "answer": "7",
                "full_solution": "x + 5 = 12\nx = 12 - 5\nx = 7\n#### 7",
                "dataset": "fallback",
            },
            {
                "problem": "What is 15% of 200?",
                "answer": "30",
                "full_solution": "15% = 0.15\n0.15 * 200 = 30\n#### 30",
                "dataset": "fallback",
            },
            {
                "problem": "A rectangle has length 8 and width 5. What is its area?",
                "answer": "40",
                "full_solution": "Area = length * width\nArea = 8 * 5\nArea = 40\n#### 40",
                "dataset": "fallback",
            },
        ]

        # Duplicate to have more samples
        self.problems = self.problems * 20

        logger.info(f"Using {len(self.problems)} fallback problems for testing")

    def sample(self) -> Dict[str, str]:
        """
        Sample a random problem.

        Returns:
            Dictionary with 'problem' and 'answer' keys
        """
        idx = self.rng.randint(0, len(self.problems))
        return self.problems[idx]

    def get_batch(self, batch_size: int) -> List[Dict[str, str]]:
        """
        Sample a batch of problems.

        Args:
            batch_size: Number of problems to sample

        Returns:
            List of problem dictionaries
        """
        indices = self.rng.randint(0, len(self.problems), size=batch_size)
        return [self.problems[i] for i in indices]

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.problems)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """Get problem by index."""
        return self.problems[idx]
