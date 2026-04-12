import logging
import multiprocessing
import random
import re
from typing import Optional, Tuple, Any, SupportsFloat, Dict

from datasets import load_dataset, Dataset, DatasetDict
from gem import Env
from gem.envs.math_env import MathEnv as GEMMathEnv
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_last_boxed_answer
import ray

from roll.datasets.global_dataset import GlobalDataset, GlobalDatasetManager
from roll.utils.constants import RAY_NAMESPACE

logger = logging.getLogger(__name__)


_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_UNIT_STRIP_RE = re.compile(r"[°\s]+$")


def extract_answer_tag(text: str) -> Optional[str]:
    """Extract the last <answer>...</answer> content, stripping trailing units like °."""
    if not text:
        return None
    matches = _ANSWER_TAG_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].strip()
    raw = _UNIT_STRIP_RE.sub("", raw)
    return raw or None


class MathEnv(GEMMathEnv):

    def __init__(
            self,
            dataset_name: Optional[str] = "",
            split: Optional[str] = None,
            dataset: Optional[Dataset] = None,
            question_key: str = "problem",
            answer_key: str = "answer",
            image_key: Optional[str] = "image",
            answer_format: str = "boxed",
            seed: int = 0,
            mode: str = "train",
            **_,
    ):
        Env.__init__(self)
        self.seed = seed
        self.question_key = question_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.answer_format = answer_format
        self.mode = mode

        # Convert train/val mode to sample/traversal for GlobalDataset
        global_dataset_mode = "sample" if self.mode == "train" else "traversal"
        self.dataset = GlobalDataset.options(name=f"{self.mode}_{dataset_name}",
                                             get_if_exists=True,
                                             namespace=RAY_NAMESPACE).remote(dataset_name=dataset_name,
                                                                             split=split,
                                                                             mode=global_dataset_mode)
        self.dataset_manager = GlobalDatasetManager.options(name=f"{self.mode}_dataset_manager",
                                                            get_if_exists=True,
                                                            namespace=RAY_NAMESPACE).remote()
        ray.get(self.dataset_manager.register.remote(dataset_name=dataset_name, dataset_ref=self.dataset))
        self.idx = 0
        self.epoch = 0
        # Process pool is used to enable the timeout mechanism for answer grading in a potential distributed training setup
        self.mp_pool = multiprocessing.Pool(1)

    def reset(self, seed: Optional[None] = None) -> Tuple[Any, dict[str, Any]]:
        """Sample a question from the dataset.

        Returns observation as a string for text-only datasets, or as a dict
        ``{"prompt": text, "image": PIL.Image}`` when an image field is present.
        """
        Env.reset(self, seed)
        data: Optional[Dict] = ray.get(self.dataset.get_data_item.remote(seed=seed))
        if data is None:
            return None, None
        question = data[self.question_key]
        raw_answer = data[self.answer_key]
        # Pre-normalize ground-truth for answer_tag format so check_correct can
        # do plain string equality with the model's extracted answer.
        if self.answer_format == "answer_tag":
            normalized = extract_answer_tag(raw_answer) if isinstance(raw_answer, str) else None
            self.answer = normalized if normalized is not None else str(raw_answer).strip()
        else:
            self.answer = raw_answer
        # Build observation. If dataset includes an image field, return a dict
        # so downstream env managers (VLTrajEnvManager) can attach the image.
        if self.image_key and self.image_key in data and data[self.image_key] is not None:
            self.first_obs = {"prompt": question, "image": data[self.image_key]}
        else:
            self.first_obs = question
        self.idx += 1
        return self.first_obs, {"env_instruction": ""}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.answer_format == "answer_tag":
            model_answer = extract_answer_tag(action)
        else:
            model_answer = extract_last_boxed_answer(action)
        action_is_valid = True
        if model_answer is None:
            reward = 0
            action_is_valid = False
        else:
            res = self.mp_pool.apply_async(
                self.check_correct, (model_answer, self.answer)
            )
            try:
                is_correct = res.get(timeout=1)
            except (multiprocessing.context.TimeoutError, Exception):
                is_correct = False
            reward = 1.0 if is_correct else 0

        metrics = {
            "action_is_valid": action_is_valid,
            "success": reward > 0,
            "raw_reward": reward,
        }
        metrics_agg_mode = {
            "action_is_valid": "mean",
            "success": "last",
            "raw_reward": "last",
        }
        info = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode
        }
        return TERMINAL_STATE, reward, True, True, info