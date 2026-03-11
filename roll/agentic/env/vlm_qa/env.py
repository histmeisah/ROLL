"""Universal VLM QA environment for ROLL.

Supports visual question-answering datasets such as GeoQA, MathVista, and
TabMWP through configurable column mappings.  The agent sees an image (or a
rendered table) and a question, then answers inside ``<answer>`` tags.

Designed for VLM training via ``VLTrajEnvManager``.
"""

import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from roll.agentic.env.base import BaseEnv
from roll.agentic.utils import all_seed

from .config import VLMQAEnvConfig
from .scoring import match_answer

logger = logging.getLogger(__name__)


class VLMQAEnv(BaseEnv):
    """Universal VLM QA environment.

    On ``reset()`` a sample is drawn from the dataset, the question text is
    injected into ``config.env_instruction`` so that ``VLTrajEnvManager``
    picks it up in ``format_messages()``, and the image (or rendered table)
    is returned as the observation.

    On ``step()`` the model's answer is compared against the ground truth
    using :func:`scoring.match_answer`.
    """

    # Class-level dataset cache (shared across env instances in the same worker)
    _dataset_cache: Dict[str, Any] = {}
    _dataset_lock = threading.Lock()

    def __init__(self, config: VLMQAEnvConfig = VLMQAEnvConfig()):
        super().__init__(config)
        self.config: VLMQAEnvConfig = config
        self.render_mode: str = config.render_mode

        # Save original instruction so we can rebuild it each reset
        self._base_instruction: str = config.env_instruction

        # Per-instance state (set in reset)
        self._image: Optional[np.ndarray] = None      # (H, W, 3) uint8
        self._answer: str = ""
        self._answer_type: str = "auto"
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------

    def _get_dataset(self):
        """Load dataset with class-level caching and double-checked locking."""
        cache_key = (
            f"{self.config.dataset_name}|{self.config.dataset_subset}|"
            f"{self.config.dataset_split}|{self.config.group_id}|{self.config.group_size}"
        )
        if cache_key not in self._dataset_cache:
            with self._dataset_lock:
                if cache_key not in self._dataset_cache:
                    logger.info(
                        "Loading VLM QA dataset: %s (subset=%s, split=%s, "
                        "group=%d/%d)",
                        self.config.dataset_name,
                        self.config.dataset_subset or "<default>",
                        self.config.dataset_split,
                        self.config.group_id,
                        self.config.group_size,
                    )
                    import os
                    import glob
                    from datasets import load_dataset, Dataset

                    # If dataset_name is a local directory, load arrow files directly
                    # via pyarrow stream + strip metadata (bypasses datasets version
                    # incompatibility in dataset_info.json / schema metadata)
                    if os.path.isdir(self.config.dataset_name):
                        import pyarrow as pa
                        dataset_path = self.config.dataset_name
                        split_dir = os.path.join(dataset_path, self.config.dataset_split)
                        load_dir = split_dir if os.path.isdir(split_dir) else dataset_path

                        arrow_files = sorted(glob.glob(os.path.join(load_dir, "*.arrow")))
                        if arrow_files:
                            tables = []
                            for af in arrow_files:
                                with open(af, "rb") as fh:
                                    tables.append(pa.ipc.open_stream(fh).read_all())
                            table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
                            table = table.replace_schema_metadata(None)
                            ds = Dataset(table)
                            logger.info("Loaded dataset from arrow files: %s (%d files)", load_dir, len(arrow_files))
                        else:
                            raise FileNotFoundError(f"No .arrow files found in {load_dir}")
                    else:
                        load_kwargs: Dict[str, Any] = {
                            "split": self.config.dataset_split,
                        }
                        if self.config.dataset_subset:
                            load_kwargs["name"] = self.config.dataset_subset
                        if self.config.dataset_cache_dir:
                            load_kwargs["cache_dir"] = self.config.dataset_cache_dir

                        ds = load_dataset(self.config.dataset_name, **load_kwargs)

                    # Partition for multi-worker
                    if self.config.group_size > 1:
                        total = len(ds)
                        shard_size = total // self.config.group_size
                        start = self.config.group_id * shard_size
                        end = start + shard_size if self.config.group_id < self.config.group_size - 1 else total
                        ds = ds.select(range(start, end))

                    self._dataset_cache[cache_key] = ds
                    logger.info("Dataset loaded: %d samples", len(ds))
        return self._dataset_cache[cache_key]

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset with a new QA sample.  Same seed -> same sample."""
        self._step_count = 0
        dataset = self._get_dataset()

        with all_seed(seed):
            idx = np.random.randint(0, len(dataset))

        sample = dataset[idx]

        # --- Load image ---
        self._image = self._load_image(sample)

        # --- Extract ground-truth answer ---
        self._answer = self._extract_answer(sample)

        # --- Determine answer type ---
        if self.config.answer_type_column and self.config.answer_type_column in sample:
            raw_type = str(sample[self.config.answer_type_column]).lower()
            if "multi" in raw_type or "choice" in raw_type:
                self._answer_type = "multi_choice"
            else:
                self._answer_type = "free_form"
        else:
            self._answer_type = "auto"

        # --- Build dynamic env_instruction with question text ---
        self.config.env_instruction = self._build_instruction(sample)

        return self.render(), {}

    def step(self, action: str) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Evaluate the model's answer."""
        self._step_count += 1
        action_info = self.parse_action(action)
        prediction = action_info["action_content"]

        if not prediction:
            # Invalid format
            metrics = {
                "action_is_valid": False,
                "action_is_effective": False,
                "success": False,
            }
            info: Dict[str, Any] = {"metrics": metrics}
            info.update(action_info)
            truncated = self._step_count >= self.config.max_steps
            return self.render(), self.config.invalid_reward, False, truncated, info

        correct = match_answer(
            prediction,
            self._answer,
            tolerance=self.config.answer_tolerance,
            answer_type=self._answer_type,
        )

        if correct:
            reward = self.config.correct_reward
            terminated = True
        else:
            reward = self.config.wrong_reward
            terminated = False

        metrics = {
            "action_is_valid": True,
            "action_is_effective": True,
            "success": correct,
        }
        info = {"metrics": metrics}
        info.update(action_info)
        truncated = self._step_count >= self.config.max_steps
        return self.render(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    def parse_action(self, text: str) -> Dict[str, Any]:
        """Extract answer from ``<answer>...</answer>`` tags."""
        if text is None:
            return {"action": None, "action_content": "", "think_content": ""}

        match = re.search(self.config.action_pattern, text, re.DOTALL)
        if not match:
            return {"action": None, "action_content": "", "think_content": ""}

        if len(match.groups()) == 1:
            think_content, action_content = "", match.group(1).strip()
        else:
            think_content = match.group(1).strip()
            action_content = match.group(2).strip()

        # Remove special tokens
        if self.config.special_token_list:
            for token in self.config.special_token_list:
                action_content = action_content.replace(token, "").strip()
                think_content = think_content.replace(token, "").strip()

        return {
            "action": action_content,
            "action_content": action_content,
            "think_content": think_content,
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, mode: Optional[str] = None) -> np.ndarray:
        """Return the current image as numpy (H, W, 3) uint8."""
        if self._image is None:
            res = self.config.image_resolution
            return np.full((res, res, 3), 255, dtype=np.uint8)
        return self._image.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image(self, sample: Dict) -> np.ndarray:
        """Load image from dataset sample, or render table as image."""
        res = self.config.image_resolution

        # Try image column first
        if self.config.image_column and self.config.image_column in sample:
            raw = sample[self.config.image_column]
            if isinstance(raw, Image.Image):
                pil_img = raw.convert("RGB").resize((res, res), Image.LANCZOS)
                return np.array(pil_img, dtype=np.uint8)
            elif isinstance(raw, np.ndarray):
                pil_img = Image.fromarray(raw).convert("RGB").resize((res, res), Image.LANCZOS)
                return np.array(pil_img, dtype=np.uint8)
            elif isinstance(raw, bytes):
                import io
                pil_img = Image.open(io.BytesIO(raw)).convert("RGB").resize((res, res), Image.LANCZOS)
                return np.array(pil_img, dtype=np.uint8)

        # Render table as image (for TabMWP)
        if self.config.table_column and self.config.table_column in sample:
            table_text = str(sample[self.config.table_column])
            title = ""
            if self.config.table_title_column and self.config.table_title_column in sample:
                title = str(sample[self.config.table_title_column])
            return self._render_table(table_text, title, res)

        # Fallback: blank white image
        logger.warning("No image or table found in sample, using blank image")
        return np.full((res, res, 3), 255, dtype=np.uint8)

    def _render_table(self, table_text: str, title: str, resolution: int) -> np.ndarray:
        """Render table text as a PIL image, returned as numpy array."""
        img = Image.new("RGB", (resolution, resolution), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Use a basic font; try to get a monospace one
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", max(10, resolution // 25))
        except (IOError, OSError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                                          max(10, resolution // 25))
            except (IOError, OSError):
                font = ImageFont.load_default()

        margin = max(8, resolution // 30)
        y = margin

        # Draw title
        if title:
            draw.text((margin, y), title, fill=(0, 0, 0), font=font)
            y += max(14, resolution // 18)
            draw.line([(margin, y), (resolution - margin, y)], fill=(180, 180, 180), width=1)
            y += 4

        # Draw table rows
        for line in table_text.split("\n"):
            if y > resolution - margin:
                break
            draw.text((margin, y), line, fill=(0, 0, 0), font=font)
            y += max(12, resolution // 20)

        return np.array(img, dtype=np.uint8)

    def _extract_answer(self, sample: Dict) -> str:
        """Extract ground-truth answer, applying optional regex."""
        raw = str(sample.get(self.config.answer_column, ""))

        if self.config.answer_extract_pattern:
            m = re.search(self.config.answer_extract_pattern, raw, re.DOTALL)
            if m:
                return m.group(1).strip()

        return raw.strip()

    def _build_instruction(self, sample: Dict) -> str:
        """Build env_instruction with question and optional choices."""
        question = str(sample.get(self.config.question_column, ""))

        choices_text = ""
        if self.config.choices_column and self.config.choices_column in sample:
            raw_choices = sample[self.config.choices_column]
            choices_text = self._format_choices(raw_choices)

        instruction = (
            f"{self._base_instruction}\n\n"
            f"Question: {question}"
        )
        if choices_text:
            instruction += f"\n{choices_text}"

        return instruction

    @staticmethod
    def _format_choices(raw_choices) -> str:
        """Format choices from various representations into A/B/C/D labels."""
        if raw_choices is None:
            return ""

        if isinstance(raw_choices, str):
            # Could be JSON string
            import json
            try:
                raw_choices = json.loads(raw_choices)
            except (json.JSONDecodeError, TypeError):
                return f"Choices: {raw_choices}"

        if isinstance(raw_choices, (list, tuple)):
            if len(raw_choices) == 0:
                return ""
            labels = "ABCDEFGHIJ"
            lines = []
            for i, choice in enumerate(raw_choices):
                label = labels[i] if i < len(labels) else str(i)
                lines.append(f"({label}) {choice}")
            return "Choices:\n" + "\n".join(lines)

        return f"Choices: {raw_choices}"

    def get_all_actions(self) -> List[str]:
        """Free-form answers; no discrete action set."""
        return []

    def close(self):
        """Release per-instance state."""
        self._image = None


# ---------------------------------------------------------------------------
# Interactive testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== VLM QA Environment Test ===\n")

    # Test with a mock dataset-like config (no real dataset needed)
    config = VLMQAEnvConfig(
        image_resolution=128,
        max_steps=3,
        correct_reward=1.0,
        wrong_reward=-0.5,
        invalid_reward=-1.0,
    )
    env = VLMQAEnv(config)

    # Manually set state for testing without a real dataset
    env._image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    env._answer = "42"
    env._answer_type = "auto"
    env._step_count = 0

    obs = env.render()
    print(f"Render: shape={obs.shape}, dtype={obs.dtype}")

    # Test correct answer
    obs, reward, term, trunc, info = env.step("<answer>42</answer>")
    print(f"Correct answer: reward={reward}, terminated={term}, info={info['metrics']}")

    # Reset step count for next test
    env._step_count = 0

    # Test wrong answer
    obs, reward, term, trunc, info = env.step("<answer>99</answer>")
    print(f"Wrong answer:   reward={reward}, terminated={term}, info={info['metrics']}")

    # Reset step count for next test
    env._step_count = 0

    # Test invalid format
    obs, reward, term, trunc, info = env.step("no tags here")
    print(f"Invalid format: reward={reward}, terminated={term}, info={info['metrics']}")

    # Test table rendering
    table_img = env._render_table(
        "Name | Age | Score\nAlice | 25 | 90\nBob | 30 | 85",
        "Student Scores",
        256,
    )
    print(f"\nTable render: shape={table_img.shape}, dtype={table_img.dtype}")

    env.close()
    print("\nTest complete.")
