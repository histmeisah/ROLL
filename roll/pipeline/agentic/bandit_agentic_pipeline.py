"""
BanditAgenticPipeline: Agentic pipeline with bandit-based prompt selection.

Creates BanditActor and EncoderActor as detached Ray Named Actors
*before* super().__init__() so that environment workers can find them
when they are created during pipeline initialization.
"""

import ray
import numpy as np
from typing import Any, Dict, List, Optional

from roll.algorithms.bandit.bandit_actor import BanditActor
from roll.algorithms.bandit.encoder_actor import EncoderActor, DEFAULT_ENCODER_ACTOR_NAME
from roll.algorithms.bandit.prompt_loader import PromptLoader
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.logging import get_logger

logger = get_logger()

DEFAULT_BANDIT_ACTOR_NAME = "bandit_actor_global"


class BanditAgenticPipeline(AgenticPipeline):
    """
    AgenticPipeline extended with Bandit REINFORCE++ prompt selection.

    Initialization order:
    1. Load prompt templates (PromptLoader)
    2. Create EncoderActor (detached Named Actor)
    3. Wait for encoder to be ready, get context_dim
    4. Create BanditActor (detached Named Actor)
    5. Call super().__init__() which creates environment workers
    """

    def __init__(self, pipeline_config: AgenticConfig):
        bandit_cfg = pipeline_config.bandit_config or {}

        # Load prompt templates
        preset = bandit_cfg.get("preset", "diverse_5")
        prompts_config_path = bandit_cfg.get("prompts_config_path", None)
        loader = PromptLoader(config_path=prompts_config_path)
        self.prompt_templates = loader.get_preset_prompts(preset)
        self.prompt_names = [p.name for p in self.prompt_templates]
        self.prompt_texts = [p.template for p in self.prompt_templates]

        # Strip "{problem}" from templates — they are used as system prompts,
        # not user messages. The problem text is already in the user message
        # via agent_template's {observation}.
        self.prompt_texts = [t.replace("{problem}", "").strip() for t in self.prompt_texts]

        # Add the original system prompt as an additional candidate ("default_system").
        # This lets the bandit choose between custom prompts and the baseline prompt,
        # making the comparison fair — the bandit can always fall back to the default.
        default_system_prompt = None
        for tag, env_cfg in pipeline_config.custom_envs.items():
            if env_cfg.get("env_type") == "roll_math_bandit":
                default_system_prompt = env_cfg.get("agent_system_template", "")
                break
        if default_system_prompt:
            self.prompt_names.append("default_system")
            self.prompt_texts.append(default_system_prompt)
            logger.info(f"[BanditPipeline] Added default system prompt as candidate: "
                        f"'{default_system_prompt[:80]}...'")

        n_prompts = len(self.prompt_texts)
        logger.info(f"[BanditPipeline] {n_prompts} total prompt candidates "
                    f"(preset '{preset}' + default): {self.prompt_names}")

        # Inject prompt_templates into env_config for all custom_envs that use roll_math_bandit
        for tag, env_cfg in pipeline_config.custom_envs.items():
            if env_cfg.get("env_type") == "roll_math_bandit":
                if "env_config" not in env_cfg:
                    env_cfg["env_config"] = {}
                if "bandit_config" not in env_cfg["env_config"]:
                    env_cfg["env_config"]["bandit_config"] = {}
                env_cfg["env_config"]["bandit_config"]["prompt_templates"] = self.prompt_texts

        # Also inject into already-generated env_configs (make_env_configs runs in __post_init__)
        for env_mgr in [pipeline_config.train_env_manager, pipeline_config.val_env_manager]:
            if not hasattr(env_mgr, "env_configs") or not env_mgr.env_configs:
                continue
            for worker_rank, worker_envs in env_mgr.env_configs.items():
                for env_id, entry in worker_envs.items():
                    if entry.get("env_class") == "roll_math_bandit":
                        config = entry.get("config", {})
                        if "bandit_config" not in config:
                            config["bandit_config"] = {}
                        config["bandit_config"]["prompt_templates"] = self.prompt_texts
        logger.info(f"[BanditPipeline] Injected {n_prompts} prompt templates into env_configs")

        # Create EncoderActor
        encoder_model = bandit_cfg.get("encoder_model", "Qwen/Qwen3-VL-Embedding-2B")
        encoder_device = bandit_cfg.get("encoder_device", "auto")
        encoder_type = bandit_cfg.get("encoder_type", "qwen3_vl")
        embedding_dim = bandit_cfg.get("embedding_dim", None)
        encoder_actor_name = bandit_cfg.get("encoder_actor_name", DEFAULT_ENCODER_ACTOR_NAME)

        # Auto-select encoder device: place on the last inference GPU to minimize
        # contention with training. The encoder model (~1.2GB for 0.6B) is small
        # enough to share a GPU with vLLM (which typically reserves 90% = ~72GB
        # on an 80GB H100, leaving ~8GB free).
        if encoder_device == "auto":
            infer_cfg = getattr(pipeline_config, "actor_infer", None)
            if infer_cfg and hasattr(infer_cfg, "device_mapping") and infer_cfg.device_mapping:
                last_infer_gpu = infer_cfg.device_mapping[-1]
                encoder_device = f"cuda:{last_infer_gpu}"
            else:
                encoder_device = "cpu"
            logger.info(f"[BanditPipeline] Auto-selected encoder device: {encoder_device}")

        # When using GPU, we keep num_gpus=0 to avoid consuming Ray's GPU quota
        # (the training pipeline needs all GPUs). Instead, we use runtime_env to
        # inject CUDA_VISIBLE_DEVICES so the actor process can see the GPU.
        # This matches the vLLM worker pattern in ROLL (see third_party/vllm/).
        encoder_actor_options = {
            "name": encoder_actor_name,
            "namespace": RAY_NAMESPACE,
            "lifetime": "detached",
        }
        if encoder_device.startswith("cuda"):
            gpu_idx = encoder_device.split(":")[-1] if ":" in encoder_device else "0"
            encoder_actor_options["runtime_env"] = {
                "env_vars": {
                    "CUDA_VISIBLE_DEVICES": gpu_idx,
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                }
            }
            encoder_device_for_actor = "cuda:0"  # Remapped inside the process
        else:
            encoder_device_for_actor = encoder_device

        self._kill_existing_actor(encoder_actor_name)
        self.encoder_actor = EncoderActor.options(**encoder_actor_options).remote(
            model_name_or_path=encoder_model,
            device=encoder_device_for_actor,
            encoder_type=encoder_type,
            embedding_dim=embedding_dim,
        )
        context_dim = ray.get(self.encoder_actor.get_context_dim.remote())
        logger.info(
            f"[BanditPipeline] EncoderActor created: type={encoder_type}, "
            f"model={encoder_model}, device={encoder_device}, context_dim={context_dim}"
        )

        # Create BanditActor
        hidden_dims = bandit_cfg.get("hidden_dims", [256, 128])
        exploration_param = bandit_cfg.get("exploration_param", 1.0)
        bandit_actor_name = bandit_cfg.get("bandit_actor_name", DEFAULT_BANDIT_ACTOR_NAME)
        bandit_kwargs = {
            "learning_rate": bandit_cfg.get("learning_rate", 1e-3),
            "reg_param": bandit_cfg.get("reg_param", 1.0),
            "l2_weight": bandit_cfg.get("l2_weight", 0.01),
            "buffer_size": bandit_cfg.get("buffer_size", 10000),
            "batch_size": bandit_cfg.get("batch_size", 32),
            "update_freq": bandit_cfg.get("update_freq", 10),
        }

        bandit_algorithm = bandit_cfg.get("bandit_algorithm", "ts")
        warmup_episodes = bandit_cfg.get("warmup_episodes", 500)

        # For cosine algorithm: pre-compute prompt embeddings using the encoder
        prompt_embeddings = None
        if bandit_algorithm == "cosine":
            import pickle as _pickle
            logger.info("[BanditPipeline] Pre-computing prompt embeddings for cosine selection...")
            prompt_embeddings = []
            for text in self.prompt_texts:
                emb_bytes = ray.get(self.encoder_actor.encode.remote({"text": text}))
                prompt_embeddings.append(_pickle.loads(emb_bytes))
            prompt_embeddings = np.stack(prompt_embeddings)  # (n_prompts, context_dim)
            logger.info(f"[BanditPipeline] Prompt embeddings shape: {prompt_embeddings.shape}")

        self._kill_existing_actor(bandit_actor_name)
        self.bandit_actor = BanditActor.options(
            name=bandit_actor_name,
            namespace=RAY_NAMESPACE,
            lifetime="detached",
        ).remote(
            n_prompts=n_prompts,
            context_dim=context_dim,
            hidden_dims=hidden_dims,
            exploration_param=exploration_param,
            bandit_kwargs=bandit_kwargs,
            prompt_names=self.prompt_names,
            enable_monitoring=True,
            device="cpu",
            bandit_algorithm=bandit_algorithm,
            warmup_episodes=warmup_episodes,
            prompt_embeddings=prompt_embeddings,
        )
        logger.info(
            f"[BanditPipeline] BanditActor created: algorithm={bandit_algorithm}, "
            f"n_prompts={n_prompts}, exploration_param={exploration_param}, "
            f"hidden_dims={hidden_dims}"
        )

        # Set bandit log directory (will be overridden by Hydra override if set)
        bandit_log_dir = getattr(pipeline_config, "output_dir", None)
        if bandit_log_dir:
            import os
            bandit_log_path = os.path.join(str(bandit_log_dir), "bandit_logs")
            ray.get(self.bandit_actor.set_log_dir.remote(bandit_log_path))
            logger.info(f"[BanditPipeline] Bandit JSONL logging to {bandit_log_path}")

        # Now safe to call super().__init__() which creates env workers
        super().__init__(pipeline_config)

        # Monkey-patch tracker.log to inject bandit metrics
        self._original_tracker_log = None
        if hasattr(self, 'tracker') and self.tracker is not None:
            self._setup_bandit_logging()

    def _kill_existing_actor(self, actor_name: str) -> None:
        """Kill existing Ray actor if it exists."""
        try:
            existing = ray.get_actor(actor_name, namespace=RAY_NAMESPACE)
            logger.info(f"[BanditPipeline] Killing existing actor: {actor_name}")
            ray.kill(existing)
        except ValueError:
            pass

    def _setup_bandit_logging(self) -> None:
        """Monkey-patch tracker.log to inject bandit metrics."""
        original_log = self.tracker.log

        def patched_log(values: Dict[str, Any], step: Optional[int] = None, **kwargs):
            # Fetch and inject bandit metrics
            try:
                bandit_metrics = ray.get(self.bandit_actor.get_monitor_metrics.remote())
                values.update(bandit_metrics)
            except Exception as e:
                logger.debug(f"[BanditPipeline] Failed to fetch bandit metrics: {e}")
            return original_log(values=values, step=step, **kwargs)

        self._original_tracker_log = original_log
        self.tracker.log = patched_log

    def run(self):
        """Run training with bandit summary at the end."""
        try:
            super().run()
        finally:
            self._print_bandit_summary()

    def _print_bandit_summary(self) -> None:
        """Print bandit performance summary and save to disk."""
        try:
            summary = ray.get(self.bandit_actor.print_summary.remote())
            logger.info(summary)
        except Exception as e:
            logger.warning(f"[BanditPipeline] Failed to print bandit summary: {e}")

        # Save full summary + plotting data to JSON
        try:
            save_path = ray.get(self.bandit_actor.save_summary.remote())
            if save_path:
                logger.info(f"[BanditPipeline] Bandit summary saved to {save_path}")
        except Exception as e:
            logger.warning(f"[BanditPipeline] Failed to save bandit summary: {e}")
