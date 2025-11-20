"""
Bandit-Agentic Pipeline: AgenticPipeline with integrated NeuralUCB prompt selection.

This pipeline extends AgenticPipeline to:
1. Initialize a centralized BanditActor for prompt selection
2. Load prompt templates from configuration
3. Initialize problem encoder for context embeddings
4. Inject bandit components into environment configuration
5. Monitor and log bandit statistics
"""

import ray
from typing import Optional
import logging

from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.algorithms.bandit.bandit_actor import BanditActor
from roll.algorithms.bandit.prompt_loader import PromptLoader
from roll.utils.logging import get_logger

logger = get_logger()


class BanditAgenticPipeline(AgenticPipeline):
    """
    Agentic Pipeline with integrated Bandit prompt selection.

    Extends AgenticPipeline to add:
    - Centralized BanditActor (Ray actor)
    - Prompt template loading
    - Problem encoder initialization
    - Automatic injection into environments
    """

    def __init__(self, pipeline_config: AgenticConfig):
        """
        Initialize pipeline.

        Args:
            pipeline_config: Pipeline configuration
        """
        # First call parent init
        super().__init__(pipeline_config)

        # Initialize bandit components if bandit_config is present
        self.bandit_actor = None
        self.prompt_templates = None
        self.problem_encoder = None

        if hasattr(pipeline_config, 'bandit_config') and pipeline_config.bandit_config is not None:
            logger.info("Initializing Bandit components...")
            self._setup_bandit()
        else:
            logger.warning(
                "No bandit_config found in pipeline_config. "
                "Bandit prompt selection will not be enabled."
            )

    def _setup_bandit(self):
        """Setup bandit components."""
        bandit_cfg = self.pipeline_config.bandit_config

        # 1. Load prompt templates
        logger.info("Loading prompt templates...")
        loader = PromptLoader(bandit_cfg.get("prompt_config_path"))
        preset_name = bandit_cfg.get("preset_name", "diverse_5")
        self.prompt_templates = loader.get_preset_prompts(preset_name)
        logger.info(f"Loaded {len(self.prompt_templates)} prompt templates from preset '{preset_name}'")

        # 2. Initialize problem encoder
        logger.info("Initializing problem encoder...")
        encoder_model = bandit_cfg.get("encoder_model", "sentence-transformers/all-MiniLM-L6-v2")

        try:
            from sentence_transformers import SentenceTransformer
            self.problem_encoder = SentenceTransformer(encoder_model)
            logger.info(f"Initialized encoder: {encoder_model}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            logger.warning("Using fallback random encoder")
            self.problem_encoder = None
        except Exception as e:
            logger.warning(f"Failed to load encoder {encoder_model}: {e}")
            logger.warning("Using fallback random encoder")
            self.problem_encoder = None

        # 3. Start centralized BanditActor
        logger.info("Starting centralized BanditActor...")
        context_dim = bandit_cfg.get("context_dim", 384)

        # Adjust context_dim if encoder is available
        if self.problem_encoder is not None:
            # Get actual embedding dimension from encoder
            test_embedding = self.problem_encoder.encode("test")
            actual_dim = test_embedding.shape[0]
            if actual_dim != context_dim:
                logger.warning(
                    f"Encoder dimension ({actual_dim}) differs from config ({context_dim}). "
                    f"Using encoder dimension."
                )
                context_dim = actual_dim

        self.bandit_actor = BanditActor.remote(
            n_prompts=len(self.prompt_templates),
            context_dim=context_dim,
            hidden_dims=bandit_cfg.get("hidden_dims", [256, 128]),
            exploration_param=bandit_cfg.get("exploration_param", 1.0),
            neural_ucb_kwargs={
                "learning_rate": bandit_cfg.get("learning_rate", 0.001),
                "reg_param": bandit_cfg.get("reg_param", 1.0),
                "buffer_size": bandit_cfg.get("buffer_size", 10000),
                "batch_size": bandit_cfg.get("batch_size", 32),
                "update_freq": bandit_cfg.get("update_freq", 10),
                "seed": self.pipeline_config.seed,
            },
            prompt_names=[p.name for p in self.prompt_templates],
            enable_monitoring=bandit_cfg.get("enable_monitoring", True),
        )

        logger.info("BanditActor started successfully")

        # 4. Inject bandit components into environment configuration
        self._inject_bandit_into_envs()

    def _inject_bandit_into_envs(self):
        """
        Inject bandit components into environment configuration.

        This modifies the env_config to include:
        - bandit_actor: Ray actor handle
        - prompt_templates: List of PromptTemplate objects
        - problem_encoder: SentenceTransformer model
        """
        # Find math_reasoning_bandit environment in custom_envs
        if not hasattr(self.pipeline_config, 'custom_envs'):
            logger.warning("No custom_envs in pipeline_config")
            return

        for env_name, env_config in self.pipeline_config.custom_envs.items():
            if 'math_reasoning_bandit' in env_config.get('env_type', '').lower():
                logger.info(f"Injecting bandit components into environment: {env_name}")

                # Get or create env_config dict
                if 'env_config' not in env_config:
                    env_config['env_config'] = {}

                # Inject components
                env_config['env_config']['bandit_actor'] = self.bandit_actor
                env_config['env_config']['prompt_templates'] = self.prompt_templates
                env_config['env_config']['problem_encoder'] = self.problem_encoder

                logger.info(f"Successfully injected bandit components into {env_name}")

    def run(self):
        """
        Run training pipeline with bandit monitoring.

        Extends parent run() to log bandit metrics.
        """
        logger.info("Starting Bandit-Agentic Pipeline training...")

        if self.bandit_actor is not None:
            logger.info("Bandit prompt selection is ENABLED")
        else:
            logger.info("Bandit prompt selection is DISABLED")

        # Call parent run
        super().run()

    def log_metrics_impl(self, metrics: dict, global_step: int):
        """
        Log metrics including bandit statistics.

        Args:
            metrics: Metrics dictionary
            global_step: Current training step
        """
        # Call parent log method
        super().log_metrics_impl(metrics, global_step)

        # Add bandit metrics if available
        if self.bandit_actor is not None:
            try:
                bandit_metrics = ray.get(self.bandit_actor.get_monitor_metrics.remote())
                metrics.update(bandit_metrics)
            except Exception as e:
                logger.warning(f"Failed to get bandit metrics: {e}")

    def save_checkpoint_impl(self, global_step: int, checkpoint_dir: str):
        """
        Save checkpoint including bandit state.

        Args:
            global_step: Current training step
            checkpoint_dir: Directory to save checkpoint
        """
        # Call parent save method
        super().save_checkpoint_impl(global_step, checkpoint_dir)

        # Save bandit statistics
        if self.bandit_actor is not None:
            try:
                import torch
                import os

                bandit_stats = ray.get(self.bandit_actor.get_statistics.remote())

                checkpoint_path = os.path.join(checkpoint_dir, f"bandit_state_step{global_step}.pt")
                torch.save({
                    "global_step": global_step,
                    "bandit_stats": bandit_stats,
                }, checkpoint_path)

                logger.info(f"Saved bandit state to {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to save bandit state: {e}")

    def print_bandit_summary(self):
        """Print bandit statistics summary."""
        if self.bandit_actor is not None:
            try:
                summary = ray.get(self.bandit_actor.print_summary.remote())
                print(summary)
            except Exception as e:
                logger.warning(f"Failed to print bandit summary: {e}")

    def __del__(self):
        """Cleanup: print final bandit summary."""
        if self.bandit_actor is not None:
            logger.info("Printing final bandit summary:")
            self.print_bandit_summary()
