"""
Launch script for Bandit-REINFORCE++ training on mathematical reasoning.

This script:
1. Loads configuration via Hydra
2. Initializes BanditAgenticPipeline
3. Runs training with integrated prompt selection

Usage:
    python experiments/start_bandit_math_reasoning.py

    # With config overrides:
    python experiments/start_bandit_math_reasoning.py \
        rollout_batch_size=256 \
        max_steps=2000 \
        bandit_config.exploration_param=1.5

    # Enable wandb logging:
    python experiments/start_bandit_math_reasoning.py \
        wandb.enabled=true \
        wandb.entity=your-entity
"""

import os
import sys
from pathlib import Path

# Add ROLL to path
roll_dir = Path(__file__).parent.parent
sys.path.insert(0, str(roll_dir))

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from roll.pipeline.agentic.bandit_agentic_pipeline import BanditAgenticPipeline
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.utils.logging import setup_logger, get_logger

logger = get_logger()


@hydra.main(
    version_base="1.2",
    config_path="bandit_math_reasoning",
    config_name="config"
)
def main(cfg: DictConfig):
    """
    Main training function.

    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    setup_logger(level=logging.INFO)

    logger.info("=" * 80)
    logger.info("Bandit-REINFORCE++ Mathematical Reasoning Training")
    logger.info("=" * 80)
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}\n")

    # Convert Hydra config to AgenticConfig
    try:
        agentic_config = AgenticConfig(**OmegaConf.to_container(cfg, resolve=True))
    except Exception as e:
        logger.error(f"Failed to create AgenticConfig: {e}")
        logger.error("Trying with OmegaConf directly...")
        agentic_config = cfg

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Initialize pipeline
    logger.info("\nInitializing BanditAgenticPipeline...")
    try:
        pipeline = BanditAgenticPipeline(pipeline_config=agentic_config)
        logger.info("Pipeline initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        raise

    # Print bandit configuration summary
    if hasattr(pipeline, 'bandit_actor') and pipeline.bandit_actor is not None:
        logger.info("\n" + "=" * 80)
        logger.info("Bandit Configuration Summary")
        logger.info("=" * 80)
        logger.info(f"Number of prompts: {len(pipeline.prompt_templates)}")
        logger.info(f"Prompt names: {[p.name for p in pipeline.prompt_templates]}")
        logger.info(f"Encoder: {cfg.bandit_config.encoder_model}")
        logger.info(f"Context dimension: {cfg.bandit_config.context_dim}")
        logger.info(f"Exploration parameter (alpha): {cfg.bandit_config.exploration_param}")
        logger.info("=" * 80 + "\n")

    # Run training
    logger.info("Starting training...\n")
    try:
        pipeline.run()
        logger.info("\nTraining completed successfully!")
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}", exc_info=True)
        raise

    # Print final bandit summary
    if hasattr(pipeline, 'bandit_actor') and pipeline.bandit_actor is not None:
        logger.info("\n" + "=" * 80)
        logger.info("Final Bandit Statistics")
        logger.info("=" * 80)
        pipeline.print_bandit_summary()

    logger.info("\n" + "=" * 80)
    logger.info("Training session ended")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
