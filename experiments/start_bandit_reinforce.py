"""
Launch script for Bandit-REINFORCE++ training on mathematical reasoning tasks.

Usage:
    python experiments/start_bandit_reinforce.py --config_name bandit_reinforce_config
"""

import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path

# Add ROLL to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from roll.pipeline.bandit.bandit_rl_pipeline import BanditRLPipeline, BanditRLConfig
from roll.utils.logging import setup_logger

logger = logging.getLogger(__name__)


def load_dataset(dataset_name: str, split: str = "train"):
    """
    Load mathematical reasoning dataset.

    Args:
        dataset_name: Name of the dataset (gsm8k, math, etc.)
        split: Data split (train, val, test)

    Returns:
        List of problem dictionaries with 'problem' and 'answer' keys
    """
    # Placeholder - implement actual dataset loading
    # Could use HuggingFace datasets or custom loaders

    if dataset_name == "gsm8k":
        # Example structure
        problems = [
            {
                "problem": "John has 5 apples. He gives 2 to Mary. How many apples does John have left?",
                "answer": "3",
            },
            {
                "problem": "A store sells pencils for $0.50 each. How much do 8 pencils cost?",
                "answer": "4",
            },
            # Add more problems...
        ]
        # In practice, load from actual dataset
        # from datasets import load_dataset
        # dataset = load_dataset("gsm8k", "main")
        # problems = [{"problem": ex["question"], "answer": ex["answer"]} for ex in dataset[split]]

    elif dataset_name == "math":
        problems = [
            {
                "problem": "Solve for x: 2x + 5 = 13",
                "answer": "x = 4",
            },
            {
                "problem": "Find the derivative of f(x) = x^2 + 3x - 1",
                "answer": "f'(x) = 2x + 3",
            },
        ]

    else:
        # Default test problems
        problems = [
            {
                "problem": f"Test problem {i}: What is {i} + {i}?",
                "answer": str(2 * i),
            }
            for i in range(1, 101)
        ]

    logger.info(f"Loaded {len(problems)} problems from {dataset_name} ({split})")
    return problems


def setup_model(config: DictConfig):
    """
    Setup the LLM policy model.

    Args:
        config: Model configuration

    Returns:
        Initialized model
    """
    # Placeholder - implement actual model loading
    # In practice, integrate with ROLL's model loading

    logger.info(f"Loading model: {config.model.name}")

    # Example with transformers
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # model = AutoModelForCausalLM.from_pretrained(
    #     config.model.path,
    #     torch_dtype=getattr(torch, config.model.dtype),
    #     device_map=config.model.device_map,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(config.model.path)

    # For now, return None (will be handled by pipeline)
    return None


@hydra.main(version_base="1.2", config_path="configs", config_name="bandit_reinforce_config")
def main(cfg: DictConfig):
    """Main training function."""

    # Setup logging
    setup_logger(level=logging.INFO)
    logger.info("Starting Bandit-REINFORCE++ training")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed
    torch.manual_seed(cfg.experiment.seed)

    # Create output directory
    output_dir = Path(cfg.logging.output_dir) / cfg.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Initialize Weights & Biases if enabled
    if cfg.logging.wandb.enabled:
        try:
            import wandb
            wandb.init(
                project=cfg.logging.wandb.project,
                entity=cfg.logging.wandb.entity,
                name=cfg.experiment.name,
                config=OmegaConf.to_container(cfg),
                tags=cfg.experiment.tags,
            )
            logger.info("Initialized Weights & Biases logging")
        except ImportError:
            logger.warning("wandb not installed, disabling W&B logging")
            cfg.logging.wandb.enabled = False

    # Load datasets
    train_problems = load_dataset(cfg.training.train_dataset, "train")
    val_problems = load_dataset(cfg.training.train_dataset, "validation")

    # Setup model
    policy_model = setup_model(cfg)

    # Create pipeline configuration
    pipeline_config = BanditRLConfig(
        model_name=cfg.model.name,
        model_path=cfg.model.path,
        n_prompts=cfg.bandit.n_prompts,
        context_dim=cfg.bandit.context_dim,
        hidden_dims=cfg.bandit.hidden_dims,
        exploration_param=cfg.bandit.exploration_param,
        num_trajectories=cfg.reinforce.num_trajectories,
        num_episodes=cfg.training.num_episodes,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.bandit.learning_rate,
        neural_ucb_buffer_size=cfg.bandit.buffer_size,
        neural_ucb_update_freq=cfg.bandit.update_freq,
        neural_ucb_reg_param=cfg.bandit.reg_param,
        log_interval=cfg.training.log_interval,
        checkpoint_interval=cfg.training.checkpoint_interval,
    )

    # Initialize pipeline
    pipeline = BanditRLPipeline(pipeline_config)

    # Override prompt templates if provided
    if hasattr(cfg.prompts, "templates"):
        from roll.algorithms.bandit.bandit_reinforce_plus import PromptTemplate
        pipeline.prompt_templates = [
            PromptTemplate(
                name=template.name,
                template=template.template,
                description=template.get("description", ""),
            )
            for template in cfg.prompts.templates
        ]
        pipeline.bandit_rl.prompt_templates = pipeline.prompt_templates
        pipeline.bandit_rl.n_prompts = len(pipeline.prompt_templates)
        logger.info(f"Loaded {len(pipeline.prompt_templates)} custom prompt templates")

    # Start training
    logger.info("Starting training loop...")
    try:
        pipeline.train(
            train_problems=train_problems,
            val_problems=val_problems,
            policy_model=policy_model,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    # Final evaluation
    logger.info("Running final evaluation...")
    final_stats = pipeline.bandit_rl.get_statistics()
    logger.info(f"Final statistics:\n{final_stats}")

    # Save final checkpoint
    checkpoint_path = output_dir / "final_checkpoint.pt"
    torch.save(
        {
            "config": cfg,
            "pipeline_state": pipeline.bandit_rl.get_statistics(),
            "episode_rewards": pipeline.episode_rewards,
        },
        checkpoint_path,
    )
    logger.info(f"Saved final checkpoint to {checkpoint_path}")

    # Close wandb
    if cfg.logging.wandb.enabled:
        wandb.finish()

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()