"""
Hierarchical Reinforcement Learning Configuration

This module implements a configurable hierarchical RL system with:
- Upper level (step-level): Processes environment rewards with GAE/n-step returns
- Lower level (token-level): Receives intrinsic rewards from upper level

Only available when using StepEnvManager.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalRLConfig:
    """
    Configuration for Hierarchical Reinforcement Learning.

    This enables a two-level hierarchy:
    1. Step-level (upper): Processes environment rewards
    2. Token-level (lower): Receives intrinsic rewards from step-level
    """

    # Master switch
    enabled: bool = field(
        default=False,
        metadata={
            "help": "Enable hierarchical RL. Only works with StepEnvManager."
        }
    )

    # ============================================
    # Upper Level (Step-level) Configuration
    # ============================================

    step_level_estimator: Literal["gae", "nstep", "monte_carlo"] = field(
        default="gae",
        metadata={
            "help": "Advantage estimator for step-level. "
                    "Options: 'gae' (Generalized Advantage Estimation), "
                    "'nstep' (N-step returns with bootstrap), "
                    "'monte_carlo' (Full episode returns)"
        }
    )

    step_gamma: float = field(
        default=0.99,
        metadata={
            "help": "Discount factor for step-level rewards (environment rewards)"
        }
    )

    step_lambda: float = field(
        default=0.95,
        metadata={
            "help": "Lambda for step-level GAE (only used if step_level_estimator='gae')"
        }
    )

    step_n_steps: int = field(
        default=5,
        metadata={
            "help": "Number of steps for n-step returns (only used if step_level_estimator='nstep')"
        }
    )

    use_step_bootstrap: bool = field(
        default=True,
        metadata={
            "help": "Whether to use bootstrap values in n-step returns"
        }
    )

    step_value_source: Literal["last_token", "mean_tokens", "max_tokens"] = field(
        default="last_token",
        metadata={
            "help": "How to extract step-level value from token values. "
                    "'last_token': Use the last token's value, "
                    "'mean_tokens': Average over all tokens, "
                    "'max_tokens': Maximum value among tokens"
        }
    )

    # ============================================
    # Lower Level (Token-level) Configuration
    # ============================================

    token_level_estimator: Literal["gae", "reinforce", "reinforce_baseline", "direct"] = field(
        default="gae",
        metadata={
            "help": "Advantage estimator for token-level. "
                    "'gae': GAE with intrinsic rewards from upper level, "
                    "'reinforce': Monte Carlo with intrinsic rewards, "
                    "'reinforce_baseline': REINFORCE with value baseline, "
                    "'direct': Directly use upper level advantages"
        }
    )

    token_gamma: float = field(
        default=0.99,
        metadata={
            "help": "Discount factor for token-level (within each step)"
        }
    )

    token_lambda: float = field(
        default=0.95,
        metadata={
            "help": "Lambda for token-level GAE (only used if token_level_estimator='gae')"
        }
    )

    # ============================================
    # Reward Assignment Strategy
    # ============================================

    reward_assignment: Literal["last_token", "uniform", "exponential", "value_weighted"] = field(
        default="last_token",
        metadata={
            "help": "How to assign step-level returns to tokens. "
                    "'last_token': Only the last token gets the reward, "
                    "'uniform': Distribute uniformly across tokens, "
                    "'exponential': Exponentially decaying from last to first, "
                    "'value_weighted': Weight by token value contributions"
        }
    )

    assignment_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for value_weighted assignment (higher = more uniform)"
        }
    )

    # ============================================
    # Mixing and Ablation Options
    # ============================================

    use_original_rewards: bool = field(
        default=False,
        metadata={
            "help": "Also include original token-level rewards (for ablation studies)"
        }
    )

    mixing_alpha: float = field(
        default=1.0,
        metadata={
            "help": "Mixing weight between hierarchical and original rewards. "
                    "1.0 = fully hierarchical, 0.0 = original only"
        }
    )

    # ============================================
    # Logging and Debugging
    # ============================================

    log_hierarchical_metrics: bool = field(
        default=True,
        metadata={
            "help": "Log detailed metrics for hierarchical components"
        }
    )

    debug_mode: bool = field(
        default=False,
        metadata={
            "help": "Enable debug logging for hierarchical computations"
        }
    )


def validate_hierarchical_config(config: HierarchicalRLConfig, pipeline_config=None) -> None:
    """
    Validate hierarchical RL configuration.

    Args:
        config: Hierarchical RL configuration
        pipeline_config: Optional pipeline config for cross-validation

    Raises:
        ValueError: If configuration is invalid
    """
    if not config.enabled:
        return

    # Validate mixing alpha
    if not 0.0 <= config.mixing_alpha <= 1.0:
        raise ValueError(
            f"mixing_alpha must be between 0 and 1, got {config.mixing_alpha}"
        )

    # Validate step-level config
    if config.step_level_estimator == "nstep" and config.step_n_steps <= 0:
        raise ValueError(
            f"step_n_steps must be positive, got {config.step_n_steps}"
        )

    # Validate temperature
    if config.assignment_temperature <= 0:
        raise ValueError(
            f"assignment_temperature must be positive, got {config.assignment_temperature}"
        )

    # Validate gamma values
    if not 0.0 < config.step_gamma <= 1.0:
        raise ValueError(
            f"step_gamma must be in (0, 1], got {config.step_gamma}"
        )

    if not 0.0 < config.token_gamma <= 1.0:
        raise ValueError(
            f"token_gamma must be in (0, 1], got {config.token_gamma}"
        )

    # Validate lambda values
    if not 0.0 <= config.step_lambda <= 1.0:
        raise ValueError(
            f"step_lambda must be in [0, 1], got {config.step_lambda}"
        )

    if not 0.0 <= config.token_lambda <= 1.0:
        raise ValueError(
            f"token_lambda must be in [0, 1], got {config.token_lambda}"
        )

    # Cross-validate with pipeline config if provided
    if pipeline_config is not None:
        # Check if using StepEnvManager
        if hasattr(pipeline_config, 'env_manager_type'):
            if pipeline_config.env_manager_type != 'step':
                raise ValueError(
                    f"Hierarchical RL only works with StepEnvManager, "
                    f"got {pipeline_config.env_manager_type}"
                )

        # Warn if not using replay buffer
        if hasattr(pipeline_config, 'replay'):
            if not pipeline_config.replay.enabled:
                logger.warning(
                    "Hierarchical RL is recommended to use with replay buffer "
                    "for better step-level learning"
                )

    logger.info(
        f"Hierarchical RL config validated: "
        f"step_estimator={config.step_level_estimator}, "
        f"token_estimator={config.token_level_estimator}, "
        f"reward_assignment={config.reward_assignment}"
    )
