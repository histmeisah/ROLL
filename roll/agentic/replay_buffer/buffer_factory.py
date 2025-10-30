"""
Replay Buffer Factory for ROLL Framework

Provides factory functions to create appropriate replay buffers
based on environment manager type and configuration.
"""

import logging
from typing import Union, Dict, Any, Optional
from functools import partial

from .base_buffer import BaseReplayBuffer
from .trajectory_buffer import TrajectoryReplayBuffer
from .step_buffer import StepReplayBuffer
from .priority_functions import PRIORITY_FUNCTIONS, get_priority_function

logger = logging.getLogger(__name__)


def create_replay_buffer(
    manager_type: str,
    capacity: int = 100000,
    batch_size: int = 128,
    seed: int = 42,
    priority_function: str = "uniform",
    priority_exponent: float = 1.0,
    priority_kwargs: Optional[Dict[str, Any]] = None,
    enable_nstep: bool = False,
    n_step: int = 5,
    gamma: float = 0.99,
    age_decay: float = 1000.0,
    use_advantage_priority: bool = False,
    **kwargs
) -> BaseReplayBuffer:
    """
    Factory function to create the appropriate replay buffer type with priority support.

    Args:
        manager_type: Type of environment manager ("trajectory" or "step")
        capacity: Buffer capacity (trajectories for trajectory buffer, steps for step buffer)
        batch_size: Default sampling batch size
        seed: Random seed for reproducibility
        priority_function: Name of priority function ("uniform", "reward", "recency", "combined", etc.)
        priority_exponent: Priority exponent for weighted sampling (alpha in PER)
        priority_kwargs: Additional kwargs for priority function (e.g., alpha for recency)
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        Appropriate replay buffer instance (NumPy-based with priority support)

    Raises:
        ValueError: If manager_type is not supported

    Examples:
        # Default uniform priority
        buffer = create_replay_buffer("trajectory", capacity=10000)

        # Reward-based priority
        buffer = create_replay_buffer(
            "trajectory",
            capacity=10000,
            priority_function="reward",
            priority_exponent=0.6
        )

        # Recency-based priority with custom decay
        buffer = create_replay_buffer(
            "trajectory",
            capacity=10000,
            priority_function="recency",
            priority_kwargs={"alpha": 0.001}
        )

        # Combined priority
        buffer = create_replay_buffer(
            "trajectory",
            capacity=10000,
            priority_function="combined",
            priority_kwargs={"reward_weight": 0.7, "recency_weight": 0.3}
        )
    """
    manager_type = manager_type.lower()

    # Get priority function
    try:
        priority_fn = get_priority_function(priority_function)
    except ValueError as e:
        logger.warning(f"{e}. Falling back to uniform priority.")
        priority_fn = get_priority_function("uniform")

    # Create partial function if kwargs provided
    if priority_kwargs:
        priority_fn = partial(priority_fn, **priority_kwargs)

    # Use NumPy-based implementation (memory-efficient, proven stable)
    if manager_type == "trajectory":
        logger.info(
            f"Creating TrajectoryReplayBuffer: capacity={capacity}, "
            f"priority_fn={priority_function}, priority_exponent={priority_exponent}, "
            f"age_decay={age_decay}, use_advantage_priority={use_advantage_priority}"
        )
        return TrajectoryReplayBuffer(
            capacity=capacity,
            batch_size=batch_size,
            seed=seed,
            priority_fn=priority_fn,
            priority_exponent=priority_exponent,
            priority_kwargs=priority_kwargs or {},
            age_decay=age_decay,
            use_advantage_priority=use_advantage_priority
        )
    elif manager_type == "step":
        logger.info(
            f"Creating StepReplayBuffer: capacity={capacity}, "
            f"priority_fn={priority_function}, priority_exponent={priority_exponent}, "
            f"enable_nstep={enable_nstep}, n_step={n_step}, gamma={gamma}, "
            f"age_decay={age_decay}, use_advantage_priority={use_advantage_priority}"
        )
        return StepReplayBuffer(
            capacity=capacity,
            batch_size=batch_size,
            seed=seed,
            priority_fn=priority_fn,
            priority_exponent=priority_exponent,
            priority_kwargs=priority_kwargs or {},
            enable_nstep=enable_nstep,
            n_step=n_step,
            gamma=gamma,
            age_decay=age_decay,
            use_advantage_priority=use_advantage_priority
        )
    else:
        raise ValueError(
            f"Unsupported manager_type: {manager_type}. "
            f"Supported types: 'trajectory', 'step'"
        )


def detect_manager_type_from_config(pipeline_config) -> str:
    """
    Detect environment manager type from pipeline configuration.

    Args:
        pipeline_config: Pipeline configuration object

    Returns:
        Manager type ("trajectory" or "step")
    """
    try:
        # Primary: direct attribute on train_env_manager
        manager_cls_name = getattr(pipeline_config.train_env_manager, 'env_manager_cls', None)
        if isinstance(manager_cls_name, str) and manager_cls_name:
            if "StepEnvManager" in manager_cls_name:
                logger.debug(f"Detected StepEnvManager from train_env_manager: {manager_cls_name}")
                return "step"
            if "TrajEnvManager" in manager_cls_name or "TrajectoryEnvManager" in manager_cls_name:
                logger.debug(f"Detected TrajEnvManager from train_env_manager: {manager_cls_name}")
                return "trajectory"

        # Fallback: detect from custom_envs by the first tag used in train_env_manager
        tags = getattr(pipeline_config.train_env_manager, 'tags', None)
        if isinstance(tags, (list, tuple)) and len(tags) > 0:
            tag0 = tags[0]
            try:
                env_cfg = pipeline_config.custom_envs[tag0]
                fallback_cls = env_cfg.get('env_manager_cls', '')
                if "StepEnvManager" in fallback_cls:
                    logger.debug(f"Detected StepEnvManager from custom_envs[{tag0}]: {fallback_cls}")
                    return "step"
                if "TrajEnvManager" in fallback_cls or "TrajectoryEnvManager" in fallback_cls:
                    logger.debug(f"Detected TrajEnvManager from custom_envs[{tag0}]: {fallback_cls}")
                    return "trajectory"
            except Exception:
                pass

        # Default
        logger.warning("Failed to detect env_manager type from config, defaulting to 'trajectory'")
        return "trajectory"
    except Exception as e:
        logger.warning(f"Failed to detect env_manager type from config: {e}, defaulting to 'trajectory'")
        return "trajectory"


def get_recommended_capacity(manager_type: str, target_memory_gb: float = 4.0) -> int:
    """
    Get recommended buffer capacity based on manager type and memory constraints.

    Args:
        manager_type: Type of environment manager ("trajectory" or "step")
        target_memory_gb: Target memory usage in GB

    Returns:
        Recommended capacity
    """
    # Rough estimates based on typical data sizes
    if manager_type == "trajectory":
        # Trajectories are larger (complete episodes), typically 1-10KB each
        avg_trajectory_size_mb = 0.005  # 5KB average
        capacity = int((target_memory_gb * 1024) / avg_trajectory_size_mb)
        return min(capacity, 100000)  # Cap at 100K trajectories
    else:  # step
        # Steps are smaller (individual turns), typically 0.5-2KB each
        avg_step_size_mb = 0.001  # 1KB average
        capacity = int((target_memory_gb * 1024) / avg_step_size_mb)
        return min(capacity, 1000000)  # Cap at 1M steps
