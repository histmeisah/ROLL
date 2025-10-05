"""
Replay Buffer Factory for ROLL Framework

Provides factory functions to create appropriate replay buffers
based on environment manager type and configuration.
"""

import logging
from typing import Union, Dict, Any

from .base_buffer import BaseReplayBuffer
from .trajectory_buffer import TrajectoryReplayBuffer
from .step_buffer import StepReplayBuffer
from .tensordict_buffer import TensorDictTrajectoryBuffer, TensorDictStepBuffer

logger = logging.getLogger(__name__)


def create_replay_buffer(
    manager_type: str,
    capacity: int = 100000,
    batch_size: int = 128,
    seed: int = 42,
    use_tensordict: bool = True,  # Default to new efficient implementation
    **kwargs
) -> BaseReplayBuffer:
    """
    Factory function to create the appropriate replay buffer type.

    Args:
        manager_type: Type of environment manager ("trajectory" or "step")
        capacity: Buffer capacity (trajectories for trajectory buffer, steps for step buffer)
        batch_size: Default sampling batch size
        seed: Random seed for reproducibility
        use_tensordict: Whether to use TensorDict-based implementation (recommended)
        **kwargs: Additional arguments specific to buffer types

    Returns:
        Appropriate replay buffer instance

    Raises:
        ValueError: If manager_type is not supported
    """
    manager_type = manager_type.lower()

    if use_tensordict:
        # Use new efficient TensorDict-based implementation
        if manager_type == "trajectory":
            logger.info(f"Creating TensorDictTrajectoryBuffer with capacity={capacity}")
            return TensorDictTrajectoryBuffer(
                capacity=capacity,
                batch_size=batch_size,
                seed=seed,
                **kwargs
            )
        elif manager_type == "step":
            logger.info(f"Creating TensorDictStepBuffer with capacity={capacity}")
            return TensorDictStepBuffer(
                capacity=capacity,
                batch_size=batch_size,
                seed=seed,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unsupported manager_type: {manager_type}. "
                f"Supported types: 'trajectory', 'step'"
            )
    else:
        # Use original implementation (for compatibility)
        if manager_type == "trajectory":
            logger.info(f"Creating TrajectoryReplayBuffer with capacity={capacity}")
            return TrajectoryReplayBuffer(
                capacity=capacity,
                batch_size=batch_size,
                seed=seed
            )
        elif manager_type == "step":
            logger.info(f"Creating StepReplayBuffer with capacity={capacity}")
            return StepReplayBuffer(
                capacity=capacity,
                batch_size=batch_size,
                seed=seed
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
