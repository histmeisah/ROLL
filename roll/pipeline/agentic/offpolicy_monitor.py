"""
Unified Off-Policy Monitoring for ROLL Framework

This module provides centralized off-policy monitoring for replay buffer training,
ensuring consistent metrics calculation across different replay buffer implementations.
"""

import torch
from typing import Dict, Optional, Any
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger

logger = get_logger()


def compute_offpolicy_metrics(
    current_batch: DataProto,
    actor_train_cluster: Any,
    old_prob_mode: str = "trajectory",
    metric_prefix: str = "offpolicy",
    pg_clip: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute off-policy metrics by comparing current policy with behavior policy.

    This function provides a unified way to calculate off-policy metrics for any
    batch sampled from replay buffer, handling both trajectory and turn modes.

    Args:
        current_batch: DataProto batch containing data from replay buffer
        actor_train_cluster: Actor cluster for computing current policy log probs
        old_prob_mode: Mode for old probability calculation ("trajectory" or "turn")
        metric_prefix: Prefix for metric keys (e.g., "offpolicy", "replay")
        pg_clip: Clipping threshold for PPO-style ratio clipping analysis

    Returns:
        Dictionary of off-policy metrics
    """
    metrics = {}

    try:
        # Ensure batch has required fields
        if current_batch is None or current_batch.batch is None:
            logger.warning("compute_offpolicy_metrics: current_batch is None or empty")
            return metrics

        # Check for behavior log probs (stored when data was generated)
        behavior_field = None
        if "behavior_log_probs" in current_batch.batch:
            behavior_field = "behavior_log_probs"
        elif "old_log_probs" in current_batch.batch:
            behavior_field = "old_log_probs"
        else:
            logger.warning("compute_offpolicy_metrics: No behavior log probs found in batch")
            return metrics

        # Check for response mask
        if "response_mask" not in current_batch.batch:
            logger.warning("compute_offpolicy_metrics: No response_mask in batch")
            return metrics

        # Set old_prob_mode in meta_info for compute_log_probs
        current_batch.meta_info["old_prob_mode"] = old_prob_mode

        # Compute current policy log probs
        import ray
        current_lp_refs = actor_train_cluster.compute_log_probs(current_batch, blocking=False)
        current_lp_data = DataProto.materialize_concat(data_refs=current_lp_refs)

        if "log_probs" not in current_lp_data.batch:
            logger.warning("compute_offpolicy_metrics: Failed to compute current log_probs")
            return metrics

        # Extract log probs and masks
        current_log_probs = current_lp_data.batch["log_probs"]
        behavior_log_probs = current_batch.batch[behavior_field]

        # Handle next-token prediction alignment (response_mask is shifted by 1)
        response_mask = current_batch.batch["response_mask"][:, 1:].bool()

        # Ensure shapes match by truncating to minimum length
        min_seq_len = min(
            current_log_probs.shape[1],
            behavior_log_probs.shape[1],
            response_mask.shape[1]
        )

        current_log_probs = current_log_probs[:, :min_seq_len]
        behavior_log_probs = behavior_log_probs[:, :min_seq_len]
        response_mask = response_mask[:, :min_seq_len]

        # Apply mask to get valid positions only
        valid_current = current_log_probs[response_mask]
        valid_behavior = behavior_log_probs[response_mask]

        if valid_current.numel() == 0 or valid_behavior.numel() == 0:
            logger.warning(f"compute_offpolicy_metrics: No valid tokens after masking (current: {valid_current.numel()}, behavior: {valid_behavior.numel()})")
            return metrics

        # Compute off-policy statistics
        log_ratio = valid_current - valid_behavior
        ratio = log_ratio.exp()

        # Basic statistics
        metrics[f"{metric_prefix}/log_ratio/mean"] = log_ratio.mean().detach().item()
        metrics[f"{metric_prefix}/log_ratio/std"] = log_ratio.std().detach().item()
        metrics[f"{metric_prefix}/log_ratio/max"] = log_ratio.max().detach().item()
        metrics[f"{metric_prefix}/log_ratio/min"] = log_ratio.min().detach().item()

        metrics[f"{metric_prefix}/ratio/mean"] = ratio.mean().detach().item()
        metrics[f"{metric_prefix}/ratio/std"] = ratio.std().detach().item()
        metrics[f"{metric_prefix}/ratio/max"] = ratio.max().detach().item()
        metrics[f"{metric_prefix}/ratio/min"] = ratio.min().detach().item()
        metrics[f"{metric_prefix}/ratio/median"] = ratio.median().detach().item()

        # Percentile statistics for distribution analysis
        if ratio.numel() > 0:
            metrics[f"{metric_prefix}/ratio/p95"] = torch.quantile(ratio, 0.95).detach().item()
            metrics[f"{metric_prefix}/ratio/p05"] = torch.quantile(ratio, 0.05).detach().item()
            metrics[f"{metric_prefix}/ratio/p99"] = torch.quantile(ratio, 0.99).detach().item()

        # PPO clipping analysis
        if pg_clip is not None and pg_clip > 0:
            clip_low = 1 - pg_clip
            clip_high = 1 + pg_clip
            clipped = (ratio < clip_low) | (ratio > clip_high)
            clip_frac = clipped.float().mean().detach().item()
            metrics[f"{metric_prefix}/ratio/clip_frac"] = clip_frac
            metrics[f"{metric_prefix}/ratio/clip_threshold"] = pg_clip

        # Effective sample size (ESS) - important for importance sampling
        # ESS = (sum(w))^2 / sum(w^2) where w = ratio
        ess = (ratio.sum() ** 2) / (ratio ** 2).sum()
        ess_ratio = ess / ratio.numel()  # Normalized by batch size
        metrics[f"{metric_prefix}/ess"] = ess.detach().item()
        metrics[f"{metric_prefix}/ess_ratio"] = ess_ratio.detach().item()

        # KL divergence approximation: E[log(p/q)] = E[log(ratio)]
        kl_approx = log_ratio.mean().detach().item()
        metrics[f"{metric_prefix}/kl_divergence"] = kl_approx

        # Count of extreme ratios (potential instability indicators)
        extreme_low = (ratio < 0.5).float().mean().detach().item()
        extreme_high = (ratio > 2.0).float().mean().detach().item()
        metrics[f"{metric_prefix}/ratio/extreme_low_frac"] = extreme_low
        metrics[f"{metric_prefix}/ratio/extreme_high_frac"] = extreme_high

        # Token count for context
        metrics[f"{metric_prefix}/valid_tokens"] = valid_current.numel()
        metrics[f"{metric_prefix}/total_tokens"] = response_mask.numel()
        metrics[f"{metric_prefix}/mask_rate"] = valid_current.numel() / max(response_mask.numel(), 1)

        logger.debug(
            f"Off-policy metrics computed successfully: "
            f"ratio_mean={metrics[f'{metric_prefix}/ratio/mean']:.3f}, "
            f"kl={kl_approx:.3f}, "
            f"ess_ratio={ess_ratio:.3f}"
        )

    except Exception as e:
        logger.error(f"Failed to compute off-policy metrics: {e}")
        logger.debug(f"Error details: {str(e)}", exc_info=True)

        # Return partial metrics with error indicator
        metrics[f"{metric_prefix}/error"] = 1.0
        metrics[f"{metric_prefix}/error_message"] = str(e)[:100]  # Truncate error message

    return metrics


def validate_replay_batch_fields(batch: DataProto) -> Dict[str, bool]:
    """
    Validate that a replay batch contains all necessary fields for off-policy training.

    Args:
        batch: DataProto batch to validate

    Returns:
        Dictionary indicating which fields are present/valid
    """
    validation = {}

    try:
        # Check tensor batch
        if batch.batch is not None:
            validation["has_batch"] = True
            validation["has_input_ids"] = "input_ids" in batch.batch
            validation["has_response_mask"] = "response_mask" in batch.batch
            validation["has_behavior_log_probs"] = "behavior_log_probs" in batch.batch
            validation["has_old_log_probs"] = "old_log_probs" in batch.batch
            validation["has_scores"] = "scores" in batch.batch
            validation["has_penalty"] = "penalty" in batch.batch

            # Check shapes consistency
            if validation["has_input_ids"] and validation["has_response_mask"]:
                input_shape = batch.batch["input_ids"].shape
                mask_shape = batch.batch["response_mask"].shape
                validation["shape_consistent"] = (input_shape[0] == mask_shape[0])
            else:
                validation["shape_consistent"] = False
        else:
            validation["has_batch"] = False

        # Check non-tensor batch
        if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch is not None:
            validation["has_non_tensor_batch"] = True
            validation["has_env_ids"] = "env_ids" in batch.non_tensor_batch
            validation["has_traj_id"] = "traj_id" in batch.non_tensor_batch
        else:
            validation["has_non_tensor_batch"] = False

        # Overall validity
        validation["is_valid"] = (
            validation.get("has_batch", False) and
            validation.get("has_response_mask", False) and
            (validation.get("has_behavior_log_probs", False) or validation.get("has_old_log_probs", False))
        )

    except Exception as e:
        logger.error(f"Failed to validate replay batch: {e}")
        validation["error"] = str(e)
        validation["is_valid"] = False

    return validation


def log_offpolicy_diagnostics(
    metrics: Dict[str, float],
    batch: DataProto,
    global_step: int,
    logger_func=logger.info
):
    """
    Log detailed off-policy diagnostics for debugging.

    Args:
        metrics: Off-policy metrics dictionary
        batch: The batch used for computation
        global_step: Current training step
        logger_func: Logging function to use
    """
    try:
        # Validate batch first
        validation = validate_replay_batch_fields(batch)

        # Prepare diagnostic message
        diag_parts = [
            f"[Step {global_step}] Off-Policy Diagnostics:",
            f"  Batch validation: {validation.get('is_valid', False)}",
        ]

        # Add validation details if not valid
        if not validation.get('is_valid', False):
            diag_parts.append("  Missing fields:")
            for field, present in validation.items():
                if field != "is_valid" and not present:
                    diag_parts.append(f"    - {field}")

        # Add metrics summary
        if metrics:
            diag_parts.append("  Metrics:")
            key_metrics = [
                "ratio/mean", "ratio/max", "ratio/min",
                "kl_divergence", "ess_ratio", "valid_tokens"
            ]
            for key in key_metrics:
                for full_key, value in metrics.items():
                    if key in full_key:
                        diag_parts.append(f"    {full_key}: {value:.4f}")
        else:
            diag_parts.append("  No metrics computed")

        # Log as single message
        logger_func("\n".join(diag_parts))

    except Exception as e:
        logger.error(f"Failed to log off-policy diagnostics: {e}")