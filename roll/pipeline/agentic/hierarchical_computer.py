"""
Hierarchical RL Advantage Computer

This module implements the core computation logic for hierarchical RL:
1. StepLevelComputer: Computes step-level values and advantages
2. TokenLevelComputer: Computes token-level advantages from intrinsic rewards
3. HierarchicalAdvantageComputer: Orchestrates the two-level computation
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

from .hierarchical_config import HierarchicalRLConfig

logger = logging.getLogger(__name__)


class StepLevelComputer:
    """
    Step-level advantage computation.

    Processes environment rewards to compute step-level values and advantages.
    """

    def __init__(self, config: HierarchicalRLConfig):
        self.config = config

    def extract_step_values(
        self,
        token_values: torch.Tensor,
        response_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract step-level values from token-level values.

        Args:
            token_values: [batch_size, seq_len] token-level values from critic
            response_masks: [batch_size, seq_len] mask for valid tokens

        Returns:
            step_values: [batch_size] step-level values
        """
        batch_size = token_values.size(0)
        step_values = torch.zeros(batch_size, device=token_values.device)

        if self.config.step_value_source == "last_token":
            # Use the last valid token's value as step value
            for i in range(batch_size):
                valid_len = int(response_masks[i].sum())
                if valid_len > 0:
                    step_values[i] = token_values[i, valid_len - 1]

        elif self.config.step_value_source == "mean_tokens":
            # Average over all valid tokens
            masked_values = token_values * response_masks
            valid_counts = response_masks.sum(dim=1).clamp(min=1)
            step_values = masked_values.sum(dim=1) / valid_counts

        elif self.config.step_value_source == "max_tokens":
            # Maximum value among valid tokens
            for i in range(batch_size):
                valid_len = int(response_masks[i].sum())
                if valid_len > 0:
                    step_values[i] = token_values[i, :valid_len].max()

        else:
            raise ValueError(f"Unknown step_value_source: {self.config.step_value_source}")

        if self.config.debug_mode:
            logger.debug(
                f"Extracted step values using {self.config.step_value_source}: "
                f"mean={step_values.mean():.4f}, std={step_values.std():.4f}"
            )

        return step_values

    def compute_step_advantages(
        self,
        env_rewards: torch.Tensor,
        step_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute step-level advantages and returns.

        Args:
            env_rewards: [batch_size] environment rewards for each step
            step_values: [batch_size] step-level values

        Returns:
            advantages: [batch_size] step-level advantages
            returns: [batch_size] step-level returns
        """
        if self.config.step_level_estimator == "gae":
            return self.compute_gae(env_rewards, step_values)
        elif self.config.step_level_estimator == "nstep":
            return self.compute_nstep_returns(env_rewards, step_values)
        elif self.config.step_level_estimator == "monte_carlo":
            return self.compute_monte_carlo(env_rewards, step_values)
        else:
            raise ValueError(f"Unknown step_level_estimator: {self.config.step_level_estimator}")

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation at step-level.

        Args:
            rewards: [batch_size] step rewards
            values: [batch_size] step values

        Returns:
            advantages: [batch_size]
            returns: [batch_size]
        """
        batch_size = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0

        gamma = self.config.step_gamma
        lambda_ = self.config.step_lambda

        # Compute GAE from last to first
        for t in reversed(range(batch_size)):
            if t < batch_size - 1:
                nextvalue = values[t + 1]
            else:
                nextvalue = 0.0  # Terminal state

            # TD error
            delta = rewards[t] + gamma * nextvalue - values[t]

            # GAE accumulation
            lastgaelam = delta + gamma * lambda_ * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values

        if self.config.debug_mode:
            logger.debug(
                f"Step-level GAE computed: "
                f"adv_mean={advantages.mean():.4f}, "
                f"ret_mean={returns.mean():.4f}"
            )

        return advantages, returns

    def compute_nstep_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute N-step returns with bootstrap at step-level.

        Args:
            rewards: [batch_size] step rewards
            values: [batch_size] step values

        Returns:
            advantages: [batch_size]
            returns: [batch_size]
        """
        batch_size = rewards.size(0)
        returns = torch.zeros_like(rewards)

        gamma = self.config.step_gamma
        n_steps = self.config.step_n_steps

        for t in range(batch_size):
            # Compute n-step return starting from t
            n_step_return = 0.0
            discount = 1.0

            # Accumulate discounted rewards
            for k in range(min(n_steps, batch_size - t)):
                n_step_return += discount * rewards[t + k]
                discount *= gamma

            # Add bootstrap value if not terminal
            if self.config.use_step_bootstrap and t + n_steps < batch_size:
                bootstrap_value = values[t + n_steps]
                n_step_return += discount * bootstrap_value

            returns[t] = n_step_return

        advantages = returns - values

        if self.config.debug_mode:
            logger.debug(
                f"Step-level n-step returns computed: "
                f"n_steps={n_steps}, "
                f"ret_mean={returns.mean():.4f}"
            )

        return advantages, returns

    def compute_monte_carlo(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Monte Carlo returns at step-level.

        Args:
            rewards: [batch_size] step rewards
            values: [batch_size] step values

        Returns:
            advantages: [batch_size]
            returns: [batch_size]
        """
        batch_size = rewards.size(0)
        returns = torch.zeros_like(rewards)

        gamma = self.config.step_gamma

        # Compute cumulative returns from last to first
        cumulative_return = 0.0
        for t in reversed(range(batch_size)):
            cumulative_return = rewards[t] + gamma * cumulative_return
            returns[t] = cumulative_return

        advantages = returns - values

        return advantages, returns


class TokenLevelComputer:
    """
    Token-level advantage computation.

    Receives intrinsic rewards from step-level and computes token-level advantages.
    """

    def __init__(self, config: HierarchicalRLConfig):
        self.config = config

    def assign_rewards_to_tokens(
        self,
        step_returns: torch.Tensor,
        response_masks: torch.Tensor,
        token_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Assign step-level returns to tokens as intrinsic rewards.

        This is the key connection between upper and lower levels!

        Args:
            step_returns: [batch_size] step-level returns from upper level
            response_masks: [batch_size, seq_len] valid token masks
            token_values: [batch_size, seq_len] optional, for value_weighted assignment

        Returns:
            token_intrinsic_rewards: [batch_size, seq_len]
        """
        batch_size, seq_len = response_masks.shape
        intrinsic_rewards = torch.zeros(batch_size, seq_len, device=step_returns.device)

        if self.config.reward_assignment == "last_token":
            # Only last token gets the reward
            for i in range(batch_size):
                valid_len = int(response_masks[i].sum())
                if valid_len > 0:
                    intrinsic_rewards[i, valid_len - 1] = step_returns[i]

        elif self.config.reward_assignment == "uniform":
            # Uniform distribution across tokens
            for i in range(batch_size):
                valid_len = int(response_masks[i].sum())
                if valid_len > 0:
                    intrinsic_rewards[i, :valid_len] = step_returns[i] / valid_len

        elif self.config.reward_assignment == "exponential":
            # Exponentially decaying from last to first
            gamma = self.config.token_gamma
            for i in range(batch_size):
                valid_len = int(response_masks[i].sum())
                if valid_len > 0:
                    # Compute weights: [gamma^(n-1), gamma^(n-2), ..., gamma^0]
                    weights = torch.pow(gamma, torch.arange(valid_len - 1, -1, -1, device=step_returns.device))
                    weights = weights / weights.sum()  # Normalize
                    intrinsic_rewards[i, :valid_len] = step_returns[i] * weights

        elif self.config.reward_assignment == "value_weighted":
            # Weight by token value contributions
            if token_values is None:
                raise ValueError("value_weighted assignment requires token_values")

            temp = self.config.assignment_temperature
            for i in range(batch_size):
                valid_len = int(response_masks[i].sum())
                if valid_len > 0:
                    valid_values = token_values[i, :valid_len]
                    weights = torch.softmax(valid_values / temp, dim=0)
                    intrinsic_rewards[i, :valid_len] = step_returns[i] * weights

        else:
            raise ValueError(f"Unknown reward_assignment: {self.config.reward_assignment}")

        if self.config.debug_mode:
            logger.debug(
                f"Assigned rewards using {self.config.reward_assignment}: "
                f"mean={intrinsic_rewards.mean():.4f}, "
                f"nonzero_ratio={(intrinsic_rewards != 0).float().mean():.4f}"
            )

        return intrinsic_rewards

    def compute_token_advantages(
        self,
        intrinsic_rewards: torch.Tensor,
        token_values: torch.Tensor,
        response_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute token-level advantages using intrinsic rewards.

        Args:
            intrinsic_rewards: [batch_size, seq_len] from upper level
            token_values: [batch_size, seq_len] from critic
            response_masks: [batch_size, seq_len] valid token masks

        Returns:
            advantages: [batch_size, seq_len]
            returns: [batch_size, seq_len]
        """
        if self.config.token_level_estimator == "gae":
            return self.compute_gae(intrinsic_rewards, token_values, response_masks)
        elif self.config.token_level_estimator == "reinforce":
            return self.compute_reinforce(intrinsic_rewards, response_masks)
        elif self.config.token_level_estimator == "reinforce_baseline":
            return self.compute_reinforce_baseline(intrinsic_rewards, token_values, response_masks)
        elif self.config.token_level_estimator == "direct":
            return self.compute_direct(intrinsic_rewards, token_values, response_masks)
        else:
            raise ValueError(f"Unknown token_level_estimator: {self.config.token_level_estimator}")

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE at token-level within each step.

        Args:
            rewards: [batch_size, seq_len] intrinsic rewards
            values: [batch_size, seq_len] token values
            mask: [batch_size, seq_len] valid token mask

        Returns:
            advantages: [batch_size, seq_len]
            returns: [batch_size, seq_len]
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)

        gamma = self.config.token_gamma
        lambda_ = self.config.token_lambda

        # Compute GAE for each sample independently
        for i in range(batch_size):
            valid_len = int(mask[i].sum())
            if valid_len == 0:
                continue

            lastgaelam = 0
            for t in reversed(range(valid_len)):
                if t < valid_len - 1:
                    nextvalue = values[i, t + 1]
                else:
                    nextvalue = 0.0  # Terminal

                # TD error
                delta = rewards[i, t] + gamma * nextvalue - values[i, t]

                # GAE accumulation
                lastgaelam = delta + gamma * lambda_ * lastgaelam
                advantages[i, t] = lastgaelam

        # Apply mask
        advantages = advantages * mask
        returns = advantages + values * mask

        return advantages, returns

    def compute_reinforce(
        self,
        rewards: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute REINFORCE (Monte Carlo) returns.

        Args:
            rewards: [batch_size, seq_len] intrinsic rewards
            mask: [batch_size, seq_len] valid token mask

        Returns:
            advantages: [batch_size, seq_len]
            returns: [batch_size, seq_len]
        """
        batch_size, seq_len = rewards.shape
        returns = torch.zeros_like(rewards)

        gamma = self.config.token_gamma

        # Compute Monte Carlo returns for each sample
        for i in range(batch_size):
            valid_len = int(mask[i].sum())
            if valid_len == 0:
                continue

            cumulative_return = 0.0
            for t in reversed(range(valid_len)):
                cumulative_return = rewards[i, t] + gamma * cumulative_return
                returns[i, t] = cumulative_return

        # No baseline, advantages = returns
        advantages = returns * mask

        return advantages, returns

    def compute_reinforce_baseline(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute REINFORCE with value baseline.

        Args:
            rewards: [batch_size, seq_len] intrinsic rewards
            values: [batch_size, seq_len] token values (baseline)
            mask: [batch_size, seq_len] valid token mask

        Returns:
            advantages: [batch_size, seq_len]
            returns: [batch_size, seq_len]
        """
        # First compute MC returns
        _, returns = self.compute_reinforce(rewards, mask)

        # Subtract baseline
        advantages = (returns - values) * mask

        return advantages, returns

    def compute_direct(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Directly use intrinsic rewards as advantages.

        Args:
            rewards: [batch_size, seq_len] intrinsic rewards
            values: [batch_size, seq_len] token values
            mask: [batch_size, seq_len] valid token mask

        Returns:
            advantages: [batch_size, seq_len]
            returns: [batch_size, seq_len]
        """
        advantages = rewards * mask
        returns = (rewards + values) * mask

        return advantages, returns


class HierarchicalAdvantageComputer:
    """
    Main orchestrator for hierarchical RL advantage computation.

    Coordinates step-level and token-level computations.
    """

    def __init__(self, config: HierarchicalRLConfig):
        self.config = config
        self.step_computer = StepLevelComputer(config)
        self.token_computer = TokenLevelComputer(config)

    def compute(
        self,
        env_rewards: torch.Tensor,
        token_values: torch.Tensor,
        response_masks: torch.Tensor,
        original_token_rewards: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Main entry point: Compute hierarchical advantages.

        Args:
            env_rewards: [batch_size] environment rewards for each step
            token_values: [batch_size, seq_len] critic values for tokens
            response_masks: [batch_size, seq_len] valid token masks
            original_token_rewards: [batch_size, seq_len] optional, for mixing

        Returns:
            Dictionary containing:
            - step_values: [batch_size]
            - step_advantages: [batch_size]
            - step_returns: [batch_size]
            - token_advantages: [batch_size, seq_len]
            - token_returns: [batch_size, seq_len]
            - metrics: dict of metrics
        """
        metrics = {}

        # Step 1: Extract step-level values from token values
        step_values = self.step_computer.extract_step_values(
            token_values=token_values,
            response_masks=response_masks
        )

        # Step 2: Compute step-level advantages and returns
        step_advantages, step_returns = self.step_computer.compute_step_advantages(
            env_rewards=env_rewards,
            step_values=step_values
        )

        # Step 3: Assign step returns to tokens as intrinsic rewards
        intrinsic_rewards = self.token_computer.assign_rewards_to_tokens(
            step_returns=step_returns,
            response_masks=response_masks,
            token_values=token_values if self.config.reward_assignment == "value_weighted" else None
        )

        # Step 4: Compute token-level advantages
        token_advantages, token_returns = self.token_computer.compute_token_advantages(
            intrinsic_rewards=intrinsic_rewards,
            token_values=token_values,
            response_masks=response_masks
        )

        # Step 5: Optional mixing with original rewards
        if self.config.use_original_rewards and original_token_rewards is not None:
            alpha = self.config.mixing_alpha
            token_advantages = alpha * token_advantages + (1 - alpha) * original_token_rewards
            metrics["hierarchical/mixing_alpha"] = alpha

        # Step 6: Collect metrics
        if self.config.log_hierarchical_metrics:
            metrics.update(self._compute_metrics(
                step_values=step_values,
                step_advantages=step_advantages,
                step_returns=step_returns,
                token_advantages=token_advantages,
                token_returns=token_returns,
                intrinsic_rewards=intrinsic_rewards,
                env_rewards=env_rewards
            ))

        return {
            "step_values": step_values,
            "step_advantages": step_advantages,
            "step_returns": step_returns,
            "token_advantages": token_advantages,
            "token_returns": token_returns,
            "intrinsic_rewards": intrinsic_rewards,
            "metrics": metrics
        }

    def _compute_metrics(
        self,
        step_values: torch.Tensor,
        step_advantages: torch.Tensor,
        step_returns: torch.Tensor,
        token_advantages: torch.Tensor,
        token_returns: torch.Tensor,
        intrinsic_rewards: torch.Tensor,
        env_rewards: torch.Tensor
    ) -> Dict[str, float]:
        """Compute detailed metrics for monitoring."""
        metrics = {}

        # Step-level metrics
        metrics["hierarchical/step_value/mean"] = step_values.mean().item()
        metrics["hierarchical/step_value/std"] = step_values.std().item()
        metrics["hierarchical/step_value/max"] = step_values.max().item()
        metrics["hierarchical/step_value/min"] = step_values.min().item()

        metrics["hierarchical/step_advantage/mean"] = step_advantages.mean().item()
        metrics["hierarchical/step_advantage/std"] = step_advantages.std().item()

        metrics["hierarchical/step_return/mean"] = step_returns.mean().item()
        metrics["hierarchical/step_return/std"] = step_returns.std().item()

        # Token-level metrics
        metrics["hierarchical/token_advantage/mean"] = token_advantages.mean().item()
        metrics["hierarchical/token_advantage/std"] = token_advantages.std().item()

        metrics["hierarchical/token_return/mean"] = token_returns.mean().item()
        metrics["hierarchical/token_return/std"] = token_returns.std().item()

        # Intrinsic reward metrics
        metrics["hierarchical/intrinsic_reward/mean"] = intrinsic_rewards.mean().item()
        metrics["hierarchical/intrinsic_reward/std"] = intrinsic_rewards.std().item()
        metrics["hierarchical/intrinsic_reward/nonzero_ratio"] = (intrinsic_rewards != 0).float().mean().item()

        # Correlation between step and env rewards
        if len(step_returns) > 1:
            corr = torch.corrcoef(torch.stack([step_returns, env_rewards]))[0, 1]
            metrics["hierarchical/step_env_correlation"] = corr.item() if not torch.isnan(corr) else 0.0

        return metrics
