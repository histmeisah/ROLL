"""
Centralized Bandit Actor using Ray for distributed prompt selection.

This actor runs as a single Ray actor that all environment instances can call
to select prompts and update statistics in a centralized manner.
"""

import ray
import torch
import pickle
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from .neural_ucb import NeuralUCB
from .prompt_monitor import PromptMonitor

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1, num_gpus=0.1)
class BanditActor:
    """
    Centralized Bandit service using Ray Actor.

    Responsibilities:
    1. Prompt selection via NeuralUCB
    2. Receive reward feedback and update statistics
    3. Periodically train neural networks
    4. Provide global statistics for monitoring

    All environment workers call this single actor, ensuring consistent
    bandit state across distributed rollout collection.
    """

    def __init__(
        self,
        n_prompts: int,
        context_dim: int,
        hidden_dims: List[int],
        exploration_param: float,
        neural_ucb_kwargs: Dict[str, Any],
        prompt_names: List[str],
        enable_monitoring: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize centralized bandit.

        Args:
            n_prompts: Number of prompt templates
            context_dim: Dimension of problem embeddings
            hidden_dims: Neural network hidden dimensions
            exploration_param: UCB exploration parameter (alpha)
            neural_ucb_kwargs: Additional kwargs for NeuralUCB
            prompt_names: Names of prompts (for monitoring)
            enable_monitoring: Enable performance monitoring
            device: Device for computation
        """
        self.n_prompts = n_prompts
        self.context_dim = context_dim
        self.device = device

        # Initialize NeuralUCB
        self.neural_ucb = NeuralUCB(
            n_arms=n_prompts,
            context_dim=context_dim,
            hidden_dims=hidden_dims,
            exploration_param=exploration_param,
            device=device,
            **neural_ucb_kwargs
        )

        # Initialize monitoring
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.monitor = PromptMonitor(
                prompt_names=prompt_names,
                save_dir=None,  # Don't save to disk in actor
            )
        else:
            self.monitor = None

        # Statistics
        self.total_selections = 0
        self.update_counter = 0
        self.update_freq = neural_ucb_kwargs.get("update_freq", 10)

        logger.info(
            f"[BanditActor] Initialized with {n_prompts} prompts, "
            f"context_dim={context_dim}, exploration_param={exploration_param}"
        )

    def select_arm(self, context_bytes: bytes) -> Dict[str, Any]:
        """
        Select a prompt (arm) using NeuralUCB.

        Args:
            context_bytes: Pickled numpy array (context vector)

        Returns:
            Dictionary with:
                - arm_idx: Selected prompt index
                - ucb_value: UCB score
                - predicted_reward: Network prediction
                - confidence: Exploration bonus
        """
        # Deserialize context
        context = pickle.loads(context_bytes)

        # Validate shape
        if context.shape != (self.context_dim,):
            raise ValueError(
                f"Expected context shape ({self.context_dim},), got {context.shape}"
            )

        # Select arm using NeuralUCB
        arm_idx = self.neural_ucb.select_arm(context)

        # Compute UCB components for monitoring
        with torch.no_grad():
            context_tensor = torch.from_numpy(context).float().to(self.device).unsqueeze(0)
            network = self.neural_ucb.networks[arm_idx]
            predicted_reward = network(context_tensor).item()
            features = network.get_features(context_tensor).squeeze()
            confidence = self.neural_ucb.exploration_param * torch.sqrt(
                torch.matmul(
                    torch.matmul(
                        features.unsqueeze(0),
                        self.neural_ucb.A_inv[arm_idx]
                    ),
                    features.unsqueeze(1)
                )
            ).item()
            ucb_value = predicted_reward + confidence

        self.total_selections += 1

        return {
            "arm_idx": arm_idx,
            "ucb_value": ucb_value,
            "predicted_reward": predicted_reward,
            "confidence": confidence,
        }

    def update(self, arm_idx: int, context_bytes: bytes, reward: float) -> Dict[str, Any]:
        """
        Update bandit with observed reward.

        Args:
            arm_idx: Selected arm index
            context_bytes: Pickled context vector
            reward: Observed reward

        Returns:
            Update information (training status, etc.)
        """
        # Deserialize context
        context = pickle.loads(context_bytes)

        # Update NeuralUCB
        self.neural_ucb.update(arm_idx, context, reward)

        # Update monitoring
        if self.enable_monitoring:
            # Recompute UCB info for logging
            with torch.no_grad():
                context_tensor = torch.from_numpy(context).float().to(self.device).unsqueeze(0)
                network = self.neural_ucb.networks[arm_idx]
                predicted_reward = network(context_tensor).item()
                features = network.get_features(context_tensor).squeeze()
                confidence = self.neural_ucb.exploration_param * torch.sqrt(
                    torch.matmul(
                        torch.matmul(
                            features.unsqueeze(0),
                            self.neural_ucb.A_inv[arm_idx]
                        ),
                        features.unsqueeze(1)
                    )
                ).item()
                ucb_value = predicted_reward + confidence

            self.monitor.log_episode(
                episode=self.update_counter,
                prompt_idx=arm_idx,
                reward=reward,
                ucb_value=ucb_value,
                predicted_reward=predicted_reward,
                confidence=confidence,
                metadata={}
            )

        self.update_counter += 1

        # Periodically train networks
        train_info = {}
        if self.update_counter % self.update_freq == 0:
            train_info = self._train_networks()

        return {
            "update_count": self.update_counter,
            "train_triggered": bool(train_info),
            **train_info
        }

    def _train_networks(self) -> Dict[str, Any]:
        """
        Train NeuralUCB networks using experience replay buffers.

        Returns:
            Training statistics
        """
        losses = []

        for arm_idx in range(self.neural_ucb.n_arms):
            buffer = self.neural_ucb.buffers[arm_idx]

            # Skip if not enough data
            if len(buffer) < self.neural_ucb.batch_size:
                continue

            # Sample from buffer
            samples = list(buffer)
            batch_size = min(self.neural_ucb.batch_size, len(samples))
            indices = torch.randperm(len(samples))[:batch_size]
            batch = [samples[i] for i in indices]

            # Prepare batch
            contexts = torch.stack([s[0] for s in batch])
            rewards = torch.tensor([s[1] for s in batch], dtype=torch.float32).to(self.device)

            # Train network
            network = self.neural_ucb.networks[arm_idx]
            optimizer = self.neural_ucb.optimizers[arm_idx]

            optimizer.zero_grad()
            predictions = network(contexts).squeeze()
            loss = torch.nn.functional.mse_loss(predictions, rewards)

            # Add L2 regularization
            l2_loss = sum(p.pow(2.0).sum() for p in network.parameters())
            total_loss = loss + self.neural_ucb.reg_param * l2_loss

            total_loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if losses:
            return {
                "mean_loss": sum(losses) / len(losses),
                "num_arms_trained": len(losses),
            }
        else:
            return {
                "mean_loss": 0.0,
                "num_arms_trained": 0,
            }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get global bandit statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_selections": self.total_selections,
            "update_count": self.update_counter,
            "bandit_stats": self.neural_ucb.get_statistics(),
        }

        if self.enable_monitoring:
            stats["monitor_summary"] = self.monitor.get_summary()

        return stats

    def get_monitor_metrics(self) -> Dict[str, float]:
        """
        Get monitoring metrics for logging (wandb, tensorboard).

        Returns:
            Dictionary of metrics with proper naming for logging
        """
        if not self.enable_monitoring:
            return {}

        summary = self.monitor.get_summary()
        metrics = {}

        # Per-prompt performance
        for prompt_name, prompt_stats in summary["prompt_stats"].items():
            safe_name = prompt_name.replace('/', '_').replace(' ', '_')
            metrics[f"bandit/prompts/{safe_name}/mean_reward"] = prompt_stats["mean_reward"]
            metrics[f"bandit/prompts/{safe_name}/success_rate"] = prompt_stats["success_rate"]
            metrics[f"bandit/prompts/{safe_name}/selections"] = prompt_stats["total_selections"]

        # Selection distribution
        for prompt_name, freq in summary["selection_distribution"].items():
            safe_name = prompt_name.replace('/', '_').replace(' ', '_')
            metrics[f"bandit/selection_dist/{safe_name}"] = freq

        # Convergence status
        metrics["bandit/is_converged"] = 1.0 if summary["convergence"]["is_converged"] else 0.0
        metrics["bandit/total_selections"] = self.total_selections
        metrics["bandit/update_count"] = self.update_counter

        return metrics

    def print_summary(self) -> str:
        """
        Print a human-readable summary.

        Returns:
            Summary string
        """
        stats = self.get_statistics()

        summary = f"\n{'='*60}\n"
        summary += "Bandit Statistics Summary\n"
        summary += f"{'='*60}\n"
        summary += f"Total Selections: {stats['total_selections']}\n"
        summary += f"Total Updates: {stats['update_count']}\n\n"

        if self.enable_monitoring:
            monitor_summary = stats["monitor_summary"]
            summary += "Prompt Performance:\n"
            summary += f"{'-'*60}\n"

            for prompt_name, prompt_stats in monitor_summary["prompt_stats"].items():
                summary += f"{prompt_name}:\n"
                summary += f"  Mean Reward: {prompt_stats['mean_reward']:.3f}\n"
                summary += f"  Success Rate: {prompt_stats['success_rate']:.2%}\n"
                summary += f"  Selections: {prompt_stats['total_selections']}\n\n"

            summary += f"{'-'*60}\n"
            summary += f"Converged: {monitor_summary['convergence']['is_converged']}\n"

        summary += f"{'='*60}\n"

        return summary
