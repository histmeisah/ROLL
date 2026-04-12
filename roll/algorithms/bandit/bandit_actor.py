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
import json
import os
from pathlib import Path

from .neural_linear_ucb import NeuralLinearUCB
from .neural_linear_ts import NeuralLinearTS
from .cosine_similarity_bandit import CosineSimilarityBandit
from .prompt_monitor import PromptMonitor

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1, num_gpus=0)
class BanditActor:
    """
    Centralized Bandit service using Ray Actor.

    Responsibilities:
    1. Prompt selection via NeuralLinearUCB
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
        bandit_kwargs: Dict[str, Any],
        prompt_names: List[str],
        enable_monitoring: bool = True,
        device: str = "cpu",
        bandit_algorithm: str = "ts",  # "ts", "ucb", or "cosine"
        warmup_episodes: int = 500,
        prompt_embeddings: Optional[np.ndarray] = None,
    ):
        """
        Initialize centralized bandit.

        Args:
            n_prompts: Number of prompt templates
            context_dim: Dimension of problem embeddings
            hidden_dims: Neural network hidden dimensions
            exploration_param: Exploration parameter.
                UCB: α in UCB = μ + α√(φᵀA⁻¹φ)
                TS:  ν in Σ = ν²·A⁻¹ (posterior width)
            bandit_kwargs: Additional kwargs for bandit algorithm
            prompt_names: Names of prompts (for monitoring)
            enable_monitoring: Enable performance monitoring
            device: Device for computation
            bandit_algorithm: Algorithm variant:
                - "ts": Thompson Sampling (recommended, Riquelme et al. ICLR 2018)
                - "ucb": Upper Confidence Bound (Xu et al. ICLR 2022)
                - "cosine": Cosine similarity baseline (no learning, ablation)
            warmup_episodes: Number of initial episodes using uniform random
                arm selection. Ensures each arm gets sufficient data before
                the bandit algorithm takes over. Set to 0 to disable.
                Ignored for cosine algorithm (no learning needed).
            prompt_embeddings: Pre-computed prompt embeddings for cosine
                algorithm, shape (n_prompts, context_dim). Required when
                bandit_algorithm="cosine".
        """
        self.n_prompts = n_prompts
        self.context_dim = context_dim
        self.device = device
        self.prompt_names = prompt_names
        self.bandit_algorithm = bandit_algorithm

        # Select bandit algorithm
        if bandit_algorithm == "cosine":
            if prompt_embeddings is None:
                raise ValueError("prompt_embeddings required for cosine algorithm")
            self.bandit = CosineSimilarityBandit(
                n_arms=n_prompts,
                context_dim=context_dim,
                prompt_embeddings=prompt_embeddings,
                device=device,
            )
        else:
            bandit_cls = NeuralLinearTS if bandit_algorithm == "ts" else NeuralLinearUCB
            self.bandit = bandit_cls(
                n_arms=n_prompts,
                context_dim=context_dim,
                hidden_dims=hidden_dims,
                exploration_param=exploration_param,
                device=device,
                **bandit_kwargs
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
        self.warmup_episodes = warmup_episodes

        # JSONL episode logging
        self.log_dir = None
        self._jsonl_file = None

        logger.info(
            f"[BanditActor] Initialized with {n_prompts} prompts, "
            f"algorithm={bandit_algorithm}, context_dim={context_dim}, "
            f"exploration_param={exploration_param}, "
            f"warmup_episodes={warmup_episodes}"
        )

    def set_log_dir(self, log_dir: str) -> None:
        """Set output directory for JSONL episode logs and final summary."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = self.log_dir / "bandit_episodes.jsonl"
        self._jsonl_file = open(jsonl_path, "a")
        logger.info(f"[BanditActor] JSONL logging to {jsonl_path}")

    def _log_episode_jsonl(self, arm_idx: int, reward: float,
                           ucb_value: float, predicted_reward: float,
                           confidence: float, problem_text: str = "") -> None:
        """Write one episode record to JSONL file."""
        if self._jsonl_file is None:
            return
        record = {
            "step": self.update_counter,
            "arm_idx": arm_idx,
            "prompt_name": self.prompt_names[arm_idx] if arm_idx < len(self.prompt_names) else str(arm_idx),
            "reward": reward,
            "ucb_value": ucb_value,
            "predicted_reward": predicted_reward,
            "confidence": confidence,
            "problem_text": problem_text[:200] if problem_text else "",
        }
        self._jsonl_file.write(json.dumps(record, ensure_ascii=False) + chr(10))
        if self.update_counter % 50 == 0:
            self._jsonl_file.flush()

    def save_summary(self) -> str:
        """Save full monitoring summary and plotting data to JSON. Returns file path."""
        if self.log_dir is None:
            return ""
        # Flush JSONL
        if self._jsonl_file:
            self._jsonl_file.flush()
        # Save summary
        summary_path = self.log_dir / "bandit_summary.json"
        data = {}
        if self.enable_monitoring:
            data["summary"] = self.monitor.get_summary()
            data["plotting_data"] = self.monitor.export_for_plotting()
        data["total_selections"] = self.total_selections
        data["update_count"] = self.update_counter
        data["bandit_algorithm"] = self.bandit_algorithm
        data["prompt_names"] = self.prompt_names
        with open(summary_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"[BanditActor] Summary saved to {summary_path}")
        return str(summary_path)

    def select_arm(self, context_bytes: bytes) -> Dict[str, Any]:
        """
        Select a prompt (arm) using NeuralLinearUCB.

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

        # During warmup: uniform random selection to gather initial data for all arms.
        # After warmup: use the bandit algorithm (TS or UCB).
        if self.total_selections < self.warmup_episodes:
            arm_idx = int(self.total_selections % self.n_prompts)
        else:
            arm_idx = self.bandit.select_arm(context)

        # Compute monitoring metrics (algorithm-dependent)
        if self.bandit_algorithm == "cosine":
            similarities = self.bandit.get_similarities(context)
            predicted_reward = float(similarities[arm_idx])
            confidence = 0.0
            ucb_value = predicted_reward
        else:
            with torch.no_grad():
                context_tensor = torch.from_numpy(context).float().to(self.device).unsqueeze(0)
                network = self.bandit.networks[arm_idx]
                predicted_reward = network(context_tensor).item()
                features = network.get_features(context_tensor).squeeze()
                confidence = self.bandit.exploration_param * torch.sqrt(
                    torch.matmul(
                        torch.matmul(
                            features.unsqueeze(0),
                            self.bandit.A_inv[arm_idx]
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

    def update(self, arm_idx: int, context_bytes: bytes, reward: float,
               problem_text: str = "") -> Dict[str, Any]:
        """
        Update bandit with observed reward.

        Args:
            arm_idx: Selected arm index
            context_bytes: Pickled context vector
            reward: Observed reward
            problem_text: Problem text for JSONL logging

        Returns:
            Update information (training status, etc.)
        """
        # Deserialize context
        context = pickle.loads(context_bytes)

        # Update bandit
        self.bandit.update(arm_idx, context, reward)

        # Update monitoring
        if self.enable_monitoring:
            if self.bandit_algorithm == "cosine":
                similarities = self.bandit.get_similarities(context)
                predicted_reward = float(similarities[arm_idx])
                confidence = 0.0
                ucb_value = predicted_reward
            else:
                with torch.no_grad():
                    context_tensor = torch.from_numpy(context).float().to(self.device).unsqueeze(0)
                    network = self.bandit.networks[arm_idx]
                    predicted_reward = network(context_tensor).item()
                    features = network.get_features(context_tensor).squeeze()
                    confidence = self.bandit.exploration_param * torch.sqrt(
                        torch.matmul(
                            torch.matmul(
                                features.unsqueeze(0),
                                self.bandit.A_inv[arm_idx]
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

            # Log to JSONL
            self._log_episode_jsonl(
                arm_idx=arm_idx, reward=reward,
                ucb_value=ucb_value, predicted_reward=predicted_reward,
                confidence=confidence, problem_text=problem_text,
            )

        self.update_counter += 1

        # Network training is handled internally by NeuralLinearUCB.update()
        # (called via self.bandit.update above) every update_freq steps.
        # No additional training needed here.

        return {
            "update_count": self.update_counter,
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
            "bandit_stats": self.bandit.get_statistics(),
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
        selection_dist = summary["selection_distribution"]
        for prompt_name, freq in selection_dist.items():
            safe_name = prompt_name.replace('/', '_').replace(' ', '_')
            metrics[f"bandit/selection_dist/{safe_name}"] = freq

        # Top selected prompt (most frequently chosen)
        if selection_dist:
            top_prompt_name = max(selection_dist, key=selection_dist.get)
            top_prompt_idx = self.prompt_names.index(top_prompt_name) if top_prompt_name in self.prompt_names else -1
            metrics["bandit/top_prompt_idx"] = float(top_prompt_idx)
            metrics["bandit/top_prompt_ratio"] = selection_dist[top_prompt_name]

        # Best performing prompt (highest mean reward, with >= 5 selections)
        best_reward = -1.0
        best_idx = -1
        for prompt_name, prompt_stats in summary["prompt_stats"].items():
            if prompt_stats["total_selections"] >= 5 and prompt_stats["mean_reward"] > best_reward:
                best_reward = prompt_stats["mean_reward"]
                best_idx = self.prompt_names.index(prompt_name) if prompt_name in self.prompt_names else -1
        metrics["bandit/best_reward_prompt_idx"] = float(best_idx)
        metrics["bandit/best_reward"] = best_reward

        # Global stats
        metrics["bandit/total_selections"] = self.total_selections
        metrics["bandit/update_count"] = self.update_counter

        # Enhanced metrics: entropy, exploration, reward gap, per-prompt curves
        global_stats = summary.get("global_stats", {})
        metrics["bandit/selection_entropy"] = global_stats.get("selection_entropy", 0.0)
        metrics["bandit/max_entropy"] = global_stats.get("max_entropy", 0.0)
        metrics["bandit/exploration_ratio"] = global_stats.get("exploration_ratio", 0.0)

        # Per-prompt enhanced: recent_reward_50, ucb_value, confidence, cumulative_selections
        for prompt_name, prompt_stats in summary["prompt_stats"].items():
            safe_name = prompt_name.replace("/", "_").replace(" ", "_")
            metrics[f"bandit/prompts/{safe_name}/recent_reward_50"] = prompt_stats.get("recent_reward_50", 0.0)
            metrics[f"bandit/prompts/{safe_name}/ucb_value"] = prompt_stats.get("latest_ucb", 0.0)
            metrics[f"bandit/prompts/{safe_name}/confidence"] = prompt_stats.get("latest_confidence", 0.0)
            metrics[f"bandit/prompts/{safe_name}/predicted_reward"] = prompt_stats.get("latest_predicted", 0.0)
            metrics[f"bandit/prompts/{safe_name}/cumulative_selections"] = float(prompt_stats.get("total_selections", 0))

        # Reward gap between best and worst prompt
        all_mean_rewards = [
            s["mean_reward"] for s in summary["prompt_stats"].values()
            if s["total_selections"] >= 5
        ]
        if len(all_mean_rewards) >= 2:
            metrics["bandit/reward_gap"] = max(all_mean_rewards) - min(all_mean_rewards)
        else:
            metrics["bandit/reward_gap"] = 0.0

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
