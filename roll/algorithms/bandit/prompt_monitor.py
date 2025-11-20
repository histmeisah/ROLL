"""
Prompt Performance Monitoring for Bandit-REINFORCE++

Tracks and analyzes the performance of different prompt templates.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import time
import json
from pathlib import Path


@dataclass
class PromptStats:
    """Statistics for a single prompt template."""

    prompt_idx: int
    prompt_name: str

    # Counts
    total_selections: int = 0
    total_episodes: int = 0

    # Rewards
    rewards: List[float] = field(default_factory=list)
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0

    # UCB statistics
    ucb_values: List[float] = field(default_factory=list)
    predicted_rewards: List[float] = field(default_factory=list)
    confidence_bounds: List[float] = field(default_factory=list)

    # Temporal statistics (moving averages)
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_mean_reward: float = 0.0

    # Success tracking (for binary rewards)
    successes: int = 0
    success_rate: float = 0.0

    # Time tracking
    first_selected: Optional[float] = None
    last_selected: Optional[float] = None

    def update(self, reward: float, ucb_value: Optional[float] = None,
               predicted_reward: Optional[float] = None,
               confidence: Optional[float] = None):
        """Update statistics with new observation."""
        self.total_selections += 1

        # Update rewards
        self.rewards.append(reward)
        self.recent_rewards.append(reward)
        self.mean_reward = np.mean(self.rewards)
        self.std_reward = np.std(self.rewards) if len(self.rewards) > 1 else 0.0
        self.min_reward = np.min(self.rewards)
        self.max_reward = np.max(self.rewards)
        self.recent_mean_reward = np.mean(self.recent_rewards)

        # Update success tracking (assuming binary rewards)
        if reward > 0.5:  # Threshold for success
            self.successes += 1
        self.success_rate = self.successes / self.total_selections

        # Update UCB statistics
        if ucb_value is not None:
            self.ucb_values.append(ucb_value)
        if predicted_reward is not None:
            self.predicted_rewards.append(predicted_reward)
        if confidence is not None:
            self.confidence_bounds.append(confidence)

        # Update time
        current_time = time.time()
        if self.first_selected is None:
            self.first_selected = current_time
        self.last_selected = current_time

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "prompt_idx": self.prompt_idx,
            "prompt_name": self.prompt_name,
            "total_selections": self.total_selections,
            "mean_reward": float(self.mean_reward),
            "std_reward": float(self.std_reward),
            "min_reward": float(self.min_reward),
            "max_reward": float(self.max_reward),
            "recent_mean_reward": float(self.recent_mean_reward),
            "success_rate": float(self.success_rate),
            "successes": self.successes,
            "ucb_trend": np.mean(self.ucb_values[-10:]) if self.ucb_values else 0.0,
        }


class PromptMonitor:
    """
    Monitor and analyze prompt performance in Bandit-REINFORCE++.

    Features:
    - Track per-prompt statistics
    - Compare prompt performance
    - Identify best/worst performing prompts
    - Detect convergence
    - Export data for visualization
    """

    def __init__(
        self,
        prompt_names: List[str],
        window_size: int = 100,
        save_dir: Optional[Path] = None,
    ):
        """
        Initialize prompt monitor.

        Args:
            prompt_names: List of prompt template names
            window_size: Window size for moving averages
            save_dir: Directory to save monitoring data
        """
        self.prompt_names = prompt_names
        self.n_prompts = len(prompt_names)
        self.window_size = window_size
        self.save_dir = Path(save_dir) if save_dir else None

        # Initialize per-prompt statistics
        self.prompt_stats: Dict[int, PromptStats] = {
            i: PromptStats(prompt_idx=i, prompt_name=name)
            for i, name in enumerate(prompt_names)
        }

        # Global statistics
        self.total_episodes = 0
        self.episode_history = []  # (episode, prompt_idx, reward)
        self.selection_history = []  # Prompt selection over time
        self.reward_history = []  # Global reward over time

        # Performance tracking
        self.best_prompt_idx: Optional[int] = None
        self.best_mean_reward: float = -float('inf')

        # Convergence detection
        self.convergence_threshold = 0.1  # Selection frequency threshold
        self.is_converged = False
        self.converged_prompt: Optional[int] = None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def log_episode(
        self,
        episode: int,
        prompt_idx: int,
        reward: float,
        ucb_value: Optional[float] = None,
        predicted_reward: Optional[float] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Log an episode.

        Args:
            episode: Episode number
            prompt_idx: Selected prompt index
            reward: Observed reward
            ucb_value: UCB value for selected prompt
            predicted_reward: Predicted reward from neural network
            confidence: Confidence bound
            metadata: Additional metadata
        """
        self.total_episodes += 1

        # Update prompt statistics
        self.prompt_stats[prompt_idx].update(
            reward=reward,
            ucb_value=ucb_value,
            predicted_reward=predicted_reward,
            confidence=confidence,
        )

        # Update global history
        self.episode_history.append((episode, prompt_idx, reward))
        self.selection_history.append(prompt_idx)
        self.reward_history.append(reward)

        # Update best prompt
        current_mean = self.prompt_stats[prompt_idx].mean_reward
        if current_mean > self.best_mean_reward and self.prompt_stats[prompt_idx].total_selections >= 10:
            self.best_mean_reward = current_mean
            self.best_prompt_idx = prompt_idx

        # Check convergence
        self._check_convergence()

    def _check_convergence(self):
        """Check if prompt selection has converged."""
        if self.total_episodes < self.window_size:
            return

        # Get recent selections
        recent_selections = self.selection_history[-self.window_size:]
        selection_counts = np.bincount(recent_selections, minlength=self.n_prompts)
        selection_freqs = selection_counts / self.window_size

        # Check if one prompt dominates
        max_freq = np.max(selection_freqs)
        if max_freq > (1 - self.convergence_threshold):
            dominant_prompt = int(np.argmax(selection_freqs))
            if not self.is_converged:
                self.is_converged = True
                self.converged_prompt = dominant_prompt
        else:
            self.is_converged = False
            self.converged_prompt = None

    def get_prompt_rankings(self) -> List[Tuple[int, str, float]]:
        """
        Get prompts ranked by mean reward.

        Returns:
            List of (prompt_idx, prompt_name, mean_reward) tuples, sorted by reward
        """
        rankings = [
            (idx, stats.prompt_name, stats.mean_reward)
            for idx, stats in self.prompt_stats.items()
            if stats.total_selections >= 5  # Minimum samples
        ]
        return sorted(rankings, key=lambda x: x[2], reverse=True)

    def get_selection_distribution(self, recent: bool = False) -> Dict[str, float]:
        """
        Get distribution of prompt selections.

        Args:
            recent: If True, only consider recent window

        Returns:
            Dictionary mapping prompt names to selection frequencies
        """
        if recent and len(self.selection_history) >= self.window_size:
            selections = self.selection_history[-self.window_size:]
        else:
            selections = self.selection_history

        if not selections:
            return {name: 0.0 for name in self.prompt_names}

        counts = np.bincount(selections, minlength=self.n_prompts)
        freqs = counts / len(selections)

        return {
            self.prompt_names[i]: float(freqs[i])
            for i in range(self.n_prompts)
        }

    def get_summary(self) -> Dict:
        """Get comprehensive monitoring summary."""
        rankings = self.get_prompt_rankings()
        selection_dist = self.get_selection_distribution(recent=True)

        summary = {
            "total_episodes": self.total_episodes,
            "n_prompts": self.n_prompts,
            "best_prompt": {
                "idx": self.best_prompt_idx,
                "name": self.prompt_names[self.best_prompt_idx] if self.best_prompt_idx is not None else None,
                "mean_reward": float(self.best_mean_reward),
            },
            "convergence": {
                "is_converged": self.is_converged,
                "converged_prompt": self.converged_prompt,
                "converged_name": self.prompt_names[self.converged_prompt] if self.converged_prompt is not None else None,
            },
            "prompt_rankings": [
                {"rank": i+1, "idx": idx, "name": name, "mean_reward": float(reward)}
                for i, (idx, name, reward) in enumerate(rankings)
            ],
            "selection_distribution": selection_dist,
            "prompt_stats": {
                name: self.prompt_stats[i].get_summary()
                for i, name in enumerate(self.prompt_names)
            },
            "global_stats": {
                "mean_reward": float(np.mean(self.reward_history)) if self.reward_history else 0.0,
                "recent_mean_reward": float(np.mean(self.reward_history[-100:])) if len(self.reward_history) >= 100 else 0.0,
            }
        }

        return summary

    def print_summary(self):
        """Print formatted summary to console."""
        summary = self.get_summary()

        print("\n" + "="*80)
        print("Prompt Performance Monitor Summary")
        print("="*80)
        print(f"Total Episodes: {summary['total_episodes']}")
        print(f"Global Mean Reward: {summary['global_stats']['mean_reward']:.4f}")
        print(f"Recent Mean Reward: {summary['global_stats']['recent_mean_reward']:.4f}")

        print("\n" + "-"*80)
        print("Best Performing Prompt:")
        print("-"*80)
        best = summary['best_prompt']
        print(f"  {best['name']} (idx={best['idx']})")
        print(f"  Mean Reward: {best['mean_reward']:.4f}")

        print("\n" + "-"*80)
        print("Prompt Rankings (by mean reward):")
        print("-"*80)
        for rank_info in summary['prompt_rankings']:
            rank = rank_info['rank']
            name = rank_info['name']
            reward = rank_info['mean_reward']
            idx = rank_info['idx']
            selections = self.prompt_stats[idx].total_selections
            success_rate = self.prompt_stats[idx].success_rate

            print(f"  {rank}. {name:30s} | Reward: {reward:.4f} | Selections: {selections:4d} | Success: {success_rate:.2%}")

        print("\n" + "-"*80)
        print("Recent Selection Distribution:")
        print("-"*80)
        for name, freq in sorted(summary['selection_distribution'].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(freq * 50)
            print(f"  {name:30s} | {freq:6.2%} {bar}")

        print("\n" + "-"*80)
        print("Convergence Status:")
        print("-"*80)
        conv = summary['convergence']
        if conv['is_converged']:
            print(f"  ✓ CONVERGED to: {conv['converged_name']} (idx={conv['converged_prompt']})")
        else:
            print(f"  ✗ Still exploring...")

        print("="*80 + "\n")

    def save_data(self, filepath: Optional[Path] = None):
        """Save monitoring data to JSON file."""
        if filepath is None:
            if self.save_dir is None:
                return
            filepath = self.save_dir / f"monitor_data_episode_{self.total_episodes}.json"

        data = self.get_summary()
        data['episode_history'] = [
            {"episode": ep, "prompt_idx": idx, "reward": float(r)}
            for ep, idx, r in self.episode_history
        ]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def export_for_plotting(self) -> Dict[str, Any]:
        """
        Export data in format suitable for plotting.

        Returns:
            Dictionary with arrays for plotting
        """
        episodes = [ep for ep, _, _ in self.episode_history]

        # Per-prompt reward trajectories
        prompt_rewards = {name: [] for name in self.prompt_names}
        prompt_episodes = {name: [] for name in self.prompt_names}

        for ep, idx, r in self.episode_history:
            name = self.prompt_names[idx]
            prompt_episodes[name].append(ep)
            prompt_rewards[name].append(r)

        # Selection frequency over time
        window = min(100, self.total_episodes // 10) if self.total_episodes > 0 else 1
        selection_freq_timeline = defaultdict(list)

        for i in range(window, len(self.selection_history), window):
            recent = self.selection_history[i-window:i]
            counts = np.bincount(recent, minlength=self.n_prompts)
            freqs = counts / len(recent)
            for idx, name in enumerate(self.prompt_names):
                selection_freq_timeline[name].append(freqs[idx])

        return {
            "episodes": episodes,
            "prompt_rewards": dict(prompt_rewards),
            "prompt_episodes": dict(prompt_episodes),
            "selection_freq_timeline": dict(selection_freq_timeline),
            "global_rewards": self.reward_history,
        }


def get_wandb_metrics(monitor: PromptMonitor) -> Dict[str, Any]:
    """
    Get metrics in format ready for wandb/ROLL logging.

    Args:
        monitor: PromptMonitor instance

    Returns:
        Dictionary of metrics compatible with ROLL's logging system
    """
    summary = monitor.get_summary()

    metrics = {
        # Best prompt metrics
        "monitor/best_prompt_reward": summary['best_prompt']['mean_reward'],
        "monitor/best_prompt_idx": summary['best_prompt']['idx'] or 0,
        "monitor/is_converged": int(summary['convergence']['is_converged']),
        "monitor/global_mean_reward": summary['global_stats']['mean_reward'],
        "monitor/recent_mean_reward": summary['global_stats']['recent_mean_reward'],
    }

    # Per-prompt statistics
    for name, stats in summary['prompt_stats'].items():
        safe_name = name.replace('/', '_').replace(' ', '_')
        metrics[f"prompts/{safe_name}/mean_reward"] = stats['mean_reward']
        metrics[f"prompts/{safe_name}/success_rate"] = stats['success_rate']
        metrics[f"prompts/{safe_name}/total_selections"] = stats['total_selections']

    # Selection distribution
    for name, freq in summary['selection_distribution'].items():
        safe_name = name.replace('/', '_').replace(' ', '_')
        metrics[f"selection_dist/{safe_name}"] = freq

    return metrics


if __name__ == "__main__":
    # Example usage
    prompt_names = [
        "zero_shot_cot_ape",
        "plan_and_solve_plus",
        "solve_and_verify",
        "detailed_explanation",
        "direct_solve",
    ]

    monitor = PromptMonitor(prompt_names)

    # Simulate some episodes
    np.random.seed(42)
    for episode in range(500):
        # Simulate prompt selection (biased towards first prompt)
        prompt_idx = np.random.choice(
            len(prompt_names),
            p=[0.4, 0.25, 0.2, 0.1, 0.05]
        )

        # Simulate reward (different means for different prompts)
        mean_rewards = [0.75, 0.65, 0.60, 0.55, 0.40]
        reward = np.clip(np.random.normal(mean_rewards[prompt_idx], 0.2), 0, 1)

        monitor.log_episode(
            episode=episode,
            prompt_idx=prompt_idx,
            reward=reward,
            ucb_value=np.random.random(),
        )

    # Print summary
    monitor.print_summary()

    # Save data
    monitor.save_data(Path("monitor_test.json"))
