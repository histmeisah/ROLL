"""
Neural-Linear Thompson Sampling for Contextual Bandits.

Combines two foundational works:
- Agrawal & Goyal, "Thompson Sampling for Contextual Bandits with Linear Payoffs"
  (ICML 2013): First theoretical guarantees for contextual TS, regret Õ(d^{3/2}√T).
- Riquelme, Tucker & Snoek, "Deep Bayesian Bandits Showdown" (ICLR 2018):
  Proposed Neural Linear and empirically showed it is the most reliable method
  among 9+ deep Bayesian approaches for Thompson Sampling.

Algorithm:
1. Deep Representation: Neural network maps context x → features φ(x)
2. Per-arm Bayesian Linear Regression on φ(x):
   - Precision matrix:  A_a = λI + Σ φ(x)φ(x)ᵀ
   - Sufficient statistic: b_a = Σ reward · φ(x)
   - Posterior mean:  μ_a = A_a⁻¹ · b_a
   - Posterior covariance: Σ_a = ν² · A_a⁻¹
3. Thompson Sampling: sample w̃ ~ N(μ_a, Σ_a), pick arm with max w̃ᵀφ(x)

Why Neural Linear + TS over NeuralLinearUCB (Xu et al., ICLR 2022):
- TS is asymptotically optimal (matches Lai-Robbins lower bound); UCB is not
- TS exploration is proportional to P(arm is optimal); UCB over-explores uncertain arms
- Same O(d²) per-step cost, no additional computational overhead
- Empirically most reliable in online decision-making (ICLR 2018 showdown)
"""

import numpy as np
import torch
from typing import Dict, List, Optional

from .neural_linear_ucb import NeuralLinearUCB


class NeuralLinearTS(NeuralLinearUCB):
    """
    Neural-Linear Thompson Sampling.

    Extends NeuralLinearUCB by:
    1. Maintaining per-arm sufficient statistics b_a for posterior mean
    2. Replacing UCB arm selection with Thompson Sampling (posterior sampling)

    The neural network, feature extraction, A_inv updates, replay buffers,
    and periodic network training are all inherited unchanged.
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        hidden_dims: List[int] = [256, 128],
        exploration_param: float = 1.0,
        learning_rate: float = 1e-3,
        reg_param: float = 1.0,
        l2_weight: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 32,
        update_freq: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
    ):
        """
        Initialize NeuralLinearTS.

        Args:
            exploration_param: ν parameter controlling posterior width.
                Σ_a = ν² · A_a⁻¹. Larger ν → more exploration.
                In UCB this was α (additive bonus); here it scales the
                posterior covariance, achieving the same effect probabilistically.
            Other args: identical to NeuralLinearUCB.
        """
        super().__init__(
            n_arms=n_arms,
            context_dim=context_dim,
            hidden_dims=hidden_dims,
            exploration_param=exploration_param,
            learning_rate=learning_rate,
            reg_param=reg_param,
            l2_weight=l2_weight,
            buffer_size=buffer_size,
            batch_size=batch_size,
            update_freq=update_freq,
            device=device,
            seed=seed,
        )

        # Per-arm sufficient statistics for posterior mean: b_a = Σ reward · φ(x)
        # Posterior mean: μ_a = A_inv_a · b_a
        self.b = [
            torch.zeros(self.feature_dim, device=device)
            for _ in range(n_arms)
        ]

    def select_arm(self, context: np.ndarray) -> int:
        """
        Select an arm using Thompson Sampling.

        For each arm a:
          1. Compute features φ(x) from neural network
          2. Compute posterior mean: μ_a = A_a⁻¹ · b_a
          3. Sample weights: w̃_a ~ N(μ_a, ν² · A_a⁻¹)
          4. Compute sampled reward: r̃_a = w̃_aᵀ · φ(x)
        Select arm with highest r̃_a.

        Args:
            context: Context vector of shape (context_dim,)

        Returns:
            Selected arm index
        """
        context_tensor = torch.from_numpy(context).float().to(self.device)
        context_tensor = context_tensor.unsqueeze(0)

        sampled_rewards = []

        for arm_idx in range(self.n_arms):
            network = self.networks[arm_idx]

            with torch.no_grad():
                features = network.get_features(context_tensor).squeeze()  # (feature_dim,)

                # Posterior mean: μ = A⁻¹ · b
                mu = torch.matmul(self.A_inv[arm_idx], self.b[arm_idx])

                # Posterior covariance: Σ = ν² · A⁻¹
                cov = (self.exploration_param ** 2) * self.A_inv[arm_idx]

                # Sample weights from posterior: w̃ ~ N(μ, Σ)
                # Use Cholesky decomposition for numerically stable sampling:
                # w̃ = μ + L · z, where L·Lᵀ = Σ, z ~ N(0, I)
                try:
                    L = torch.linalg.cholesky(cov)
                    z = torch.randn(self.feature_dim, device=self.device)
                    w_sampled = mu + torch.matmul(L, z)
                except torch.linalg.LinAlgError:
                    # Fallback: diagonal approximation if Cholesky fails
                    diag_std = self.exploration_param * torch.sqrt(
                        torch.diag(self.A_inv[arm_idx])
                    )
                    w_sampled = mu + diag_std * torch.randn(
                        self.feature_dim, device=self.device
                    )

                # Sampled reward: r̃ = w̃ᵀ · φ(x)
                sampled_reward = torch.dot(w_sampled, features).item()
                sampled_rewards.append(sampled_reward)

        selected_arm = int(np.argmax(sampled_rewards))

        self.t += 1
        self.arm_counts[selected_arm] += 1

        return selected_arm

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """
        Update bandit with observed reward.

        Extends parent update (which handles A_inv, replay buffer, network training)
        by also updating the sufficient statistic b_a.

        Posterior update:
          b_a ← b_a + reward · φ(x)

        Combined with A_inv update from parent (Sherman-Morrison):
          A_a⁻¹ ← A_a⁻¹ - (A_a⁻¹ φ φᵀ A_a⁻¹) / (1 + φᵀ A_a⁻¹ φ)

        This gives posterior mean μ_a = A_a⁻¹ · b_a at any point.

        Args:
            arm: The arm that was pulled
            context: The context when the arm was pulled
            reward: Observed reward
        """
        # Update b_a BEFORE parent call (parent may retrain network)
        context_tensor = torch.from_numpy(context).float().to(self.device)
        context_tensor = context_tensor.unsqueeze(0)

        with torch.no_grad():
            features = self.networks[arm].get_features(context_tensor).squeeze()
            self.b[arm] = self.b[arm] + reward * features

        # Parent handles: buffer, A_inv (Sherman-Morrison), periodic network training
        super().update(arm, context, reward)

    def reset(self) -> None:
        """Reset bandit to initial state."""
        super().reset()
        self.b = [
            torch.zeros(self.feature_dim, device=self.device)
            for _ in range(self.n_arms)
        ]

    def get_arm_info(self, arm: int) -> Dict:
        """Get detailed information about a specific arm, including posterior stats."""
        info = super().get_arm_info(arm)

        with torch.no_grad():
            mu = torch.matmul(self.A_inv[arm], self.b[arm])
            info["posterior_mean_norm"] = float(torch.norm(mu).item())
            info["posterior_trace"] = float(torch.trace(self.A_inv[arm]).item())

        return info
