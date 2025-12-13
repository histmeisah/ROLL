"""
Neural-Linear UCB: Contextual Bandits with Deep Representation and Shallow Exploration

Based on the paper: "Neural Contextual Bandits with Deep Representation and Shallow Exploration"
Xu et al., ICLR 2022 (arXiv:2012.01780)

Key idea:
- Deep Representation: Use last hidden layer features φ(x) as context representation
- Shallow Exploration: Apply UCB exploration only in the feature space

Related work:
- "Neural Contextual Bandits with UCB-based Exploration" (Zhou et al., ICML 2020)
  uses gradient-based features instead, which is more expensive computationally.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import deque

from .base_bandit import BaseContextualBandit


class NeuralNetwork(nn.Module):
    """
    Neural network for reward prediction in Neural-Linear UCB.

    Uses a simple MLP architecture with ReLU activations.
    The last hidden layer output is used as feature representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 1,
        activation: str = "relu",
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.feature_dim = hidden_dims[-1]  # Last hidden layer dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the last hidden layer."""
        # Forward through all layers except the last
        for layer in self.network[:-1]:
            x = layer(x)
        return x


class NeuralLinearUCB(BaseContextualBandit):
    """
    Neural-Linear UCB algorithm for contextual bandits.

    Based on "Neural Contextual Bandits with Deep Representation and Shallow Exploration"
    (Xu et al., ICLR 2022).

    Algorithm:
    1. Deep Representation: Extract features φ(x) from the last hidden layer
    2. Shallow Exploration: Compute UCB = μ(x) + α * sqrt(φᵀ A⁻¹ φ)
       where A⁻¹ is updated incrementally using Sherman-Morrison formula

    This achieves Õ(√T) regret with much lower computational cost than
    gradient-based NeuralUCB (Zhou et al., ICML 2020).
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
        Initialize NeuralLinearUCB.

        Args:
            n_arms: Number of arms (prompt templates)
            context_dim: Dimension of context features
            hidden_dims: Hidden layer dimensions for the neural network
            exploration_param: Exploration parameter α (controls UCB width)
                UCB = μ(x) + α * sqrt(φᵀ A⁻¹ φ)
            learning_rate: Learning rate for neural network training
            reg_param: Ridge regression regularization λ for UCB matrix
                A = λI + Σ φ(x)φ(x)ᵀ, ensures A is invertible
                Typical value: 1.0
            l2_weight: L2 regularization weight for neural network training
                loss = MSE + l2_weight * ||θ||²
                Typical value: 0.01 (much smaller than reg_param!)
            buffer_size: Size of experience replay buffer
            batch_size: Batch size for neural network training
            update_freq: Frequency of neural network updates
            device: Device for computation
            seed: Random seed
        """
        super().__init__(n_arms, context_dim, exploration_param, device, seed)

        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.reg_param = reg_param  # λ for UCB matrix: A = λI + Σφφᵀ
        self.l2_weight = l2_weight  # L2 regularization for NN training
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_freq = update_freq

        # Neural network for each arm
        self.networks = []
        self.optimizers = []
        for _ in range(n_arms):
            net = NeuralNetwork(
                input_dim=context_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
            ).to(device)
            self.networks.append(net)
            self.optimizers.append(torch.optim.Adam(net.parameters(), lr=learning_rate))

        # Feature dimension (last hidden layer)
        self.feature_dim = hidden_dims[-1]

        # Inverse covariance matrices for each arm (for UCB computation)
        # Ridge regression: A = λI + Σ φ(x)φ(x)ᵀ
        # Initial: A = λI, so A⁻¹ = I/λ
        # reg_param (λ) ensures A is invertible and controls regularization strength
        # Typical value: λ = 1.0
        self.A_inv = [
            torch.eye(self.feature_dim, device=device) / reg_param
            for _ in range(n_arms)
        ]

        # Experience replay buffers for each arm
        self.buffers = [deque(maxlen=buffer_size) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        """
        Select an arm using Neural-Linear UCB strategy.

        UCB_i = mu_i + alpha * sqrt(phi^T * A_i^{-1} * phi)
        where mu_i is the predicted reward and phi is the neural network features.

        Args:
            context: Context vector of shape (context_dim,)

        Returns:
            Selected arm index
        """
        context_tensor = torch.from_numpy(context).float().to(self.device)
        context_tensor = context_tensor.unsqueeze(0)  # Add batch dimension

        ucb_scores = []

        for arm_idx in range(self.n_arms):
            network = self.networks[arm_idx]

            with torch.no_grad():
                # Get predicted reward
                predicted_reward = network(context_tensor).item()

                # Get neural network features
                features = network.get_features(context_tensor).squeeze()  # (feature_dim,)

                # Compute confidence bound
                # UCB = mu + alpha * sqrt(phi^T * A^{-1} * phi)
                confidence = self.exploration_param * torch.sqrt(
                    torch.matmul(
                        torch.matmul(features.unsqueeze(0), self.A_inv[arm_idx]),
                        features.unsqueeze(1)
                    )
                ).item()

                ucb_score = predicted_reward + confidence
                ucb_scores.append(ucb_score)

        # Select arm with highest UCB
        selected_arm = int(np.argmax(ucb_scores))

        self.t += 1
        self.arm_counts[selected_arm] += 1

        return selected_arm

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """
        Update bandit with observed reward.

        1. Store experience in replay buffer
        2. Update neural network periodically
        3. Update inverse covariance matrix

        Args:
            arm: The arm that was pulled
            context: The context when the arm was pulled
            reward: Observed reward
        """
        # Store experience
        self.buffers[arm].append((context, reward))
        self.rewards_history[arm].append(reward)

        # Convert to tensor
        context_tensor = torch.from_numpy(context).float().to(self.device)
        context_tensor = context_tensor.unsqueeze(0)

        # Update inverse covariance matrix with new features
        with torch.no_grad():
            features = self.networks[arm].get_features(context_tensor).squeeze()
            features = features.unsqueeze(1)  # (feature_dim, 1)

            # Sherman-Morrison formula for incremental inverse update
            # A_inv = A_inv - (A_inv @ x @ x^T @ A_inv) / (1 + x^T @ A_inv @ x)
            numerator = torch.matmul(
                torch.matmul(self.A_inv[arm], features),
                torch.matmul(features.t(), self.A_inv[arm])
            )
            denominator = 1 + torch.matmul(
                torch.matmul(features.t(), self.A_inv[arm]),
                features
            )
            self.A_inv[arm] = self.A_inv[arm] - numerator / denominator

        # Train neural network periodically
        if len(self.buffers[arm]) >= self.batch_size and self.t % self.update_freq == 0:
            self._train_network(arm)

    def _train_network(self, arm: int, n_epochs: int = 10) -> None:
        """
        Train the neural network for a specific arm.

        Args:
            arm: Arm index
            n_epochs: Number of training epochs
        """
        if len(self.buffers[arm]) < self.batch_size:
            return

        network = self.networks[arm]
        optimizer = self.optimizers[arm]

        for _ in range(n_epochs):
            # Sample batch from buffer
            batch_indices = self.rng.choice(
                len(self.buffers[arm]),
                size=min(self.batch_size, len(self.buffers[arm])),
                replace=False
            )

            batch_contexts = []
            batch_rewards = []
            for idx in batch_indices:
                ctx, rew = self.buffers[arm][idx]
                batch_contexts.append(ctx)
                batch_rewards.append(rew)

            # Convert to tensors
            contexts = torch.from_numpy(np.array(batch_contexts)).float().to(self.device)
            rewards = torch.from_numpy(np.array(batch_rewards)).float().to(self.device).unsqueeze(1)

            # Forward pass
            predictions = network(contexts)
            loss = F.mse_loss(predictions, rewards)

            # Add L2 regularization (weight decay)
            # Note: l2_weight is separate from reg_param (UCB ridge regression λ)
            # Typical l2_weight ~ 0.01, while reg_param ~ 1.0
            if self.l2_weight > 0:
                l2_reg = 0
                for param in network.parameters():
                    l2_reg += torch.sum(param ** 2)
                loss = loss + self.l2_weight * l2_reg

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def reset(self) -> None:
        """Reset bandit to initial state."""
        super().reset()

        # Reset networks
        for net, opt in zip(self.networks, self.optimizers):
            net.apply(self._init_weights)
            opt.state = {}

        # Reset inverse covariance matrices
        self.A_inv = [
            torch.eye(self.feature_dim, device=self.device) / self.reg_param
            for _ in range(self.n_arms)
        ]

        # Clear buffers
        for buffer in self.buffers:
            buffer.clear()

    @staticmethod
    def _init_weights(module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_arm_info(self, arm: int) -> Dict:
        """Get detailed information about a specific arm."""
        info = {
            "arm_index": arm,
            "pull_count": int(self.arm_counts[arm]),
            "buffer_size": len(self.buffers[arm]),
            "mean_reward": np.mean(self.rewards_history[arm]) if self.rewards_history[arm] else 0.0,
            "std_reward": np.std(self.rewards_history[arm]) if len(self.rewards_history[arm]) > 1 else 0.0,
        }
        return info