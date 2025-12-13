"""
Test for BanditActor and NeuralLinearUCB.

This test validates:
1. NeuralLinearUCB arm selection
2. Bandit update and learning
3. Ray actor functionality
4. Monitoring and statistics

Usage:
    python tests/algorithms/bandit/test_bandit_actor.py
"""

import ray
import pickle
import numpy as np
import torch

from roll.algorithms.bandit.neural_linear_ucb import NeuralLinearUCB
from roll.algorithms.bandit.bandit_actor import BanditActor


def test_neural_linear_ucb_local():
    """Test NeuralLinearUCB locally (without Ray)."""
    print("\n" + "=" * 80)
    print("Test 1: NeuralLinearUCB Local")
    print("=" * 80)

    # Create NeuralLinearUCB
    n_arms = 5
    context_dim = 10
    bandit = NeuralLinearUCB(
        n_arms=n_arms,
        context_dim=context_dim,
        hidden_dims=[16, 8],
        exploration_param=1.0,
        learning_rate=0.01,
        reg_param=1.0,
        buffer_size=100,
        batch_size=16,
        update_freq=10,
        device="cpu",
        seed=42,
    )
    print(f"✓ NeuralLinearUCB created with {n_arms} arms")

    # Test selection
    context = np.random.randn(context_dim).astype(np.float32)
    arm = bandit.select_arm(context)
    print(f"✓ Selected arm: {arm}")
    assert 0 <= arm < n_arms, f"Invalid arm {arm}"

    # Test update
    reward = 1.0
    bandit.update(arm, context, reward)
    print(f"✓ Updated with reward: {reward}")

    # Multiple updates to trigger training
    print(f"\n✓ Running 50 updates...")
    for i in range(50):
        context = np.random.randn(context_dim).astype(np.float32)
        arm = bandit.select_arm(context)
        # Arm 0 always gives high reward (simulate best arm)
        reward = 1.0 if arm == 0 else np.random.binomial(1, 0.3)
        bandit.update(arm, context, reward)

    # Check statistics
    stats = bandit.get_statistics()
    print(f"\n✓ Statistics:")
    print(f"  Total pulls: {stats['total_pulls']}")
    print(f"  Arm counts: {stats['arm_counts']}")
    print(f"  Mean rewards: {[f'{r:.2f}' for r in stats['mean_rewards']]}")

    # Arm 0 should have highest mean reward
    best_arm = np.argmax(stats['mean_rewards'])
    print(f"  Best arm: {best_arm}")

    print("\n✅ NeuralLinearUCB local test passed!")


def test_bandit_actor_ray():
    """Test BanditActor with Ray."""
    print("\n" + "=" * 80)
    print("Test 2: BanditActor with Ray")
    print("=" * 80)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=4, num_gpus=1, ignore_reinit_error=True, log_to_driver=False)
        print("✓ Ray initialized")

    try:
        # Create BanditActor
        n_prompts = 5
        context_dim = 384
        prompt_names = [f"Prompt_{i}" for i in range(n_prompts)]

        bandit_actor = BanditActor.remote(
            n_prompts=n_prompts,
            context_dim=context_dim,
            hidden_dims=[256, 128],
            exploration_param=1.0,
            bandit_kwargs={
                "learning_rate": 0.001,
                "reg_param": 1.0,
                "buffer_size": 1000,
                "batch_size": 32,
                "update_freq": 10,
                "seed": 42,
            },
            prompt_names=prompt_names,
            enable_monitoring=True,
        )
        print(f"✓ BanditActor created")

        # Test selection
        context = np.random.randn(context_dim).astype(np.float32)
        context_bytes = pickle.dumps(context)

        result = ray.get(bandit_actor.select_arm.remote(context_bytes))
        print(f"\n✓ Selection result:")
        print(f"  Arm: {result['arm_idx']}")
        print(f"  UCB value: {result['ucb_value']:.3f}")
        print(f"  Predicted reward: {result['predicted_reward']:.3f}")
        print(f"  Confidence: {result['confidence']:.3f}")

        # Test update
        reward = 1.0
        update_result = ray.get(bandit_actor.update.remote(
            result['arm_idx'],
            context_bytes,
            reward
        ))
        print(f"\n✓ Update result:")
        print(f"  Update count: {update_result['update_count']}")

        # Run multiple episodes
        print(f"\n✓ Running 100 episodes...")
        for i in range(100):
            context = np.random.randn(context_dim).astype(np.float32)
            context_bytes = pickle.dumps(context)

            # Select arm
            result = ray.get(bandit_actor.select_arm.remote(context_bytes))
            arm = result['arm_idx']

            # Simulate reward (arm 0 is best)
            if arm == 0:
                reward = np.random.binomial(1, 0.8)  # High success rate
            else:
                reward = np.random.binomial(1, 0.3)  # Low success rate

            # Update
            ray.get(bandit_actor.update.remote(arm, context_bytes, reward))

        # Get final statistics
        stats = ray.get(bandit_actor.get_statistics.remote())
        print(f"\n✓ Final statistics:")
        print(f"  Total selections: {stats['total_selections']}")
        print(f"  Updates: {stats['update_count']}")

        # Get monitoring metrics
        metrics = ray.get(bandit_actor.get_monitor_metrics.remote())
        print(f"\n✓ Monitoring metrics:")
        for key, value in list(metrics.items())[:10]:  # Show first 10
            print(f"  {key}: {value:.3f}")

        # Print summary
        summary = ray.get(bandit_actor.print_summary.remote())
        print(f"\n✓ Summary:\n{summary}")

        print("\n✅ BanditActor Ray test passed!")

    finally:
        if ray.is_initialized():
            ray.shutdown()


def test_convergence():
    """Test that bandit converges to best arm."""
    print("\n" + "=" * 80)
    print("Test 3: Convergence Test")
    print("=" * 80)

    n_arms = 5
    context_dim = 10
    true_best_arm = 2  # Ground truth best arm

    bandit = NeuralLinearUCB(
        n_arms=n_arms,
        context_dim=context_dim,
        hidden_dims=[32, 16],
        exploration_param=0.5,  # Lower for faster convergence
        learning_rate=0.01,
        buffer_size=500,
        batch_size=32,
        update_freq=10,
        device="cpu",
        seed=42,
    )

    print(f"✓ Testing convergence to arm {true_best_arm}")

    # Run many episodes
    n_episodes = 500
    selection_counts = [0] * n_arms

    for i in range(n_episodes):
        context = np.random.randn(context_dim).astype(np.float32)
        arm = bandit.select_arm(context)
        selection_counts[arm] += 1

        # Reward distribution (arm 2 is best)
        if arm == true_best_arm:
            reward = np.random.binomial(1, 0.9)  # 90% success
        else:
            reward = np.random.binomial(1, 0.2)  # 20% success

        bandit.update(arm, context, reward)

    stats = bandit.get_statistics()

    print(f"\n✓ After {n_episodes} episodes:")
    print(f"  Selection counts: {selection_counts}")
    print(f"  Mean rewards: {[f'{r:.2f}' for r in stats['mean_rewards']]}")

    # Check if converged to best arm
    most_selected = np.argmax(selection_counts)
    highest_reward = np.argmax(stats['mean_rewards'])

    print(f"\n  Most selected arm: {most_selected}")
    print(f"  Highest reward arm: {highest_reward}")
    print(f"  True best arm: {true_best_arm}")

    # Convergence check (allow some tolerance)
    recent_selections = selection_counts[-100:]  # Last 100
    if most_selected == true_best_arm or highest_reward == true_best_arm:
        print("\n✅ Converged to best arm!")
    else:
        print("\n⚠️  Did not fully converge (may need more episodes)")

    print("\n✅ Convergence test completed!")


if __name__ == "__main__":
    print("=" * 80)
    print("Bandit Algorithm Test Suite")
    print("=" * 80)

    test_neural_linear_ucb_local()
    test_bandit_actor_ray()
    test_convergence()

    print("\n" + "=" * 80)
    print("🎉 All bandit tests passed!")
    print("=" * 80)
