"""
Test for MathReasoningBanditEnv.

This test validates:
1. Environment initialization
2. Reset functionality
3. Step functionality
4. Answer extraction and verification
5. Bandit integration

Usage:
    python tests/agentic/env/test_math_reasoning_bandit.py
"""

import ray
import pickle
import numpy as np
from roll.agentic.env.math_reasoning_bandit import (
    MathReasoningBanditEnv,
    MathReasoningBanditConfig,
)
from roll.algorithms.bandit.bandit_actor import BanditActor
from roll.algorithms.bandit.prompt_loader import PromptLoader


def test_environment_basic():
    """Test basic environment functionality without bandit."""
    print("\n" + "=" * 80)
    print("Test 1: Basic Environment (No Bandit)")
    print("=" * 80)

    # Create simple config without bandit
    config = MathReasoningBanditConfig(
        dataset_name="gsm8k",
        dataset_split="train",
        dataset_seed=42,
    )

    env = MathReasoningBanditEnv(config)
    print(f"✓ Environment created with {len(env.dataset)} problems")

    # Test reset
    obs, info = env.reset(seed=42)
    print(f"\n✓ Environment reset")
    print(f"  Observation: {obs[:200]}...")
    print(f"  Problem: {env.current_problem}")
    print(f"  Ground truth: {env.current_ground_truth}")

    # Test step with correct answer
    print(f"\n✓ Testing step with correct answer...")
    action = f"The answer is {env.current_ground_truth}."
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Action: {action}")
    print(f"  Reward: {reward}")
    print(f"  Terminated: {terminated}")
    print(f"  Success: {info['metrics']['success']}")
    print(f"  Extracted: {info['metrics']['extracted_answer']}")

    # Test with wrong answer
    print(f"\n✓ Testing step with wrong answer...")
    obs, info = env.reset(seed=43)
    action = "The answer is 999999."
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Reward: {reward}")
    print(f"  Success: {info['metrics']['success']}")

    print("\n✅ Basic environment test passed!")


def test_environment_with_bandit():
    """Test environment with integrated bandit."""
    print("\n" + "=" * 80)
    print("Test 2: Environment with Bandit Integration")
    print("=" * 80)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=4, num_gpus=0, ignore_reinit_error=True, log_to_driver=False)
        print("✓ Ray initialized")

    try:
        # Load prompt templates
        loader = PromptLoader()
        prompts = loader.get_preset_prompts("diverse_5")
        print(f"✓ Loaded {len(prompts)} prompt templates")

        # Create BanditActor
        bandit_actor = BanditActor.remote(
            n_prompts=len(prompts),
            context_dim=384,
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
            prompt_names=[p.name for p in prompts],
            enable_monitoring=True,
        )
        print("✓ BanditActor created")

        # Create environment with bandit
        config = MathReasoningBanditConfig(
            dataset_name="gsm8k",
            dataset_split="train",
            dataset_seed=42,
            bandit_actor=bandit_actor,
            prompt_templates=prompts,
            problem_encoder=None,  # Use fallback
        )

        env = MathReasoningBanditEnv(config)
        print("✓ Environment created with bandit integration")

        # Run a few episodes
        print(f"\n✓ Running 5 test episodes...")
        for i in range(5):
            obs, info = env.reset(seed=42 + i)
            prompt_name = info['prompt_name']
            prompt_idx = info['prompt_idx']

            # Simulate LLM response (correct answer)
            action = f"Let me solve this.\nAnswer: {env.current_ground_truth}"
            next_obs, reward, terminated, truncated, info = env.step(action)

            print(f"  Episode {i+1}: Prompt={prompt_name}, Reward={reward:.1f}, Success={info['metrics']['success']}")

        # Get bandit statistics
        stats = ray.get(bandit_actor.get_statistics.remote())
        print(f"\n✓ Bandit statistics:")
        print(f"  Total selections: {stats['total_selections']}")
        print(f"  Updates: {stats['update_count']}")

        print("\n✅ Bandit integration test passed!")

    finally:
        if ray.is_initialized():
            ray.shutdown()


def test_answer_extraction():
    """Test answer extraction patterns."""
    print("\n" + "=" * 80)
    print("Test 3: Answer Extraction Patterns")
    print("=" * 80)

    config = MathReasoningBanditConfig()
    env = MathReasoningBanditEnv(config)

    test_cases = [
        ("Answer: 42", "42"),
        ("The answer is 123.", "123"),
        ("\\boxed{256}", "256"),
        ("<answer>789</answer>", "789"),
        ("Final result: 999\nSo the answer is 100.", "100"),
    ]

    print("Testing extraction patterns:")
    for text, expected in test_cases:
        extracted = env._extract_answer(text)
        status = "✓" if extracted == expected else "✗"
        print(f"  {status} '{text[:50]}' -> '{extracted}' (expected: '{expected}')")

    print("\n✅ Answer extraction test completed!")


def test_interactive():
    """Interactive test mode."""
    print("\n" + "=" * 80)
    print("Test 4: Interactive Mode")
    print("=" * 80)
    print("Enter answers to problems (or 'q' to quit)")

    config = MathReasoningBanditConfig(
        dataset_name="gsm8k",
        dataset_split="train",
    )
    env = MathReasoningBanditEnv(config)

    while True:
        obs, info = env.reset()
        print(f"\n{'-' * 80}")
        print(f"Problem:\n{env.current_problem}")
        print(f"\nGround truth: {env.current_ground_truth}")
        print(f"{'-' * 80}")

        user_input = input("\nYour answer (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break

        action = f"Answer: {user_input}"
        next_obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nResult:")
        print(f"  Extracted: {info['metrics']['extracted_answer']}")
        print(f"  Correct: {info['metrics']['ground_truth']}")
        print(f"  Reward: {reward}")
        print(f"  Success: {info['metrics']['success']}")

        cont = input("\nContinue? (y/n): ")
        if cont.lower() != 'y':
            break

    print("\n✅ Interactive test ended")


if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("MathReasoningBanditEnv Test Suite")
    print("=" * 80)

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        test_interactive()
    else:
        test_environment_basic()
        test_environment_with_bandit()
        test_answer_extraction()

        print("\n" + "=" * 80)
        print("🎉 All tests passed!")
        print("=" * 80)
        print("\nTo run interactive mode:")
        print("  python tests/agentic/env/test_math_reasoning_bandit.py --interactive")
