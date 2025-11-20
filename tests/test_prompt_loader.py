"""
Test script for prompt loader functionality.

Run this to verify that the prompt loading system works correctly.
"""

import sys
from pathlib import Path

# Add ROLL to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from roll.algorithms.bandit import PromptLoader, load_preset, create_bandit_reinforce_from_preset


def test_prompt_loader():
    """Test basic prompt loader functionality."""
    print("\n" + "=" * 70)
    print("Testing Prompt Loader")
    print("=" * 70)

    # Initialize loader
    loader = PromptLoader()

    # Print summary
    loader.print_summary()

    # Test getting a specific prompt
    print("\n" + "-" * 70)
    print("Test 1: Load specific prompt")
    print("-" * 70)
    prompt = loader.get_prompt("zero_shot_cot_ape")
    if prompt:
        print(f"✓ Loaded: {prompt.name}")
        print(f"  Category: {prompt.category}")
        print(f"  Description: {prompt.description}")
        print(f"  Template: {prompt.template[:100]}...")

        # Test formatting
        problem = "What is 25 + 37?"
        formatted = prompt.format(problem)
        print(f"\n  Formatted with problem:")
        print(f"  {formatted[:150]}...")
    else:
        print("✗ Failed to load prompt")
        return False

    # Test loading a preset
    print("\n" + "-" * 70)
    print("Test 2: Load preset 'diverse_5'")
    print("-" * 70)
    prompts = loader.get_preset_prompts("diverse_5")
    print(f"✓ Loaded {len(prompts)} prompts from preset:")
    for i, p in enumerate(prompts, 1):
        print(f"  {i}. {p.name} ({p.category})")

    # Test loading by category
    print("\n" + "-" * 70)
    print("Test 3: Load prompts by category")
    print("-" * 70)
    cot_prompts = loader.get_prompts_by_category("zero_shot_cot")
    print(f"✓ Found {len(cot_prompts)} zero-shot CoT prompts:")
    for p in cot_prompts:
        print(f"  - {p.name}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70 + "\n")

    return True


def test_bandit_creation():
    """Test creating Bandit-REINFORCE++ with prompt presets."""
    print("\n" + "=" * 70)
    print("Testing Bandit-REINFORCE++ Creation")
    print("=" * 70)

    try:
        # Test 1: Create from preset
        print("\nTest 1: Create from 'diverse_5' preset")
        print("-" * 70)
        bandit_rl = create_bandit_reinforce_from_preset(
            preset_name="diverse_5",
            context_dim=768,
            exploration_param=1.0
        )
        print(f"✓ Created Bandit-REINFORCE++ with {bandit_rl.n_prompts} prompts")
        print(f"  Prompts:")
        for i, prompt in enumerate(bandit_rl.prompt_templates):
            print(f"    {i+1}. {prompt.name}")

        # Test 2: Get statistics
        print("\nTest 2: Get initial statistics")
        print("-" * 70)
        stats = bandit_rl.get_statistics()
        print(f"✓ Statistics retrieved:")
        print(f"  Episode count: {stats['episode_count']}")
        print(f"  Total reward: {stats['total_reward']}")

        # Test 3: Test prompt selection (mock)
        print("\nTest 3: Test prompt selection")
        print("-" * 70)
        import numpy as np
        problem_embedding = np.random.randn(768).astype(np.float32)
        prompt_idx, selected_prompt = bandit_rl.select_prompt(problem_embedding)
        print(f"✓ Selected prompt: {selected_prompt.name} (index {prompt_idx})")

        print("\n" + "=" * 70)
        print("All bandit creation tests passed!")
        print("=" * 70 + "\n")

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_presets():
    """Test loading all available presets."""
    print("\n" + "=" * 70)
    print("Testing All Presets")
    print("=" * 70)

    loader = PromptLoader()
    presets = loader.list_presets()

    for preset_name in presets:
        print(f"\nPreset: {preset_name}")
        print("-" * 70)
        preset_info = loader.get_preset_info(preset_name)
        print(f"  Description: {preset_info['description']}")
        print(f"  Prompts ({preset_info['num_prompts']}):")
        for name in preset_info['prompt_names']:
            print(f"    - {name}")

    print("\n" + "=" * 70)
    print(f"Successfully tested {len(presets)} presets")
    print("=" * 70 + "\n")

    return True


if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print(" " * 15 + "PROMPT LOADER TEST SUITE")
    print("*" * 70)

    # Run all tests
    success = True

    success &= test_prompt_loader()
    success &= test_bandit_creation()
    success &= test_all_presets()

    if success:
        print("\n" + "=" * 70)
        print(" " * 20 + "ALL TESTS PASSED ✓")
        print("=" * 70 + "\n")
    else:
        print("\n" + "=" * 70)
        print(" " * 20 + "SOME TESTS FAILED ✗")
        print("=" * 70 + "\n")
        sys.exit(1)