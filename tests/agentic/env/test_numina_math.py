import sys
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from roll.agentic.env.numina_math import NuminaMathEnv, NuminaMathEnvConfig


def test_basic_math():
    """Basic math functionality test"""
    print("Testing basic math functionality...")

    config = NuminaMathEnvConfig(
        dataset_path="/mnt/chensiheng/ai_researcher_data/numinamath_new/train_subset.parquet",
        max_instances=5,
        use_mock_api=True,  # Use mock for testing
        use_remote_service=False,  # Don't use remote service for basic test
        max_steps=5,
        max_search_calls=3,
        disable_limiter=True,  # Disable Ray limiter for testing
        use_xverify=False,  # Disable xverify for basic test
        use_otc=False       # Disable OTC for basic test
    )

    env = NuminaMathEnv(config)
    obs, info = env.reset(seed=42)

    action = '''<think>
I need to solve this math problem step by step.
</think>

<answer>302</answer>'''

    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Math problem completed. Reward: {reward}, Success: {info.get('success', False)}")
    return True


def test_math_with_search():
    """Test math problem solving with search"""
    print("Testing math with search functionality...")

    config = NuminaMathEnvConfig(
        dataset_path="/mnt/chensiheng/ai_researcher_data/numinamath_new/train_subset.parquet",
        max_instances=5,
        use_mock_api=True,  # Use mock for testing
        use_remote_service=False,
        max_steps=8,
        max_search_calls=3,
        disable_limiter=True,
        use_xverify=False,
        use_otc=False
    )

    env = NuminaMathEnv(config)
    obs, info = env.reset(seed=42)

    # First action: search for mathematical concepts
    action1 = '''<think>
This is a combinatorics problem about coloring a grid with constraints.
Let me search for relevant information.
</think>

<code>
result = web_search("combinatorics grid coloring constraints maximum")
print(result)
</code>'''

    obs1, reward1, terminated1, truncated1, info1 = env.step(action1)
    print(f"Search step completed. Reward: {reward1}")

    if not (terminated1 or truncated1):
        # Second action: provide answer based on search
        action2 = '''<think>
Based on the search results and mathematical analysis, I can solve this problem.
For a 5×100 grid where each cell has at most 2 adjacent black cells,
the maximum number of black cells is 302.
</think>

<answer>302</answer>'''

        obs2, reward2, terminated2, truncated2, info2 = env.step(action2)
        print(f"Answer provided. Final reward: {reward2}, Success: {info2.get('success', False)}")

    return True


def test_reward_computation():
    """Test reward computation functionality"""
    print("Testing reward computation...")

    try:
        from roll.agentic.reward_compute.numina_math_reward import compute_score
        
        # Test basic math reward
        solution_str = "<answer>302</answer>"
        ground_truth = ["302"]
        question = "Find the largest possible value of n."
        
        result = compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            question=question,
            use_xverify=False,
            use_otc=False,
            return_dict=True
        )
        
        print(f"Reward computation successful. Score: {result['score']}, Correct: {result['is_correct']}")
        return True
        
    except Exception as e:
        print(f"Reward computation failed: {e}")
        return False


def test_prime_math_grader():
    """Test prime math grader"""
    print("Testing prime math grader...")

    try:
        from roll.agentic.reward_compute.prime_math import grade_answer
        
        test_cases = [
            ("302", "302", True),
            ("1/2", "0.5", True),
            ("2+2", "4", True),
            ("wrong", "302", False),
        ]
        
        all_passed = True
        for given, truth, expected in test_cases:
            try:
                result = grade_answer(given, truth)
                if result == expected:
                    print(f"  ✓ grade_answer('{given}', '{truth}') = {result}")
                else:
                    print(f"  ✗ grade_answer('{given}', '{truth}') = {result}, expected {expected}")
                    all_passed = False
            except Exception as e:
                print(f"  ✗ grade_answer('{given}', '{truth}') failed: {e}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"Prime math grader test failed: {e}")
        return False


def test_trajectory_and_save(dataset_path, output_file, num_samples=3):
    """Test environment interaction and save trajectories"""
    print(f"Testing trajectory interaction with {num_samples} samples...")

    config = NuminaMathEnvConfig(
        dataset_path=dataset_path,
        max_instances=num_samples,
        use_mock_api=True,
        use_remote_service=False,
        max_steps=6,
        max_search_calls=3,
        disable_limiter=True,
        use_xverify=False,
        use_otc=False
    )

    env = NuminaMathEnv(config)
    trajectories = []

    for i in range(num_samples):
        print(f"Processing sample {i+1}/{num_samples}")

        trajectory = {
            "sample_id": i,
            "seed": i + 42,
            "question_data": None,
            "trajectory": {},
            "success": False,
            "total_reward": 0.0
        }

        try:
            # obs0 - initial state
            obs0, info = env.reset(seed=i + 42)
            trajectory["question_data"] = env.current_question_data
            trajectory["trajectory"]["obs0"] = obs0

            question = env.current_question_data.get("question", "")
            total_reward = 0.0
            step_count = 0

            # Generate actions until done
            for step in range(config.max_steps):
                if step == 0:  # First search
                    if "grid" in question.lower() or "table" in question.lower():
                        search_query = "combinatorics grid coloring maximum constraints"
                    elif "digit" in question.lower():
                        search_query = "number theory digits sum constraints"
                    elif "circle" in question.lower():
                        search_query = "geometry circles tangent intersection"
                    else:
                        search_query = question[:50]

                    action = f'''<think>I need to search for information about this math problem.</think>
<code>
result = web_search("{search_query}")
print(result)
</code>'''

                elif step == 1:  # Second search or analysis
                    action = f'''<think>Let me analyze this problem more carefully.</think>
<code>
result = web_search("mathematical solution {question[:30]}")
print(result)
</code>'''

                else:  # Final answer
                    golden_answer = env.current_question_data["golden_answers"][0] if env.current_question_data["golden_answers"] else "Unknown"
                    action = f'''<think>Based on my analysis, I can provide the answer.</think>
<answer>{golden_answer}</answer>'''

                # action_t
                trajectory["trajectory"][f"action{step}"] = action

                # Execute action and get reward and next state
                next_obs, reward, terminated, truncated, info = env.step(action)

                # reward_t
                trajectory["trajectory"][f"reward{step}"] = reward
                total_reward += reward

                # obs_{t+1}
                trajectory["trajectory"][f"obs{step+1}"] = next_obs
                step_count = step + 1

                # Record termination reason
                if terminated or truncated:
                    trajectory["success"] = info.get("success", False)
                    trajectory["termination_reason"] = info.get("termination_reason", "unknown")
                    break

            trajectory["total_reward"] = total_reward
            trajectory["num_steps"] = step_count

        except Exception as e:
            trajectory["error"] = str(e)
            print(f"Error in sample {i+1}: {e}")

        trajectories.append(trajectory)

    # Save trajectories
    with open(output_file, 'w') as f:
        json.dump(trajectories, f, indent=2)

    success_count = sum(1 for t in trajectories if t.get("success", False))
    avg_steps = sum(t.get("num_steps", 0) for t in trajectories) / len(trajectories)
    print(f"Trajectory test completed: {success_count}/{num_samples} successful")
    print(f"Average steps per trajectory: {avg_steps:.1f}")
    print(f"Results saved to {output_file}")

    return trajectories


def test_xverify_math_evaluation():
    """Test xverify-based math evaluation"""
    print("Testing xverify math evaluation...")

    try:
        from roll.agentic.reward_compute.numina_math_reward import compute_score

        # Test case 1: Complex mathematical expression
        solution_complex = '''<think>
This is a combinatorics problem about coloring a 5×100 grid.
Each cell can have at most 2 adjacent black cells.
Let me work through this systematically.

For a 5×100 grid, we have 500 total cells.
The constraint is that each cell has at most 2 adjacent black cells.

This is equivalent to finding the maximum independent set in a constraint graph.
Using graph theory and optimization techniques, the maximum is 302.
</think>

<code>
# Let's verify this with a calculation
rows, cols = 5, 100
total_cells = rows * cols
print(f"Total cells: {total_cells}")

# Theoretical maximum with constraints
max_black = (rows * cols) // 2 + 2
print(f"Theoretical maximum: {max_black}")
</code>

<execution_results>
Total cells: 500
Theoretical maximum: 252
</execution_results>

<think>
Actually, let me reconsider. The optimal pattern gives us 302 black cells.
</think>

<answer>302</answer>'''

        ground_truth = ["302"]
        question = "A 5×100 table is divided into 500 unit square cells, where n of them are coloured black and the rest are coloured white. Each of the unit square cells has at most two adjacent black unit square cells. Find the largest possible value of n."

        # Test with xverify enabled
        result_xverify = compute_score(
            solution_str=solution_complex,
            ground_truth=ground_truth,
            question=question,
            use_xverify=True,
            use_otc=False,
            return_dict=True
        )

        # Test with traditional evaluation
        result_traditional = compute_score(
            solution_str=solution_complex,
            ground_truth=ground_truth,
            question=question,
            use_xverify=False,
            use_otc=False,
            return_dict=True
        )

        print(f"Xverify evaluation: Score={result_xverify['score']:.3f}, Method={result_xverify['evaluation_method']}")
        print(f"Traditional evaluation: Score={result_traditional['score']:.3f}, Method={result_traditional['evaluation_method']}")

        # Test case 2: Mathematical expressions that should be equivalent
        test_cases = [
            ("<answer>1/2</answer>", ["0.5"], "What is one half?"),
            ("<answer>0.5</answer>", ["1/2"], "What is 1/2 as a decimal?"),
            ("<answer>2+2</answer>", ["4"], "What is 2+2?"),
            ("<answer>π</answer>", ["3.14159"], "What is pi?"),
        ]

        for solution, gt, q in test_cases:
            result = compute_score(
                solution_str=solution,
                ground_truth=gt,
                question=q,
                use_xverify=True,
                use_otc=False,
                return_dict=True
            )
            print(f"  {solution} vs {gt}: Score={result['score']:.3f}, Correct={result['is_correct']}")

        return True

    except Exception as e:
        print(f"Xverify math test failed: {e}")
        return False


def test_otc_math_rewards():
    """Test OTC rewards for math problems"""
    print("Testing OTC math rewards...")

    try:
        from roll.agentic.reward_compute.numina_math_reward import compute_score

        # Test case: Efficient vs inefficient tool usage
        solution_efficient = '''<think>
I need to solve this geometry problem about circles.
</think>

<code>
result = web_search("circle tangent intersection angle geometry")
print(result)
</code>

<execution_results>
Found relevant geometric theorems about circle tangents and intersections.
The angle NMB can be calculated using properties of tangent lines.
</execution_results>

<think>
Based on the geometric properties, the angle is 45 degrees.
</think>

<answer>45</answer>'''

        solution_inefficient = '''<think>
I need to solve this geometry problem.
</think>

<code>
result1 = web_search("circle")
print(result1)
</code>

<execution_results>
General information about circles.
</execution_results>

<code>
result2 = web_search("tangent")
print(result2)
</code>

<execution_results>
Information about tangent lines.
</execution_results>

<code>
result3 = web_search("intersection")
print(result3)
</code>

<execution_results>
Information about intersections.
</execution_results>

<code>
result4 = web_search("angle calculation geometry")
print(result4)
</code>

<execution_results>
Methods for calculating angles in geometry.
</execution_results>

<code>
result5 = web_parse("https://mathworld.wolfram.com/Circle.html", "tangent angle")
print(result5)
</code>

<execution_results>
Detailed information about circle tangent angles.
</execution_results>

<answer>45</answer>'''

        ground_truth = ["45"]
        question = "Let the circles k₁ and k₂ intersect at two distinct points A and B, and let t be a common tangent of k₁ and k₂, that touches k₁ and k₂ at M and N, respectively. If t ⊥ AM and MN=2AM, evaluate ∠NMB."

        # Test OTC-PPO
        result_efficient_ppo = compute_score(
            solution_str=solution_efficient,
            ground_truth=ground_truth,
            question=question,
            use_xverify=False,
            use_otc=True,
            otc_method="ppo",
            return_dict=True
        )

        result_inefficient_ppo = compute_score(
            solution_str=solution_inefficient,
            ground_truth=ground_truth,
            question=question,
            use_xverify=False,
            use_otc=True,
            otc_method="ppo",
            return_dict=True
        )

        # Test OTC-GRPO
        correct_trajectories = [solution_efficient, "45"]  # Efficient trajectory as reference

        result_inefficient_grpo = compute_score(
            solution_str=solution_inefficient,
            ground_truth=ground_truth,
            question=question,
            use_xverify=False,
            use_otc=True,
            otc_method="grpo",
            correct_trajectories=correct_trajectories,
            return_dict=True
        )

        print(f"Efficient solution (PPO): Score={result_efficient_ppo['score']:.3f}")
        print(f"Inefficient solution (PPO): Score={result_inefficient_ppo['score']:.3f}")
        print(f"Inefficient solution (GRPO): Score={result_inefficient_grpo['score']:.3f}")

        # OTC should penalize inefficient tool usage
        assert result_efficient_ppo['score'] >= result_inefficient_ppo['score'], "OTC should favor efficient tool usage"

        return True

    except Exception as e:
        print(f"OTC math test failed: {e}")
        return False


def test_combined_xverify_otc_math():
    """Test combined xverify and OTC evaluation for math"""
    print("Testing combined xverify + OTC math evaluation...")

    try:
        from roll.agentic.reward_compute.numina_math_reward import compute_score

        solution_str = '''<think>
I need to find the largest possible value of n for three-digit numbers with specific constraints.
</think>

<code>
result = web_search("three digit numbers sum digits constraints combinatorics")
print(result)
</code>

<execution_results>
Found information about combinatorial problems with digit constraints.
For numbers with sum of digits = 9 and unique position digits, the maximum count is limited.
</execution_results>

<think>
Given the constraints:
1. No digit 0
2. Sum of digits = 9
3. All units digits different
4. All tens digits different
5. All hundreds digits different

This is a constrained optimization problem. The maximum n is 5.
</think>

<answer>5</answer>'''

        ground_truth = ["5"]
        question = "Let n three-digit numbers satisfy: (1) No number contains digit 0. (2) Sum of digits of each number is 9. (3) Units digits of any two numbers are different. (4) Tens digits of any two numbers are different. (5) Hundreds digits of any two numbers are different. Find the largest possible value of n."

        # Test all combinations
        result_basic = compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            question=question,
            use_xverify=False,
            use_otc=False,
            return_dict=True
        )

        result_xverify_only = compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            question=question,
            use_xverify=True,
            use_otc=False,
            return_dict=True
        )

        result_otc_only = compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            question=question,
            use_xverify=False,
            use_otc=True,
            otc_method="ppo",
            return_dict=True
        )

        result_combined = compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            question=question,
            use_xverify=True,
            use_otc=True,
            otc_method="ppo",
            return_dict=True
        )

        print(f"Basic: Score={result_basic['score']:.3f}, Method={result_basic['evaluation_method']}")
        print(f"Xverify only: Score={result_xverify_only['score']:.3f}, Method={result_xverify_only['evaluation_method']}")
        print(f"OTC only: Score={result_otc_only['score']:.3f}")
        print(f"Combined: Score={result_combined['score']:.3f}, Method={result_combined['evaluation_method']}")

        return True

    except Exception as e:
        print(f"Combined math test failed: {e}")
        return False


def test_math_edge_cases():
    """Test edge cases for math evaluation"""
    print("Testing math evaluation edge cases...")

    try:
        from roll.agentic.reward_compute.numina_math_reward import compute_score

        # Test cases with different mathematical formats
        test_cases = [
            # (solution, ground_truth, description)
            ("<answer>\\frac{1}{2}</answer>", ["0.5"], "LaTeX fraction"),
            ("<answer>$\\pi$</answer>", ["π"], "LaTeX pi symbol"),
            ("<answer>2.000</answer>", ["2"], "Decimal with trailing zeros"),
            ("<answer>(1+1)</answer>", ["2"], "Expression in parentheses"),
            ("<answer>1,000</answer>", ["1000"], "Number with comma"),
            ("<answer>50%</answer>", ["0.5"], "Percentage"),
        ]

        for solution, gt, desc in test_cases:
            result = compute_score(
                solution_str=solution,
                ground_truth=gt,
                question=f"Test case: {desc}",
                use_xverify=False,
                use_otc=False,
                return_dict=True
            )
            print(f"  {desc}: {solution} vs {gt} -> Score={result['score']:.3f}, Correct={result['is_correct']}")

        return True

    except Exception as e:
        print(f"Math edge cases test failed: {e}")
        return False


if __name__ == "__main__":
    try:
        # Test 1: Basic functionality
        test_basic_math()

        # Test 2: Math with search
        test_math_with_search()

        # Test 3: Reward computation
        test_reward_computation()

        # Test 4: Prime math grader
        test_prime_math_grader()

        # Test 5: Trajectory generation
        dataset_path = "/mnt/chensiheng/ai_researcher_data/numinamath_new/train_subset.parquet"
        test_trajectory_and_save(dataset_path, "numina_math_trajectories.json", num_samples=3)

        # Test 6: Advanced xverify math evaluation
        test_xverify_math_evaluation()

        # Test 7: OTC math rewards
        test_otc_math_rewards()

        # Test 8: Combined xverify + OTC for math
        test_combined_xverify_otc_math()

        # Test 9: Math edge cases
        test_math_edge_cases()

        print("All tests completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
