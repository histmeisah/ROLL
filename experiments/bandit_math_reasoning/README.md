# Bandit-REINFORCE++ for Mathematical Reasoning

This experiment implements a two-layer optimization framework that combines:
- **Outer loop**: NeuralUCB for prompt selection (contextual bandit)
- **Inner loop**: REINFORCE++ for LLM policy training

## Overview

The system automatically selects optimal prompts for mathematical reasoning tasks using multi-armed bandit algorithms. Each problem is:
1. Encoded into a context embedding
2. Passed to NeuralUCB which selects a prompt using UCB strategy
3. Formatted with the selected prompt
4. Solved by the LLM policy
5. Evaluated and the reward is fed back to NeuralUCB

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install sentence-transformers datasets

# Or install from requirements if available
pip install -r requirements.txt
```

### Basic Training

```bash
# Run with default configuration
python experiments/start_bandit_math_reasoning.py

# Or from ROLL root directory
cd roll_dev/ROLL
python experiments/start_bandit_math_reasoning.py
```

### Configuration Overrides

```bash
# Change rollout batch size
python experiments/start_bandit_math_reasoning.py rollout_batch_size=256

# Adjust exploration parameter
python experiments/start_bandit_math_reasoning.py \
    bandit_config.exploration_param=1.5

# Use different dataset
python experiments/start_bandit_math_reasoning.py \
    custom_envs.MathReasoningBandit.env_config.dataset_name=math

# Enable wandb logging
python experiments/start_bandit_math_reasoning.py \
    wandb.enabled=true \
    wandb.entity=your-entity \
    wandb.project=bandit-math
```

## Configuration Structure

### Bandit Configuration

```yaml
bandit_config:
  # Prompts
  prompt_config_path: "configs/prompts/math_reasoning_prompts.yaml"
  preset_name: "diverse_5"  # Use 5 different prompts

  # Context encoder
  encoder_model: "sentence-transformers/all-MiniLM-L6-v2"
  context_dim: 384

  # NeuralUCB parameters
  hidden_dims: [256, 128]
  exploration_param: 1.0  # UCB alpha
  learning_rate: 0.001
  buffer_size: 10000
  update_freq: 10
```

### Environment Configuration

```yaml
custom_envs:
  MathReasoningBandit:
    env_type: "math_reasoning_bandit"
    env_manager_cls: "roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager"

    env_config:
      dataset_name: "gsm8k"  # Options: gsm8k, math
      reward_correct: 1.0
      reward_incorrect: 0.0
```

## Architecture

```
Pipeline Initialization:
├─ Load prompt templates (5 prompts)
├─ Initialize encoder (SentenceTransformer)
├─ Start BanditActor (Ray)
└─ Create environments (32 groups × 4 envs = 128 parallel)

Training Loop (each step):
├─ Environment Reset
│  ├─ Sample problem from dataset
│  ├─ Encode problem → embedding
│  ├─ BanditActor.select_arm(embedding) → prompt_idx
│  └─ Format observation with selected prompt
│
├─ Policy Generation
│  └─ LLM generates solution
│
├─ Environment Step
│  ├─ Extract answer from solution
│  ├─ Verify against ground truth
│  ├─ Compute reward (1.0 / 0.0)
│  └─ BanditActor.update(prompt_idx, embedding, reward)
│
└─ Policy Training
   └─ REINFORCE++ updates
```

## Monitoring

### Bandit Metrics (logged to wandb/tensorboard)

- `bandit/prompts/{name}/mean_reward`: Per-prompt average reward
- `bandit/prompts/{name}/success_rate`: Per-prompt success rate
- `bandit/prompts/{name}/selections`: Number of times selected
- `bandit/selection_dist/{name}`: Selection frequency distribution
- `bandit/is_converged`: Whether bandit has converged

### Policy Metrics

- Standard RL metrics (loss, advantage, KL divergence, etc.)
- Episode rewards and success rates

## Customization

### Adding New Prompts

1. Edit `configs/prompts/math_reasoning_prompts.yaml`
2. Add your prompt to a category or create new preset
3. Update config to use your preset:
   ```yaml
   bandit_config:
     preset_name: "your_preset_name"
   ```

### Using Different Datasets

Modify environment config:
```yaml
custom_envs:
  MathReasoningBandit:
    env_config:
      dataset_name: "math"  # or "gsm8k"
      dataset_split: "train"
```

### Changing Model Size

```yaml
pretrain: "Qwen/Qwen2.5-3B-Instruct"  # or 7B, 14B, etc.

actor_train:
  model_args:
    model_name_or_path: "Qwen/Qwen2.5-3B-Instruct"
```

## Troubleshooting

### ImportError: sentence_transformers not found
```bash
pip install sentence-transformers
```

### ImportError: datasets not found
```bash
pip install datasets
```

### Ray initialization errors
```bash
# Check Ray is installed
pip install ray

# Restart Ray if needed
ray stop
```

### CUDA out of memory
Reduce batch sizes:
```yaml
rollout_batch_size: 64  # Reduce from 128
actor_train:
  training_args:
    per_device_train_batch_size: 2  # Reduce from 4
```

## Expected Results

On GSM8K with Qwen2.5-0.5B-Instruct:
- Initial success rate: ~20-30%
- After 1000 steps: ~40-50%
- Bandit should converge to best prompt(s) within 200-500 steps

## Advanced: Off-Policy Training

Enable replay buffer for sample efficiency:
```yaml
replay:
  enabled: true
  capacity: 10000
  train_steps_per_env_step: 2  # Train 2x per rollout
```

## Citation

If you use this code, please cite:
```bibtex
@inproceedings{zhou2020neural,
  title={Neural contextual bandits with ucb-based exploration},
  author={Zhou, Dongruo and Li, Lihong and Gu, Quanquan},
  booktitle={International Conference on Machine Learning},
  year={2020}
}
```
