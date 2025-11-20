# Bandit Algorithm Tests

This directory contains tests for the Bandit-REINFORCE++ components.

## Test Files

### test_bandit_actor.py

Tests for BanditActor and NeuralUCB algorithm:
- `test_neural_ucb_local()`: Test NeuralUCB without Ray
- `test_bandit_actor_ray()`: Test BanditActor with Ray
- `test_convergence()`: Test convergence to best arm

**Run:**
```bash
cd roll_dev/ROLL
python tests/algorithms/bandit/test_bandit_actor.py
```

## Test Coverage

### NeuralUCB Algorithm
- ✓ Arm selection with UCB strategy
- ✓ Reward updates
- ✓ Neural network training
- ✓ Buffer management
- ✓ Covariance matrix updates
- ✓ Convergence behavior

### BanditActor (Ray)
- ✓ Remote arm selection
- ✓ Remote updates
- ✓ Statistics tracking
- ✓ Monitoring integration
- ✓ Multi-process safety

## Expected Results

### test_neural_ucb_local
- Should complete without errors
- Best arm (arm 0) should have highest mean reward
- Takes ~5 seconds

### test_bandit_actor_ray
- Should initialize Ray successfully
- Should show increasing update counts
- Should produce monitoring metrics
- Takes ~10 seconds

### test_convergence
- Should converge to true best arm (arm 2)
- After 500 episodes, most selections should be arm 2
- Mean reward of arm 2 should be highest
- Takes ~15 seconds

## Troubleshooting

### Ray initialization errors
```bash
# Stop existing Ray instance
ray stop

# Try again
python tests/algorithms/bandit/test_bandit_actor.py
```

### Import errors
```bash
# Ensure ROLL is in PYTHONPATH
export PYTHONPATH=/path/to/roll_dev/ROLL:$PYTHONPATH

# Or run from ROLL root
cd roll_dev/ROLL
python tests/algorithms/bandit/test_bandit_actor.py
```

### CUDA errors
All tests default to CPU. If you want to use GPU:
- Set `device="cuda"` in test code
- Ensure PyTorch with CUDA is installed
