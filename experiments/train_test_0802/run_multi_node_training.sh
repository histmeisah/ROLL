#!/bin/bash
set +x

# Multi-node multi-GPU training script for ROLL dual environment (search + math)
# This script should be run on the MASTER node AFTER Ray cluster is already established
#
# Prerequisites:
# 1. Ray cluster should be already running (use setup_cluster.sh)
# 2. All worker nodes should be connected to the cluster
# 3. Verify cluster status with: ray status

echo "=== ROLL Multi-Node Multi-GPU Dual Environment Training ==="
echo "Configuration: Multi-node × 8 GPUs per node"
echo "Environments: Search + Math (1:1 ratio)"
echo "Reward System: Outcome-based with xverify"
echo "=============================================="

# Set environment variables for dual environment training
export XHPANG_TRAIN="/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet"
export NUMINA_TRAIN="/mnt/chensiheng/ai_researcher_data/numinamath_new/train_subset.parquet"
export XHPANG_VAL="/mnt/chensiheng/ai_researcher_data/xhpang_search/validation.parquet"
export NUMINA_VAL="/mnt/chensiheng/ai_researcher_data/numinamath_new/validation_subset.parquet"

# ROLL environment configuration
export PYTHONPATH="/mnt/chensiheng/weiyu/xueban_v2/ROLL:$PYTHONPATH"

# Multi-node configuration (should match Ray cluster setup)
export MASTER_ADDR="172.26.81.88"  # Should match setup_cluster.sh
export MASTER_PORT="6379"
export NCCL_SOCKET_IFNAME=eth0  # Network interface
export GLOO_SOCKET_IFNAME=eth0

# Ray configuration for multi-node
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0

# Wandb configuration for offline logging
export WANDB_MODE=offline
export WANDB_DIR=./output/wandb
mkdir -p $WANDB_DIR

# Trajectory logging
export ROLL_TRAJECTORY_LOGGING=true
export ROLL_TRAJECTORY_LOG_DIR="./output/trajectories"
mkdir -p $ROLL_TRAJECTORY_LOG_DIR

# Print configuration for verification
echo "=== Multi-Node Configuration ==="
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Network Interface: $NCCL_SOCKET_IFNAME"
echo "PYTHONPATH: $PYTHONPATH"
echo "================================"

# Print dataset paths for verification
echo "=== Dataset Configuration ==="
echo "Search Train: $XHPANG_TRAIN"
echo "Math Train: $NUMINA_TRAIN"
echo "Search Val: $XHPANG_VAL"
echo "Math Val: $NUMINA_VAL"
echo "============================="

# Check if datasets exist
echo "=== Dataset Verification ==="
for dataset in "$XHPANG_TRAIN" "$NUMINA_TRAIN" "$XHPANG_VAL" "$NUMINA_VAL"; do
    if [ -f "$dataset" ]; then
        size=$(du -h "$dataset" | cut -f1)
        echo "✓ $dataset ($size)"
    else
        echo "✗ $dataset (NOT FOUND)"
        exit 1
    fi
done
echo "============================"

# Verify Ray cluster is running and ready
echo "=== Verifying Ray Cluster Status ==="
if ! ray status > /dev/null 2>&1; then
    echo "✗ Ray cluster is not running!"
    echo "Please start the Ray cluster first:"
    echo "  bash setup_cluster.sh"
    exit 1
fi

echo "Ray cluster status:"
ray status
echo "✓ Ray cluster is running and ready"

# Check cluster resources
cluster_resources=$(ray status --format json 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    total_gpus = data.get('cluster_resources', {}).get('GPU', 0)
    total_cpus = data.get('cluster_resources', {}).get('CPU', 0)
    print(f'GPUs: {int(total_gpus)}, CPUs: {int(total_cpus)}')
except:
    print('Unable to parse cluster resources')
")

echo "✓ Cluster resources: $cluster_resources"
echo "================================"

# Get config path
CONFIG_PATH=$(basename $(dirname $0))

echo "=== Starting Multi-Node Training ==="
echo "Config Path: $CONFIG_PATH"
echo "Config Name: agentic_search_train"
echo "=================================="

# Run the dual environment agentic pipeline
python examples/start_agentic_pipeline.py \
    --config_path $CONFIG_PATH \
    --config_name agentic_search_train

echo "=== Training Completed ==="
echo "Note: Ray cluster is still running for potential additional experiments"
echo "To stop the Ray cluster, run: bash shutdown_cluster.sh"
echo "To sync wandb logs, run: bash start_wandb_sync.sh"
echo "Trajectory logs saved to: $ROLL_TRAJECTORY_LOG_DIR"
echo "========================="
