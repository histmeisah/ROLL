#!/bin/bash
# ==============================================================================
# Single GPU Test: DeepSpeed ZeRO-2 (Training) + vLLM (Inference)
# ==============================================================================
# Model: Qwen2.5-0.5B-Instruct
# Environment: FrozenLake
# GPU: 1x (all roles share GPU 0)
# ==============================================================================
set -e

# Get script directory (works regardless of where the script is called from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH=$(basename "$SCRIPT_DIR")

# Set ROLL path
ROLL_PATH="/mnt/project_modelware_roce/zhaojian/liangsirui/weiyu/projects/local_roll_dev/roll_dev/ROLL"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

# Set wandb to offline mode (no login required)
export WANDB_MODE=offline
export WANDB_API_KEY=local

# Generate unified timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TRAINING_TIMESTAMP="$TIMESTAMP"
OUTPUT_DIR="$SCRIPT_DIR/output/$TIMESTAMP"

# Set wandb directory
export WANDB_DIR="$OUTPUT_DIR/wandb"

# Create output directories
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/models"
mkdir -p "$OUTPUT_DIR/render"
mkdir -p "$OUTPUT_DIR/wandb"

# Set log file
LOG_FILE="$OUTPUT_DIR/logs/training_${TIMESTAMP}.log"

# Redirect stdout and stderr to both screen and log file
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "========================================"
echo "Single GPU Test: DeepSpeed + vLLM"
echo "========================================"
echo "Timestamp: $TIMESTAMP"
echo "Config: agent_val_frozen_lake_single_gpu_ds_vllm"
echo "Log file: $LOG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

# Check GPU availability
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
else
    echo "nvidia-smi not available, skipping GPU check"
fi
echo ""

# Clean up existing Ray clusters
echo "Cleaning up existing Ray clusters..."
ray stop --force 2>/dev/null || true
sleep 2
echo "Cleanup completed."
echo ""

# Start training
cd "$ROLL_PATH"
echo "Training started at $(date)"
echo ""

python examples/start_agentic_pipeline.py \
    --config_path "$CONFIG_PATH" \
    --config_name agent_val_frozen_lake_single_gpu_ds_vllm

echo ""
echo "========================================"
echo "Training completed at $(date)"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo "Training log: $LOG_FILE"
echo "========================================"
