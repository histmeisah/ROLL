#!/bin/bash
set +x

# Set environment variables for dual environment training
export XHPANG_TRAIN="/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet"
export NUMINA_TRAIN="/mnt/chensiheng/ai_researcher_data/numinamath_new/train_subset.parquet"
export XHPANG_VAL="/mnt/chensiheng/ai_researcher_data/xhpang_search/validation.parquet"
export NUMINA_VAL="/mnt/chensiheng/ai_researcher_data/numinamath_new/validation_subset.parquet"

# Print dataset paths for verification
echo "=== Dual Environment Training Configuration ==="
echo "Search Train Dataset: $XHPANG_TRAIN"
echo "Math Train Dataset: $NUMINA_TRAIN"
echo "Search Val Dataset: $XHPANG_VAL"
echo "Math Val Dataset: $NUMINA_VAL"
echo "=============================================="

# Get config path
CONFIG_PATH=$(basename $(dirname $0))

# Run the dual environment agentic pipeline
python examples/start_agentic_pipeline.py \
    --config_path $CONFIG_PATH \
    --config_name agentic_search_train
