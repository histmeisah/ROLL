#!/usr/bin/env bash
# Cluster setup script for ROLL dual environment training (search + math)
# Run this script from the jumphost (aicoder) to set up the entire Ray cluster

set -e

# Cluster configuration
MASTER_IP="172.26.81.88"  # Head node IP, siheng7
PET_MASTER_PORT=6379

# Node configuration - add your cluster node IPs here
WORKER_NODES=(
    "172.26.85.167" # siheng8
    "172.26.87.246" # siheng13
    "172.26.91.117" # siheng14
    # Add more worker nodes as needed
)

# Environment configuration - MODIFY THESE AS NEEDED
CONDA_PATH="/mnt/chensiheng/conda/ourconda_bashrc"
CONDA_ENV="roll"  # ROLL conda environment
WANDB_API_KEY="5d830c409e2aa7dff34c333a2f79798a877bfc7b"
WANDB_DIR="/mnt/chensiheng/weiyu/xueban_v2/ROLL/experiments/train_test_0802/output/wandb"

# Trajectory logging configuration
TRAJECTORY_LOG_DIR="/mnt/chensiheng/weiyu/xueban_v2/ROLL/experiments/train_test_0802/output/trajectories"

# Common Ray configuration
RAY_TEMP_DIR="/tmp"
RAY_DASHBOARD_PORT=8265
RAY_CLIENT_SERVER_PORT=10001

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print configuration
print_config() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  ROLL Dual Environment Cluster Configuration${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo -e "${YELLOW}Cluster Setup:${NC}"
    echo "  Master IP: $MASTER_IP:$PET_MASTER_PORT"
    echo "  Worker Nodes: ${#WORKER_NODES[@]} nodes"
    for i in "${!WORKER_NODES[@]}"; do
        echo "    Node $i: ${WORKER_NODES[$i]}"
    done
    echo ""
    echo -e "${YELLOW}Environment:${NC}"
    echo "  Conda Path: $CONDA_PATH"
    echo "  Conda Env: $CONDA_ENV"
    echo "  WandB Dir: $WANDB_DIR"
    echo "  Trajectory Log Dir: $TRAJECTORY_LOG_DIR"
    echo ""
    echo -e "${YELLOW}Ray Config:${NC}"
    echo "  Dashboard: http://$MASTER_IP:$RAY_DASHBOARD_PORT"
    echo "  Client Port: $RAY_CLIENT_SERVER_PORT"
    echo "  Temp Dir: $RAY_TEMP_DIR"
    echo ""
    echo -e "${YELLOW}Training Config:${NC}"
    echo "  Environments: Search + Math (1:1 ratio)"
    echo "  Total GPUs: $((${#WORKER_NODES[@]} + 1)) nodes × 8 GPUs = $(((${#WORKER_NODES[@]} + 1) * 8)) GPUs"
    echo "  Reward System: Outcome-based with xverify"
    echo ""
}

# Function to setup head node
setup_head_node() {
    echo -e "${YELLOW}Setting up head node at $MASTER_IP...${NC}"
    
    ssh -o StrictHostKeyChecking=no $MASTER_IP << EOF
# Create head node setup script
cat > /tmp/start_head.sh << 'HEAD_SCRIPT'
#!/bin/bash

# Force stop any existing Ray instances
ray stop -f && pkill -f ray && sleep 3

# Activate conda environment
source $CONDA_PATH
conda activate $CONDA_ENV

# Set environment variables for ROLL
export WANDB_MODE=offline
export WANDB_DIR="$WANDB_DIR"
mkdir -p \$WANDB_DIR

export WANDB_API_KEY='$WANDB_API_KEY'
export LD_LIBRARY_PATH=/usr/lib64:\$LD_LIBRARY_PATH
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export RAY_raylet_start_wait_time_s=60

# ROLL-specific environment variables
export PYTHONPATH="/mnt/chensiheng/weiyu/xueban_v2/ROLL:\$PYTHONPATH"

# Trajectory logging configuration
export ROLL_TRAJECTORY_LOGGING=true
export ROLL_TRAJECTORY_LOG_DIR="$TRAJECTORY_LOG_DIR"
export ROLL_TRAJECTORY_LOG_PERCENTAGE=0.1
export ROLL_TRAJECTORY_LOG_VERBOSE=true
mkdir -p \$ROLL_TRAJECTORY_LOG_DIR

# Dataset paths for dual environment training
export XHPANG_TRAIN="/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet"
export NUMINA_TRAIN="/mnt/chensiheng/ai_researcher_data/numinamath_new/train_subset.parquet"
export XHPANG_VAL="/mnt/chensiheng/ai_researcher_data/xhpang_search/validation.parquet"
export NUMINA_VAL="/mnt/chensiheng/ai_researcher_data/numinamath_new/validation_subset.parquet"

# Ray configuration
PET_MASTER_PORT=$PET_MASTER_PORT
MASTER_ADDR="$MASTER_IP"

echo "Starting Ray head node..."
ray start --head \\
    --port=\$PET_MASTER_PORT \\
    --node-ip-address=\$MASTER_ADDR \\
    --dashboard-host=0.0.0.0 \\
    --dashboard-port=$RAY_DASHBOARD_PORT \\
    --include-dashboard=true \\
    --ray-client-server-port $RAY_CLIENT_SERVER_PORT \\
    --temp-dir $RAY_TEMP_DIR

echo "Ray head node started on \$MASTER_ADDR:\$PET_MASTER_PORT"
echo "Dashboard available at: http://\$MASTER_ADDR:$RAY_DASHBOARD_PORT"

# Keep the session alive
tail -f /dev/null
HEAD_SCRIPT

chmod +x /tmp/start_head.sh

# Kill existing tmux session if it exists
tmux kill-session -t head 2>/dev/null || true

# Start tmux session for head node
tmux new-session -d -s head '/tmp/start_head.sh'

echo "Head node tmux session 'head' started"
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Head node setup completed${NC}"
    else
        echo -e "${RED}✗ Head node setup failed${NC}"
        exit 1
    fi
}

# Function to setup worker nodes
setup_worker_node() {
    local worker_ip=$1
    local node_index=$2
    
    echo -e "${YELLOW}Setting up worker node $node_index at $worker_ip...${NC}"
    
    ssh -o StrictHostKeyChecking=no $worker_ip << EOF
# Create worker node setup script
cat > /tmp/start_worker.sh << 'WORKER_SCRIPT'
#!/bin/bash

# Force stop any existing Ray instances
ray stop -f && pkill -f ray && sleep 3

# Activate conda environment
source $CONDA_PATH
conda activate $CONDA_ENV

# Set environment variables for ROLL worker
export WANDB_MODE=offline
export WANDB_DIR="$WANDB_DIR"
mkdir -p \$WANDB_DIR

export WANDB_API_KEY='$WANDB_API_KEY'
export LD_LIBRARY_PATH=/usr/lib64:\$LD_LIBRARY_PATH
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# ROLL-specific environment variables
export PYTHONPATH="/mnt/chensiheng/weiyu/xueban_v2/ROLL:\$PYTHONPATH"

# Trajectory logging configuration
export ROLL_TRAJECTORY_LOGGING=true
export ROLL_TRAJECTORY_LOG_DIR="$TRAJECTORY_LOG_DIR"
export ROLL_TRAJECTORY_LOG_PERCENTAGE=0.1
export ROLL_TRAJECTORY_LOG_VERBOSE=true
mkdir -p \$ROLL_TRAJECTORY_LOG_DIR

# Dataset paths for dual environment training
export XHPANG_TRAIN="/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet"
export NUMINA_TRAIN="/mnt/chensiheng/ai_researcher_data/numinamath_new/train_subset.parquet"
export XHPANG_VAL="/mnt/chensiheng/ai_researcher_data/xhpang_search/validation.parquet"
export NUMINA_VAL="/mnt/chensiheng/ai_researcher_data/numinamath_new/validation_subset.parquet"

# Ray configuration
PET_MASTER_PORT=$PET_MASTER_PORT
MASTER_ADDR="$MASTER_IP"
NODE_IP_ADDRESS=\$(ip -4 addr show eth0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

echo "Starting Ray worker node..."
echo "Local IP: \$NODE_IP_ADDRESS"
echo "Connecting to master: \$MASTER_ADDR:\$PET_MASTER_PORT"

ray start --address=\$MASTER_ADDR:\$PET_MASTER_PORT \\
    --node-ip-address=\$NODE_IP_ADDRESS \\
    --temp-dir $RAY_TEMP_DIR

echo "Ray worker node with IP \$NODE_IP_ADDRESS connected to \$MASTER_ADDR:\$PET_MASTER_PORT"

# Keep the session alive
tail -f /dev/null
WORKER_SCRIPT

chmod +x /tmp/start_worker.sh

# Kill existing tmux session if it exists
tmux kill-session -t node$node_index 2>/dev/null || true

# Start tmux session for worker node
tmux new-session -d -s node$node_index '/tmp/start_worker.sh'

echo "Worker node $node_index tmux session 'node$node_index' started"
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Worker node $node_index ($worker_ip) setup completed${NC}"
    else
        echo -e "${RED}✗ Worker node $node_index ($worker_ip) setup failed${NC}"
    fi
}

# Function to check cluster status
check_cluster_status() {
    echo -e "${YELLOW}Checking cluster status...${NC}"
    
    ssh -o StrictHostKeyChecking=no $MASTER_IP << EOF
source $CONDA_PATH
conda activate $CONDA_ENV

echo "Ray cluster status:"
ray status

echo ""
echo "Available nodes:"
python -c "
import ray
ray.init(address='auto')
print(f'Cluster has {len(ray.nodes())} nodes')
for i, node in enumerate(ray.nodes()):
    print(f'Node {i}: {node}')
ray.shutdown()
"
EOF
}

# Function to cleanup cluster
cleanup_cluster() {
    echo -e "${YELLOW}Cleaning up existing cluster...${NC}"
    
    # Cleanup head node
    ssh -o StrictHostKeyChecking=no $MASTER_IP 'tmux kill-session -t head 2>/dev/null || true; ray stop -f 2>/dev/null || true' &
    
    # Cleanup worker nodes
    for i in "${!WORKER_NODES[@]}"; do
        worker_ip=${WORKER_NODES[$i]}
        ssh -o StrictHostKeyChecking=no $worker_ip "tmux kill-session -t node$i 2>/dev/null || true; ray stop -f 2>/dev/null || true" &
    done
    
    wait
    echo -e "${GREEN}✓ Cleanup completed${NC}"
}

# Main execution
case "${1:-setup}" in
    "setup")
        print_config
        echo -e "${GREEN}Setting up ROLL dual environment Ray cluster...${NC}"
        
        # Setup head node
        setup_head_node
        
        # Wait for head node to start
        echo -e "${YELLOW}Waiting for head node to initialize...${NC}"
        sleep 10
        
        # Setup worker nodes in parallel
        echo -e "${YELLOW}Setting up worker nodes...${NC}"
        for i in "${!WORKER_NODES[@]}"; do
            setup_worker_node ${WORKER_NODES[$i]} $i &
        done
        
        # Wait for all worker nodes to complete
        wait
        
        # Wait for all nodes to connect
        echo -e "${YELLOW}Waiting for cluster to stabilize...${NC}"
        sleep 15
        
        # Check cluster status
        check_cluster_status
        
        echo -e "${GREEN}Cluster setup completed!${NC}"
        echo -e "${GREEN}Ray Dashboard: http://$MASTER_IP:8265${NC}"
        echo ""
        echo "To check individual nodes:"
        echo "  Head node: ssh $MASTER_IP 'tmux attach -t head'"
        for i in "${!WORKER_NODES[@]}"; do
            echo "  Worker $i: ssh ${WORKER_NODES[$i]} 'tmux attach -t node$i'"
        done
        ;;
        
    "status")
        check_cluster_status
        ;;
        
    "cleanup")
        cleanup_cluster
        ;;
        
    "restart")
        cleanup_cluster
        sleep 5
        $0 setup
        ;;
        
    *)
        echo "Usage: $0 {setup|status|cleanup|restart}"
        echo ""
        echo "Commands:"
        echo "  setup   - Set up the entire Ray cluster"
        echo "  status  - Check cluster status"
        echo "  cleanup - Stop all Ray processes and tmux sessions"
        echo "  restart - Cleanup and setup again"
        exit 1
        ;;
esac 