#!/usr/bin/env bash
# Cluster shutdown script for ROLL dual environment training
# Run this script from the jumphost (aicoder) to shutdown the entire Ray cluster

set -e

# Cluster configuration - MUST MATCH YOUR setup_cluster.sh
MASTER_IP="172.26.81.88"  # Head node IP, siheng7
PET_MASTER_PORT=6379

# Node configuration - MUST MATCH YOUR setup_cluster.sh
WORKER_NODES=(
    "172.26.85.167" # siheng8
    "172.26.87.246" # siheng13
    "172.26.91.117" # siheng14
    # Add more worker nodes as needed
)

# Environment configuration
CONDA_PATH="/mnt/chensiheng/conda/ourconda_bashrc"
CONDA_ENV="roll"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print shutdown info
print_shutdown_info() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  ROLL Dual Environment Cluster Shutdown${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo -e "${YELLOW}Shutting down cluster:${NC}"
    echo "  Master IP: $MASTER_IP:$PET_MASTER_PORT"
    echo "  Worker Nodes: ${#WORKER_NODES[@]} nodes"
    for i in "${!WORKER_NODES[@]}"; do
        echo "    Node $i: ${WORKER_NODES[$i]}"
    done
    echo ""
}

# Function to shutdown head node
shutdown_head_node() {
    echo -e "${YELLOW}Shutting down head node at $MASTER_IP...${NC}"
    
    ssh -o StrictHostKeyChecking=no $MASTER_IP << 'EOF'
#!/bin/bash

echo "=== Shutting down Ray head node ==="

# Activate conda environment for proper ray command access
source /mnt/chensiheng/conda/ourconda_bashrc
conda activate roll

# Show current Ray status before shutdown
echo "Current Ray status:"
ray status 2>/dev/null || echo "Ray not running or not accessible"

# Kill tmux session first
echo "Killing tmux session 'head'..."
tmux kill-session -t head 2>/dev/null && echo "✓ Tmux session 'head' killed" || echo "! Tmux session 'head' not found"

# Stop Ray gracefully
echo "Stopping Ray processes gracefully..."
ray stop -f 2>/dev/null && echo "✓ Ray stopped gracefully" || echo "! Ray stop command failed"

# Force kill any remaining Ray processes
echo "Force killing any remaining Ray processes..."
pkill -f "ray::" 2>/dev/null && echo "✓ Ray processes killed" || echo "! No Ray processes found"
pkill -f "raylet" 2>/dev/null && echo "✓ Raylet processes killed" || echo "! No raylet processes found"
pkill -f "gcs_server" 2>/dev/null && echo "✓ GCS server processes killed" || echo "! No GCS server processes found"
pkill -f "dashboard" 2>/dev/null && echo "✓ Dashboard processes killed" || echo "! No dashboard processes found"

# Clean up Ray temp directories
echo "Cleaning up Ray temp files..."
rm -rf /tmp/ray* 2>/dev/null && echo "✓ Ray temp files cleaned" || echo "! No Ray temp files found"

# Check for any remaining processes
echo "Checking for remaining Ray processes..."
remaining_processes=$(ps aux | grep -E "(ray::|raylet|gcs_server)" | grep -v grep | wc -l)
if [ $remaining_processes -eq 0 ]; then
    echo "✓ All Ray processes successfully terminated"
else
    echo "! Warning: $remaining_processes Ray processes still running"
    ps aux | grep -E "(ray::|raylet|gcs_server)" | grep -v grep
fi

echo "=== Head node shutdown completed ==="
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Head node shutdown completed${NC}"
    else
        echo -e "${RED}✗ Head node shutdown had issues${NC}"
    fi
}

# Function to shutdown worker node
shutdown_worker_node() {
    local worker_ip=$1
    local node_index=$2
    
    echo -e "${YELLOW}Shutting down worker node $node_index at $worker_ip...${NC}"
    
    ssh -o StrictHostKeyChecking=no $worker_ip << 'EOF'
#!/bin/bash

echo "=== Shutting down Ray worker node ==="

# Activate conda environment for proper ray command access
source /mnt/chensiheng/conda/ourconda_bashrc
conda activate roll

# Get node IP for identification
NODE_IP=$(ip -4 addr show eth0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
echo "Shutting down worker node with IP: $NODE_IP"

# Kill tmux session first
echo "Killing tmux session 'node*'..."
for session in $(tmux list-sessions 2>/dev/null | grep "^node" | cut -d: -f1); do
    tmux kill-session -t $session 2>/dev/null && echo "✓ Tmux session '$session' killed" || echo "! Tmux session '$session' not found"
done

# Stop Ray gracefully
echo "Stopping Ray processes gracefully..."
ray stop -f 2>/dev/null && echo "✓ Ray stopped gracefully" || echo "! Ray stop command failed"

# Force kill any remaining Ray processes
echo "Force killing any remaining Ray processes..."
pkill -f "ray::" 2>/dev/null && echo "✓ Ray processes killed" || echo "! No Ray processes found"
pkill -f "raylet" 2>/dev/null && echo "✓ Raylet processes killed" || echo "! No raylet processes found"

# Clean up Ray temp directories
echo "Cleaning up Ray temp files..."
rm -rf /tmp/ray* 2>/dev/null && echo "✓ Ray temp files cleaned" || echo "! No Ray temp files found"

# Check for any remaining processes
echo "Checking for remaining Ray processes..."
remaining_processes=$(ps aux | grep -E "(ray::|raylet)" | grep -v grep | wc -l)
if [ $remaining_processes -eq 0 ]; then
    echo "✓ All Ray processes successfully terminated"
else
    echo "! Warning: $remaining_processes Ray processes still running"
    ps aux | grep -E "(ray::|raylet)" | grep -v grep
fi

echo "=== Worker node shutdown completed ==="
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Worker node $node_index ($worker_ip) shutdown completed${NC}"
    else
        echo -e "${RED}✗ Worker node $node_index ($worker_ip) shutdown had issues${NC}"
    fi
}

# Function to force cleanup all nodes
force_cleanup_all() {
    echo -e "${YELLOW}Performing force cleanup on all nodes...${NC}"
    
    # Force cleanup head node
    ssh -o StrictHostKeyChecking=no $MASTER_IP 'pkill -f ray; pkill -f tmux; rm -rf /tmp/ray*' 2>/dev/null &
    
    # Force cleanup worker nodes
    for i in "${!WORKER_NODES[@]}"; do
        worker_ip=${WORKER_NODES[$i]}
        ssh -o StrictHostKeyChecking=no $worker_ip 'pkill -f ray; pkill -f tmux; rm -rf /tmp/ray*' 2>/dev/null &
    done
    
    wait
    echo -e "${GREEN}✓ Force cleanup completed${NC}"
}

# Function to check cluster status after shutdown
check_shutdown_status() {
    echo -e "${YELLOW}Verifying cluster shutdown...${NC}"
    
    echo "Checking head node..."
    ssh -o StrictHostKeyChecking=no $MASTER_IP << 'EOF'
ray_processes=$(ps aux | grep -E "(ray::|raylet|gcs_server)" | grep -v grep | wc -l)
tmux_sessions=$(tmux list-sessions 2>/dev/null | grep -E "(head|node)" | wc -l)
echo "Head node - Ray processes: $ray_processes, Tmux sessions: $tmux_sessions"
EOF
    
    for i in "${!WORKER_NODES[@]}"; do
        worker_ip=${WORKER_NODES[$i]}
        echo "Checking worker node $i ($worker_ip)..."
        ssh -o StrictHostKeyChecking=no $worker_ip << 'EOF'
ray_processes=$(ps aux | grep -E "(ray::|raylet)" | grep -v grep | wc -l)
tmux_sessions=$(tmux list-sessions 2>/dev/null | grep -E "node" | wc -l)
echo "Worker node - Ray processes: $ray_processes, Tmux sessions: $tmux_sessions"
EOF
    done
}

# Main execution
case "${1:-shutdown}" in
    "shutdown" | "stop")
        print_shutdown_info
        echo -e "${GREEN}Shutting down ROLL dual environment Ray cluster...${NC}"
        
        # Shutdown worker nodes first (parallel)
        echo -e "${YELLOW}Shutting down worker nodes...${NC}"
        for i in "${!WORKER_NODES[@]}"; do
            shutdown_worker_node ${WORKER_NODES[$i]} $i &
        done
        
        # Wait for all worker nodes to complete
        wait
        
        # Shutdown head node last
        shutdown_head_node
        
        # Wait a bit for cleanup
        sleep 5
        
        # Verify shutdown
        check_shutdown_status
        
        echo -e "${GREEN}Cluster shutdown completed!${NC}"
        ;;
        
    "force")
        print_shutdown_info
        echo -e "${RED}Performing FORCE cleanup of all nodes...${NC}"
        force_cleanup_all
        sleep 3
        check_shutdown_status
        echo -e "${GREEN}Force cleanup completed!${NC}"
        ;;
        
    "status")
        check_shutdown_status
        ;;
        
    *)
        echo "Usage: $0 {shutdown|stop|force|status}"
        echo ""
        echo "Commands:"
        echo "  shutdown  - Gracefully shutdown the entire Ray cluster (default)"
        echo "  stop      - Same as shutdown"
        echo "  force     - Force kill all Ray processes and tmux sessions"
        echo "  status    - Check if cluster is properly shut down"
        echo ""
        echo "Examples:"
        echo "  $0                    # Graceful shutdown"
        echo "  $0 shutdown           # Graceful shutdown"
        echo "  $0 force              # Force cleanup"
        echo "  $0 status             # Check status"
        exit 1
        ;;
esac 