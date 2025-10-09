#!/bin/bash

# ROLL WandB同步服务快速启动脚本 - 双环境训练版
# 直接修改下面的WANDB_DIR为你要同步的具体路径

set -e

# 配置 - 直接填写绝对路径
WANDB_DIR="/mnt/chensiheng/weiyu/xueban_v2/ROLL/experiments/train_test_0802/output/wandb"
PROXY_HOST="127.0.0.1"
PROXY_PORT="5000"
SESSION_NAME="roll-dual-env-wandb"
SYNC_INTERVAL=3600  # 修改：改为1小时（3600秒）
WANDB_PROJECT="roll_dual_env_training"
WANDB_API_KEY="5d830c409e2aa7dff34c333a2f79798a877bfc7b"

# Conda环境配置
CONDA_BASHRC="/mnt/chensiheng/conda/ourconda_bashrc"
CONDA_ENV="roll"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 启动ROLL双环境训练 WandB同步服务"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📂 WandB路径: $WANDB_DIR"
echo "🌐 代理设置: $PROXY_HOST:$PROXY_PORT"
echo "📊 WandB项目: $WANDB_PROJECT"
echo "🔑 API密钥: ${WANDB_API_KEY:0:10}..."
echo "⏱️  同步间隔: ${SYNC_INTERVAL}秒 (1小时)"
echo "📺 Tmux会话: $SESSION_NAME"
echo "🐍 Conda环境: $CONDA_ENV"
echo "🔧 训练类型: 搜索环境 + 数学环境 (1:1比例)"
echo "🎯 奖励系统: Outcome-based + xverify + OTC"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查WandB目录是否存在
if [ ! -d "$WANDB_DIR" ]; then
    echo "⚠️ WandB目录不存在，将创建: $WANDB_DIR"
    mkdir -p "$WANDB_DIR"
    echo "✅ 已创建WandB目录"
fi

# 检查Python脚本是否存在
PYTHON_SCRIPT="$SCRIPT_DIR/sync_wandb_to_cloud.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ 错误: Python同步脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

# 检查conda环境文件是否存在
if [ ! -f "$CONDA_BASHRC" ]; then
    echo "❌ 错误: Conda bashrc文件不存在: $CONDA_BASHRC"
    exit 1
fi

# 检查依赖
echo "🔍 检查依赖..."

# 检查tmux
if ! command -v tmux &> /dev/null; then
    echo "❌ 错误: 未安装tmux"
    echo "请运行: sudo apt install tmux"
    exit 1
fi

echo "✅ 依赖检查通过"

# 检查是否在跳板机(有网络)
echo ""
echo "🌐 检查网络连接..."
if curl -s --connect-timeout 5 https://api.wandb.ai > /dev/null; then
    echo "✅ 网络连接正常，可以同步到WandB"
else
    echo "⚠️ 无法连接到WandB API，请检查网络或代理设置"
    echo "如果您在训练机上，请在跳板机上运行此脚本"
fi

# 启动同步服务
echo ""
echo "🎯 启动同步服务..."

# 添加执行权限
chmod +x "$PYTHON_SCRIPT"

python3 "$PYTHON_SCRIPT" \
    --wandb-dir "$WANDB_DIR" \
    --session-name "$SESSION_NAME" \
    --sync-interval "$SYNC_INTERVAL" \
    --proxy-host "$PROXY_HOST" \
    --proxy-port "$PROXY_PORT" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-api-key "$WANDB_API_KEY" \
    --conda-bashrc "$CONDA_BASHRC" \
    --conda-env "$CONDA_ENV"

echo ""
echo "🎉 同步服务启动完成!"
echo ""
echo "💡 常用命令:"
echo "   查看同步状态: tmux attach -t $SESSION_NAME"
echo "   停止同步服务: tmux kill-session -t $SESSION_NAME"
echo "   查看日志文件: tail -f $SCRIPT_DIR/wandb_logs/wandb_sync.log"
echo "   查看同步状态文件: cat $SCRIPT_DIR/wandb_logs/.wandb_sync_state.json"
echo "   重置同步状态: python3 $PYTHON_SCRIPT --wandb-dir $WANDB_DIR --reset-sync-state"
echo "   列出所有tmux会话: tmux list-sessions"
echo ""
echo "🔧 优化说明:"
echo "   - 同步间隔改为1小时，只同步必要的.wandb文件"
echo "   - 在tmux会话中自动设置conda环境 ($CONDA_ENV)"
echo "   - 利用wandb原生增量同步能力，避免重复传输"
echo ""
echo "📋 要同步其他run，直接修改脚本中的WANDB_DIR路径即可"
echo ""