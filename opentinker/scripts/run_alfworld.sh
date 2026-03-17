#!/bin/bash
# ALFWorld Training Script (Multi-Turn)
#
# This script runs ALFWorld RL training with OpenTinker.
# You need to run these steps in SEPARATE terminals.
#
# For Training (3 terminals):
#   Terminal 1: bash run_alfworld.sh scheduler
#   Terminal 2: bash run_alfworld.sh env
#   Terminal 3: bash run_alfworld.sh client
#
# Prerequisites:
#   - pip install alfworld
#   - alfworld-download
#   - See docs/alfworld_multiturn.md for environment setup

# =============================================================================
# Configuration
# =============================================================================
SCHEDULER_PORT="${SCHEDULER_PORT:-9780}"
ENV_PORT="${ENV_PORT:-1234}"
GPUS="${GPUS:-[0,1,2,3]}"
MODEL_PATH="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct"

# OpenTinker root (relative to this script: opentinker/scripts/run_alfworld.sh)
OPENTINKER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export VLLM_DISABLE_SLEEP_MODE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# Activate conda environment (adjust to your setup if needed)
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
# conda activate opentinker

# Change to OpenTinker directory
cd "$OPENTINKER_ROOT"

# Get current host IP for communication between components
# Use 127.0.0.1 if running everything on the same machine
HOST_IP="${HOST_IP:-127.0.0.1}"

# =============================================================================
# Step Selection
# =============================================================================
case "$1" in
    scheduler|1)
        echo "========================================"
        echo "Step 1: Starting Scheduler on port $SCHEDULER_PORT"
        echo "========================================"
        bash opentinker/scripts/launch_scheduler.sh \
            --scheduler-port $SCHEDULER_PORT \
            --gpus "$GPUS"
        ;;

    env|2)
        echo "========================================"
        echo "Step 2: Starting ALFWorld Environment Server on port $ENV_PORT"
        echo "========================================"
        python -m opentinker.environment.alfworld.alfworld_server \
            --port "$ENV_PORT" \
            --max_steps 50 \
            --split train \
            --num_games -1 \
        ;;

    client|3)
        echo "========================================"
        echo "Step 3: Starting ALFWorld RL Client"
        echo "========================================"
        python opentinker/client/alfworld_rl.py \
            --config-name alfworld_wmc_erc_param \
            tokenizer_path="$MODEL_PATH" \
            batch_size=4 \
            val_batch_size=50 \
            num_steps=1000 \
            save_freq=2000 \
            test_freq=10 \
            scheduler_url="http://$HOST_IP:$SCHEDULER_PORT" \
            interaction.config.env_port="$ENV_PORT" \
            interaction.config.env_host="$HOST_IP"
        ;;

    *)
        echo "Usage: $0 {scheduler|env|client}"
        echo ""
        echo "Example (separate terminals):"
        echo "  Terminal 1: bash $0 scheduler"
        echo "  Terminal 2: bash $0 env"
        echo "  Terminal 3: bash $0 client"
        exit 1
        ;;
esac
