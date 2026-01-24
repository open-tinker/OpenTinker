#!/bin/bash
# ALFWorld Training & Inference Script
# 
# This script runs ALFWorld RL training or inference with OpenTinker.
# You need to run these steps in SEPARATE terminals.
#
# For Training (3 terminals):
#   Terminal 1: bash run_alfworld.sh scheduler
#   Terminal 2: bash run_alfworld.sh env
#   Terminal 3: bash run_alfworld.sh client
#
# For Inference/Evaluation (3 terminals):
#   Terminal 1: bash run_alfworld.sh scheduler
#   Terminal 2: bash run_alfworld.sh env-eval
#   Terminal 3: bash run_alfworld.sh inference model_path=/path/to/checkpoint

# =============================================================================
# Configuration
# =============================================================================
SCHEDULER_PORT=8089
ENV_PORT=8091
GPUS='[0,1,2,3,4,5,6,7,8,9]'
NUM_GPUS=4

# Fix vLLM v1 cumem allocator issue (V1 is required for async engine)
# Disable sleep mode to avoid cumem allocator CUDA errors
export VLLM_DISABLE_SLEEP_MODE=1

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate opentinker

# Change to OpenTinker directory
cd /home/haofeiy2/OpenTinker

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
        echo "Step 2: Starting ALFWorld Environment Server on ports $ENV_PORT-$((ENV_PORT+7)) (8 shards)"
        echo "========================================"
        python opentinker/environment/alfworld/alfworld_server.py \
            --port $ENV_PORT \
            --shards 8 \
            --split train \
            --max_steps 50
        ;;
    
    env-eval)
        echo "========================================"
        echo "Step 2 (Eval): Starting ALFWorld Environment Server for Evaluation"
        echo "========================================"
        python opentinker/environment/alfworld/alfworld_server.py \
            --port $ENV_PORT \
            --shards 1 \
            --split eval_in_distribution \
            --max_steps 50
        ;;
    
    client|3)
        echo "========================================"
        echo "Step 3: Running ALFWorld RL Client"
        echo "========================================"
        python opentinker/client/alfworld_rl.py \
            tokenizer_path=Qwen/Qwen2.5-3B-Instruct \
            batch_size=16 \
            val_batch_size=32 \
            num_epochs=5 \
            save_freq=10 \
            test_freq=100 \
            num_gpus=$NUM_GPUS \
            scheduler_url=http://0.0.0.0:$SCHEDULER_PORT \
            interaction.config.env_port=$ENV_PORT \
            interaction.config.env_host=0.0.0.0 \
            interaction.config.env_shards=8
        ;;
    
    inference|4)
        echo "========================================"
        echo "Step 3 (Inference): Running ALFWorld Evaluation"
        echo "========================================"
        # Pass remaining arguments (e.g., model_path=/path/to/checkpoint)
        shift  # Remove 'inference' from args
        python opentinker/client/alfworld_inference.py \
            num_gpus=$NUM_GPUS \
            scheduler_url=http://0.0.0.0:$SCHEDULER_PORT \
            env_endpoint=http://0.0.0.0:$ENV_PORT \
            split=eval_in_distribution \
            "$@"
        ;;
    
    *)
        echo "ALFWorld Training & Inference Script"
        echo ""
        echo "Usage: $0 {scheduler|env|env-eval|client|inference}"
        echo "       $0 {1|2|3|4}"
        echo ""
        echo "=== For Training (3 terminals) ==="
        echo "  Terminal 1: $0 scheduler   # Start scheduler (port $SCHEDULER_PORT)"
        echo "  Terminal 2: $0 env         # Start environment server (train split)"
        echo "  Terminal 3: $0 client      # Start RL training client"
        echo ""
        echo "=== For Inference/Evaluation (3 terminals) ==="
        echo "  Terminal 1: $0 scheduler   # Start scheduler (port $SCHEDULER_PORT)"
        echo "  Terminal 2: $0 env-eval    # Start environment server (eval split)"
        echo "  Terminal 3: $0 inference model_path=/path/to/checkpoint"
        echo ""
        echo "Inference options:"
        echo "  model_path=...       # Path to trained checkpoint (REQUIRED)"
        echo "  max_samples=N        # Limit evaluation samples"
        echo "  output_path=...      # Save results to file"
        echo "  split=...            # eval_in_distribution (default) or eval_out_of_distribution"
        echo ""
        echo "Configuration:"
        echo "  SCHEDULER_PORT=$SCHEDULER_PORT"
        echo "  ENV_PORT=$ENV_PORT"
        echo "  GPUS=$GPUS"
        echo "  NUM_GPUS=$NUM_GPUS"
        ;;
esac
