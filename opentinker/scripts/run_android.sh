#!/bin/bash
# AndroidWorld Training Script (Multi-Turn, Multi-Emulator)
#
# This script runs AndroidWorld RL training with OpenTinker.
# You need to run these steps in SEPARATE terminals.
#
# For Training (4 terminals):
#   Terminal 1: bash run_android.sh scheduler
#   Terminal 2: bash run_android.sh simulator    # Android Emulator (start BEFORE env)
#   Terminal 3: bash run_android.sh env
#   Terminal 4: bash run_android.sh client
#
# Prerequisites:
#   - Android SDK, AVD "AndroidWorldAvd" (or set AVD_NAME), and emulator in PATH
#   - See docs/android_world_multiturn.md for environment setup

# =============================================================================
# Configuration
# =============================================================================
SCHEDULER_PORT=9780
ENV_PORT=9092
GPUS="${GPUS:-[0,1,2,3]}"
NUM_GPUS="${NUM_GPUS:-4}"  # For tensor_model_parallel_size (model spans N GPUs)

# Multi-emulator configuration
# Set NUM_EMULATORS to match NUM_GPUS for true parallelism
NUM_EMULATORS="${NUM_EMULATORS:-4}"

# Emulator (simulator) base ports
AVD_NAME="${AVD_NAME:-AndroidWorldAvd}"
# Console ports: 5556, 5558, 5560, 5562 (each +2 because ADB uses console+1)
EMULATOR_BASE_CONSOLE_PORT="${EMULATOR_BASE_CONSOLE_PORT:-5556}"
# gRPC ports: 8554, 8555, 8556, 8557
EMULATOR_BASE_GRPC_PORT="${EMULATOR_BASE_GRPC_PORT:-8554}"
# EMULATOR_HEADLESS=1  -> -no-window -no-audio
EMULATOR_HEADLESS="${EMULATOR_HEADLESS:-1}"
# EMULATOR_NO_KVM=1    -> no "sg kvm", add -accel off (slow, for hosts without KVM)
EMULATOR_NO_KVM="${EMULATOR_NO_KVM:-0}"

# Fix vLLM v1 cumem allocator issue
export VLLM_DISABLE_SLEEP_MODE=1

# Model path (set to your model path)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"

# OpenTinker root (relative to this script: opentinker/scripts/run_android.sh)
OPENTINKER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Activate conda environment (adjust to your setup)
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
# conda activate <your_env_name>

# Change to OpenTinker directory
cd "$OPENTINKER_ROOT"

# =============================================================================
# Step Selection
# =============================================================================
case "$1" in
    setup-avds)
        echo "========================================"
        echo "Creating $NUM_EMULATORS AVDs for parallel training"
        echo "========================================"
        echo ""
        echo "This will create AVDs named: ${AVD_NAME}_0, ${AVD_NAME}_1, ..., ${AVD_NAME}_$((NUM_EMULATORS-1))"
        echo ""
        
        # Detect system image (x86_64 or arm64-v8a)
        SYSTEM_IMAGE="${SYSTEM_IMAGE:-system-images;android-33;google_apis;x86_64}"
        echo "Using system image: $SYSTEM_IMAGE"
        echo ""
        
        for i in $(seq 0 $((NUM_EMULATORS - 1))); do
            AVD_NAME_I="${AVD_NAME}_${i}"
            echo "Creating AVD: $AVD_NAME_I"
            
            # Check if AVD already exists
            if avdmanager list avd | grep -q "Name: $AVD_NAME_I"; then
                echo "  AVD $AVD_NAME_I already exists, skipping..."
            else
                echo "no" | avdmanager create avd \
                    --name "$AVD_NAME_I" \
                    --package "$SYSTEM_IMAGE" \
                    --device "pixel_6" \
                    --force
                echo "  Created $AVD_NAME_I"
            fi
        done
        
        echo ""
        echo "Done! Created $NUM_EMULATORS AVDs."
        echo "You can now run: bash run_android.sh simulator"
        ;;

    scheduler|1)
        echo "========================================"
        echo "Step 1: Starting Scheduler on port $SCHEDULER_PORT"
        echo "========================================"
        bash opentinker/scripts/launch_scheduler.sh \
            --scheduler-port $SCHEDULER_PORT \
            --gpus "$GPUS"
        ;;

    simulator|2)
        echo "========================================"
        echo "Step 2: Starting $NUM_EMULATORS Android Emulators"
        echo "  AVD base name: ${AVD_NAME}_0 ... ${AVD_NAME}_$((NUM_EMULATORS-1))"
        echo "  Base Console Port=$EMULATOR_BASE_CONSOLE_PORT"
        echo "  Base gRPC Port=$EMULATOR_BASE_GRPC_PORT"
        echo "========================================"
        echo ""
        echo "IMPORTANT: Before starting, ensure NO other emulators are running!"
        echo "  Check: adb devices"
        echo "  Kill all emulators: adb emu kill (or close emulator windows)"
        echo ""
        echo "Starting $NUM_EMULATORS emulators:"
        for i in $(seq 0 $((NUM_EMULATORS - 1))); do
            CONSOLE_PORT=$((EMULATOR_BASE_CONSOLE_PORT + i * 2))
            GRPC_PORT=$((EMULATOR_BASE_GRPC_PORT + i))
            echo "  Emulator $i: console=$CONSOLE_PORT (ADB=$((CONSOLE_PORT + 1))), grpc=$GRPC_PORT"
        done
        echo ""
        echo "Ensure the env server is started AFTER all emulators are fully booted."
        echo ""

        # Check if AVDs exist
        echo "Checking AVDs..."
        for i in $(seq 0 $((NUM_EMULATORS - 1))); do
            AVD_NAME_I="${AVD_NAME}_${i}"
            if ! avdmanager list avd 2>/dev/null | grep -q "Name: $AVD_NAME_I"; then
                echo "ERROR: AVD '$AVD_NAME_I' not found!"
                echo "Run 'bash run_android.sh setup-avds' first to create the AVDs."
                exit 1
            fi
        done
        echo "All AVDs found."
        echo ""

        # Start all emulators in background
        PIDS=()
        for i in $(seq 0 $((NUM_EMULATORS - 1))); do
            AVD_NAME_I="${AVD_NAME}_${i}"
            CONSOLE_PORT=$((EMULATOR_BASE_CONSOLE_PORT + i * 2))
            GRPC_PORT=$((EMULATOR_BASE_GRPC_PORT + i))
            
            BASE="emulator -avd $AVD_NAME_I -no-snapshot -port $CONSOLE_PORT -grpc $GRPC_PORT"
            if [ "$EMULATOR_NO_KVM" = "1" ]; then
                CMD="$BASE -no-window -no-audio -accel off"
            elif [ "$EMULATOR_HEADLESS" = "1" ]; then
                CMD="$BASE -no-window -no-audio"
            else
                CMD="$BASE"
            fi
            
            echo "Starting emulator $i ($AVD_NAME_I): $CMD"
            if [ "$EMULATOR_NO_KVM" = "1" ]; then
                $CMD &
            else
                sg kvm -c "$CMD" &
            fi
            PIDS+=($!)
            sleep 2  # Wait a bit between emulator starts
        done
        
        echo ""
        echo "All $NUM_EMULATORS emulators started. PIDs: ${PIDS[*]}"
        echo "Waiting for all emulators... Press Ctrl+C to stop."
        wait
        ;;

    env|3)
        echo "========================================"
        echo "Step 3: Starting AndroidWorld Environment Server"
        echo "  Shards: $NUM_EMULATORS (ports $ENV_PORT..$((ENV_PORT + NUM_EMULATORS - 1)))"
        echo "  Emulator base ports: console=$EMULATOR_BASE_CONSOLE_PORT, grpc=$EMULATOR_BASE_GRPC_PORT"
        echo "========================================"
        echo "Make sure all $NUM_EMULATORS Android Emulators are running first."
        echo ""
        python opentinker/environment/android_world/android_world_server.py \
            --port $ENV_PORT \
            --shards $NUM_EMULATORS \
            --emulator_base_console_port $EMULATOR_BASE_CONSOLE_PORT \
            --emulator_base_grpc_port $EMULATOR_BASE_GRPC_PORT \
            --split train \
            --max_steps 50
        ;;

    client|4)
        echo "========================================"
        echo "Step 4: Running AndroidWorld RL Client"
        echo "  Emulators: $NUM_EMULATORS (parallel rollouts)"
        echo "  GPUs: $NUM_GPUS (tensor parallelism)"
        echo "========================================"
        # Multi-emulator parallel training:
        # - batch_size=NUM_GPUS: satisfies batch_size >= num_gpus for data partitioning
        # - agent_num_workers=NUM_EMULATORS: parallel rollouts (one per emulator)
        # - env_shards=NUM_EMULATORS: routes requests to different env servers/emulators
        # - num_gpus=NUM_GPUS: model tensor parallelism (solves OOM)
        python opentinker/client/android_world_rl.py \
            tokenizer_path=$MODEL_PATH \
            batch_size=$NUM_GPUS \
            val_batch_size=$NUM_GPUS \
            rollout_n=1 \
            adv_estimator=gae \
            agent_num_workers=$NUM_EMULATORS \
            num_steps=1000 \
            save_freq=50 \
            test_freq=10 \
            num_gpus=$NUM_GPUS \
            scheduler_url=http://0.0.0.0:$SCHEDULER_PORT \
            interaction.config.env_port=$ENV_PORT \
            interaction.config.env_host=0.0.0.0 \
            interaction.config.env_shards=$NUM_EMULATORS
        ;;

    *)
        echo "AndroidWorld Training Script (Multi-Turn, Multi-Emulator)"
        echo ""
        echo "Usage: $0 {setup-avds|scheduler|simulator|env|client}"
        echo "       $0 {1|2|3|4}"
        echo ""
        echo "=== First Time Setup ==="
        echo "  $0 setup-avds            # Create $NUM_EMULATORS AVDs for parallel training"
        echo ""
        echo "=== For Training (4 terminals) ==="
        echo "  Terminal 1: $0 scheduler   # Start scheduler (port $SCHEDULER_PORT)"
        echo "  Terminal 2: $0 simulator   # Start $NUM_EMULATORS Android Emulators (start BEFORE env)"
        echo "  Terminal 3: $0 env         # Start $NUM_EMULATORS env server shards (ports $ENV_PORT..$((ENV_PORT+NUM_EMULATORS-1)))"
        echo "  Terminal 4: $0 client      # Start RL training client"
        echo ""
        echo "Multi-Emulator Configuration (env vars):"
        echo "  NUM_EMULATORS=$NUM_EMULATORS        # Number of parallel emulators"
        echo "  AVD_NAME=$AVD_NAME            # AVD base name (creates ${AVD_NAME}_0, ${AVD_NAME}_1, ...)"
        echo "  EMULATOR_BASE_CONSOLE_PORT=$EMULATOR_BASE_CONSOLE_PORT   # Base console port"
        echo "  EMULATOR_BASE_GRPC_PORT=$EMULATOR_BASE_GRPC_PORT      # Base gRPC port"
        echo "  EMULATOR_HEADLESS=1       # Headless: -no-window -no-audio"
        echo "  EMULATOR_NO_KVM=1         # No KVM: -accel off (slow, for containers/no KVM)"
        echo ""
        echo "IMPORTANT: Before running, ensure no other emulators are running!"
        echo "  Check: adb devices"
        echo "  Kill all: adb emu kill"
        echo ""
        echo "Configuration:"
        echo "  SCHEDULER_PORT=$SCHEDULER_PORT"
        echo "  ENV_PORT=$ENV_PORT"
        echo "  NUM_EMULATORS=$NUM_EMULATORS"
        echo "  GPUS=$GPUS"
        echo "  NUM_GPUS=$NUM_GPUS"
        echo "  MODEL_PATH=$MODEL_PATH"
        echo ""
        echo "See docs/android_world_multiturn.md for Android SDK and AVD setup."
        ;;
esac
