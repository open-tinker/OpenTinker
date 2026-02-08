#!/bin/bash
# Convenience script to launch the job scheduler

# Set CUDA 12.8 environment explicitly
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export ROLLOUT_TRACE_DIR="opentinker/traces"
export NVCC_EXECUTABLE=$CUDA_HOME/bin/nvcc
export TORCH_CUDA_ARCH_LIST="9.0"
export FLASHINFER_HOMOGENEOUS_MS=1

# Default configuration
AVAILABLE_GPUS="[0,1,2,3,4,5,6,7,8,9]"
PORT_RANGE="null"  # Set to null for auto-detection
NUM_PORTS=200
SCHEDULER_PORT=8780

# Writable output/log directories (override via env)
RUNS_BASE_DIR="${RUNS_BASE_DIR:-opentinker/outputs}"
LOGS_DIR="${LOGS_DIR:-/tmp/opentinker/logs}"
HYDRA_RUN_DIR="${RUNS_BASE_DIR}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"

mkdir -p "$RUNS_BASE_DIR" "$LOGS_DIR"

# Parse command line arguments (optional)
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            AVAILABLE_GPUS="$2"
            shift 2
            ;;
        --ports)
            PORT_RANGE="$2"
            shift 2
            ;;
        --num-ports)
            NUM_PORTS="$2"
            shift 2
            ;;
        --scheduler-port)
            SCHEDULER_PORT="$2"
            shift 2
            ;;
        --auto-ports)
            PORT_RANGE="null"
            shift 1
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift 1
            ;;
    esac
done

echo "========================================"
echo "Launching Job Scheduler"
echo "========================================"
echo "Available GPUs: $AVAILABLE_GPUS"
if [ "$PORT_RANGE" = "null" ]; then
    echo "Port mode: Auto-detect ($NUM_PORTS ports)"
else
    echo "Port range: $PORT_RANGE"
fi
echo "Scheduler port: $SCHEDULER_PORT"
echo "========================================"
echo ""

# Launch scheduler
if [ "$PORT_RANGE" = "null" ]; then
    python opentinker/scheduler/launch_scheduler_kill.py \
        available_gpus=$AVAILABLE_GPUS \
        port_range=null \
        num_ports=$NUM_PORTS \
        scheduler_port=$SCHEDULER_PORT \
        logs_dir="$LOGS_DIR" \
        hydra.run.dir="$HYDRA_RUN_DIR" \
        "${EXTRA_ARGS[@]}"
else
    python opentinker/scheduler/launch_scheduler_kill.py \
        available_gpus=$AVAILABLE_GPUS \
        port_range=$PORT_RANGE \
        scheduler_port=$SCHEDULER_PORT \
        logs_dir="$LOGS_DIR" \
        hydra.run.dir="$HYDRA_RUN_DIR" \
        "${EXTRA_ARGS[@]}"
fi
