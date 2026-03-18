#!/usr/bin/env bash
# Usage: ./run.sh [config] [gpus] [scheduler_port] [env_port] [steps] [model] [mode] [wm_coeff]
# Example:
#   ./run.sh                                          # all defaults
#   ./run.sh alfworld_param 4,5 8782 8110 150
#   ./run.sh alfworld_param 4,5 8782 8110 150 Qwen/Qwen2.5-3B-Instruct grpo_wm 0.1
# Modes:
#   grpo     : standard GRPO
#   grpo_wm  : GRPO + world model SFT loss (adds +world_model_coeff=wm_coeff)
set -euo pipefail

CONFIG="${1:-alfworld_param}"
RAW_GPUS="${2:-4,6}"
SCHEDULER_PORT="${3:-8782}"
ENV_PORT="${4:-8120}"
STEPS="${5:-600}"
MODEL="${6:-Qwen/Qwen2.5-3B-Instruct}"
MODE="${7:-grpo}"
WM_COEFF="${8:-0.1}"

# Normalize GPU list so Hydra always receives a clean list override.
# Accepted input forms: "4,6" or "[4,6]".
GPUS="${RAW_GPUS// /}"
GPUS="${GPUS#[}"
GPUS="${GPUS%]}"
if [[ -z "$GPUS" || ! "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Invalid GPU list: '$RAW_GPUS'"
  echo "Expected format: 4,6 (or [4,6])"
  exit 1
fi
NUM_GPUS=$(awk -F',' '{print NF}' <<< "$GPUS")
SCHEDULER_GPU_OVERRIDE="available_gpus=[${GPUS}]"

# conda.sh may reference PS1; in non-interactive shells PS1 can be unset.
set +u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate opentinker
set -u
cd "$(dirname "$0")"

EXTRA_HYDRA_ARGS=()
MODE_TAG="grpo"
case "$MODE" in
  grpo)
    ;;
  grpo_wm|grpo+wm|wm|wm_sft)
    EXTRA_HYDRA_ARGS+=("+world_model_coeff=${WM_COEFF}")
    MODE_TAG="grpo_wm_${WM_COEFF}"
    ;;
  *)
    echo "Unsupported mode: $MODE"
    echo "Supported modes: grpo | grpo_wm"
    exit 1
    ;;
esac

# vLLM / NCCL fixes (cumem allocator crash)
export VLLM_DISABLE_SLEEP_MODE=1
export NCCL_CUMEM_ENABLE=0
export VLLM_GPU_MEMORY_UTILIZATION=0.25

# Step 1: Scheduler
echo "=== Step 1: Scheduler (GPUs=[$GPUS], port=$SCHEDULER_PORT) ==="
if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"${SCHEDULER_PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "Scheduler port ${SCHEDULER_PORT} is already in use."
  echo "Stop old scheduler first, or choose another port."
  echo "Hint: lsof -iTCP:${SCHEDULER_PORT} -sTCP:LISTEN -P -n"
  exit 1
fi
echo "Scheduler override: ${SCHEDULER_GPU_OVERRIDE}"
ROLLOUT_TRACE_DIR=./traces TORCH_CUDA_ARCH_LIST="9.0" FLASHINFER_HOMOGENEOUS_MS=1 \
nohup python opentinker/scheduler/launch_scheduler_kill.py \
  "${SCHEDULER_GPU_OVERRIDE}" gpus_per_job="${NUM_GPUS}" \
  port_range=null num_ports=200 scheduler_port="${SCHEDULER_PORT}" \
  > /tmp/scheduler_${SCHEDULER_PORT}.log 2>&1 &
echo "PID: $!"
sleep 12
curl -sf http://0.0.0.0:${SCHEDULER_PORT}/ > /dev/null && echo "OK" || { echo "FAIL"; exit 1; }

# Step 2: ALFWorld server
echo "=== Step 2: ALFWorld server (port=$ENV_PORT, shards=8) ==="
# Kill any stale shard processes on our port range
for ((p=ENV_PORT; p<ENV_PORT+8; p++)); do
  fuser -k "${p}/tcp" 2>/dev/null || true
done
sleep 1
nohup python opentinker/environment/alfworld/alfworld_server.py \
  --port "${ENV_PORT}" --shards 8 \
  > /tmp/alfworld_server_${ENV_PORT}.log 2>&1 &
echo "PID: $!"
sleep 30
curl -sf http://0.0.0.0:${ENV_PORT}/health > /dev/null && echo "OK" || { echo "FAIL"; exit 1; }

# Step 3: RL training
LOG="/tmp/${CONFIG}_${MODE_TAG}_p${SCHEDULER_PORT}.log"
echo "=== Step 3: Training (config=$CONFIG, mode=$MODE, gpus=$NUM_GPUS, steps=$STEPS) ==="
nohup python opentinker/client/alfworld_rl.py \
  --config-name "${CONFIG}" \
  tokenizer_path="${MODEL}" \
  num_gpus="${NUM_GPUS}" num_steps="${STEPS}" \
  scheduler_url="http://0.0.0.0:${SCHEDULER_PORT}" \
  interaction.config.env_port="${ENV_PORT}" \
  "${EXTRA_HYDRA_ARGS[@]}" \
  > "$LOG" 2>&1 &
echo "PID: $! | Log: $LOG"
echo "=== Done. tail -f $LOG ==="
