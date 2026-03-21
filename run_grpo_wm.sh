#!/usr/bin/env bash
# GRPO + World Model SFT loss
# Usage: ./run_grpo_wm.sh [gpus] [scheduler_port] [env_port] [steps] [wm_coeff]
exec ./run.sh alfworld_param "${1:-1,9}" "${2:-8782}" "${3:-8120}" "${4:-1000}" Qwen/Qwen2.5-3B-Instruct grpo_wm "${5:-0.01}"
