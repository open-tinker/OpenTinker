#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# GRPO baseline
# Usage: ./run_grpo.sh [gpus] [scheduler_port] [env_port] [steps]
exec ./run.sh alfworld_param "${1:-2,6}" "${2:-8782}" "${3:-8120}" "${4:-1000}" Qwen/Qwen2.5-3B-Instruct grpo
