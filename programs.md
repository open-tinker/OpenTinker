# ALFWorld 3B WMC-ERC Iteration Plan

## Objective

**Iterate on the WMC-ERC (World Model Correction) algorithm until it consistently beats baseline PPO in reward growth speed on ALFWorld.**

The first milestone is: **WMC-ERC reward increases faster than baseline PPO over 150 training steps.**

If it doesn't, we iterate on the WMC-ERC hyperparameters (mu_base, lambda_wm) and algorithm design until it does.

## Iteration Loop

```
1. Run baseline PPO → record reward curve (one-time)
2. Run WMC-ERC + PPO → record reward curve
3. Compare: does WMC-ERC reward grow faster?
   - YES → success, move to larger model / more steps
   - NO  → analyze why, adjust WMC-ERC params, go to step 2
```

### What to tune in WMC-ERC:

- `mu_base`: base clipping coefficient (controls tightness of the gate)
- `lambda_wm`: how much WM uncertainty widens the gate
- `entropy_target`, `base_entropy_coeff`: adaptive entropy control via beta_token
- Algorithm logic in `opentinker/backend_patch/verl/trainer/ppo/wmc_erc.py`

## Setup

- **Model**: Qwen/Qwen2.5-3B-Instruct
- **GPUs**: 0, 1 (2 GPUs)
- **Steps per experiment**: 150
- **Batch size**: 8 (reduced from 24 due to 3B model memory)
- **RL Algorithm**: PPO (adv_estimator: gae, rollout_n: 1)
- **Environment**: ALFWorld (alfworld_server, NOT sciworld_server)
- **Env shards**: 8 (8 parallel env server processes on ports 8092-8099)

---

## Full Restart Procedure

**IMPORTANT: Every time you restart, you MUST kill and restart ALL THREE services in this order:**

1. **Kill everything**
2. **Start Scheduler** (FIRST)
3. **Start ALFWorld Server** (SECOND)
4. **Start RL Training** (THIRD)

### Step 1: Kill Everything

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate opentinker

pkill -9 -f "alfworld_rl" -u haofeiy2
pkill -9 -f "launch_scheduler" -u haofeiy2
pkill -9 -f "ray::" -u haofeiy2
pkill -9 -f "alfworld_server" -u haofeiy2
ray stop --force
sleep 5

# Verify everything is dead
ps aux | grep -E "alfworld|scheduler|ray::" | grep -v grep | wc -l  # Should be 0
```

### Step 2: Start Scheduler (FIRST)

```bash
ROLLOUT_TRACE_DIR=./traces TORCH_CUDA_ARCH_LIST="9.0" FLASHINFER_HOMOGENEOUS_MS=1 \
nohup python opentinker/scheduler/launch_scheduler_kill.py \
  available_gpus='[0,1]' gpus_per_job=2 \
  port_range=null num_ports=200 scheduler_port=8780 \
  > /tmp/scheduler_8780.log 2>&1 &

sleep 15
curl -s http://0.0.0.0:8780/  # Should return scheduler JSON
```

### Step 3: Start ALFWorld Server (SECOND)

```bash
nohup python opentinker/environment/alfworld/alfworld_server.py \
  --port 8092 --shards 8 \
  > /tmp/alfworld_server.log 2>&1 &

sleep 10
curl -s http://0.0.0.0:8092/health  # Should return {"status":"healthy"}
```

### Step 4: Start RL Training (THIRD)

**Experiment A — WMC-ERC + PPO:**

```bash
python opentinker/client/alfworld_rl.py \
  --config-name alfworld_wmc_erc_param \
  tokenizer_path=Qwen/Qwen2.5-3B-Instruct \
  num_gpus=2 num_steps=150
```

**Experiment B — Baseline PPO (no WMC-ERC):**

```bash
python opentinker/client/alfworld_rl.py \
  --config-name alfworld_param \
  tokenizer_path=Qwen/Qwen2.5-3B-Instruct \
  num_gpus=2 num_steps=150
```

---

## Extract Reward Data

```bash
# Find the latest wandb output log
LOG=$(ls -t outputs/*/wandb/run-*/files/output.log | head -1)

# Extract reward per step
grep "training_step_reward" $LOG | grep -oP "training_step_reward:\K[^ ]+" | \
  awk '{n++; printf "step %2d: %+.3f\n", n, $0}'

# Extract completion rate
grep "game/completion_rate" $LOG | grep -oP "game/completion_rate:\K[^ ]+" | \
  awk '{n++; printf "step %2d: %.1f%%\n", n, $0*100}'
```

## Key Config Differences

| Setting                    | WMC-ERC (`alfworld_wmc_erc_param`) | Baseline (`alfworld_param`) |
| -------------------------- | ---------------------------------- | --------------------------- |
| batch_size                 | 8                                  | 8                           |
| wmc_erc.enable             | true                               | N/A                         |
| wmc_erc.mu_base            | 1.0                                | N/A                         |
| wmc_erc.lambda_wm          | 10.0                               | N/A                         |
| wmc_erc.entropy_target     | 2.0                                | N/A                         |
| wmc_erc.base_entropy_coeff | 0.02                               | N/A                         |
| wmc_erc.entropy_floor      | 0.1                                | N/A                         |
| entropy_coeff              | 0.0 (adaptive via beta_token)      | 0                           |
| adv_estimator              | gae (PPO)                          | gae (PPO)                   |
| rollout_n                  | 1                                  | 1                           |
| env_shards                 | 8                                  | 8                           |

## Previous 3B Results (for reference)

WMC-ERC with Qwen2.5-3B-Instruct (36 steps, grpo_per_step):

```
Step  1: -0.030    Step 10: +0.375    Step 20: +1.244    Step 30: +1.580
Step  5: -0.001    Step 15: +1.447    Step 25: +0.926    Step 36: +3.262
```

Reward grew from -0.03 to +3.26, completion rate reached ~35%.

## 0.5B Experiment Results (2026-03-16)

### WMC-ERC Run 5 (best): Adaptive Entropy via beta_token

Config: `mu_base=1.0, lambda_wm=10.0, entropy_floor=0.1, entropy_target=2.0, base_entropy_coeff=0.02, entropy_coeff=0.0`

```
Step   1: -1.335 (entropy: 1.96)    Step  50: -0.187 (entropy: 1.44)
Step  10: -1.051 (entropy: 2.17)    Step  90: -0.092 (entropy: 3.58)
Step  30: -0.344 (entropy: 1.05)    Step 150: -0.098 (entropy: 3.93)
```

### Baseline PPO (0.5B, 150 steps)

```
Step   1: -1.424 (entropy: 2.02)    Step  50: -0.160 (entropy: 0.02)
Step  10: -0.866 (entropy: 1.51)    Step  90: -0.130 (entropy: 0.32)
Step  30: -0.173 (entropy: 0.04)    Step 150: -0.073 (entropy: 0.01)
```

### 0.5B Conclusion

Both WMC-ERC and baseline PPO converged to similar rewards (~-0.07 to -0.10) with 0% completion rate. The 0.5B model lacks capacity for ALFWorld. Moving to 3B.

### Key Findings from 0.5B

1. **Entropy collapse** is the #1 failure mode for small models on ALFWorld
2. Fixed `entropy_coeff` is too blunt — too weak (0.005) doesn't prevent collapse, too strong (0.01) causes explosion
3. **Adaptive beta_token** (WMC-ERC v2) solves this: per-turn entropy → per-token entropy coefficient
4. `lambda_wm=10.0` needed because H_WM values are small (0.03-0.09)
5. Completion rate remained 0% across all 0.5B runs — model too small

### Algorithm Changes Made (from 0.5B iteration)

1. `wmc_erc.py`: Added `compute_per_turn_entropy()`, entropy floor gating, adaptive beta_token injection
2. `http_training_client.py`: Pass `wmc_erc` and `entropy_coeff` config to server
3. Config files: batch_size adjustable, steps 150

## Code Changes Made

1. `http_training_server.py`: Added `NCCL_CUMEM_ENABLE=0`, `VLLM_DISABLE_SLEEP_MODE=1`, `VLLM_GPU_MEMORY_UTILIZATION=0.25` to `ray.init()` runtime_env
2. `vllm_async_server.py`: Added env var override for `gpu_memory_utilization`
3. `job_scheduler.py`: pmon check disabled for GPU sharing; memory threshold adjustable
4. `alfworld_wmc_erc_param.yaml`: `adv_estimator` changed to `gae`, `rollout_n` to 1, `env_shards` to 8
5. `alfworld_param.yaml`: `adv_estimator` changed to `gae`, `rollout_n` to 1, `env_shards` to 8
6. `http_training_server.py`: Added `gc.collect()` + `torch.cuda.empty_cache()` after each training step

## 3B GRPO Baseline (2026-03-17)

**Setup**: GPU 2,7 | Scheduler port 8781 | ALFWorld server port 8100 | 150 steps
**Config**: `adv_estimator=grpo, rollout_n=4, batch_size=8, Qwen2.5-3B-Instruct`
**W&B**: `alfworld_grpo_baseline_3b` (run f1nf4cy1)
**Log**: `/tmp/alfworld_grpo_baseline.log`

### Results

_(pending — check with `tail -20 /tmp/alfworld_grpo_baseline.log`)_

---

## Troubleshooting

- **QUEUED forever**: Check `grep "OCCUPIED" /tmp/scheduler_8780.log` — GPU not free
- **cumem error**: Ensure `VLLM_DISABLE_SLEEP_MODE=1` in ray.init runtime_env
- **env_shards connection error**: Ensure env server started with `--shards 8` matching config `env_shards: 8`
- **Java zombie processes**: Run `pkill -9 -f scienceworld.jar` — sciworld server leaks Java
- **RAM OOM**: Check `free -g` and kill Java zombies if present
- **vLLM GPU OOM**: Lower `VLLM_GPU_MEMORY_UTILIZATION` (currently 0.25)
