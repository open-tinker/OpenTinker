# OPSD: On-Policy Self-Distillation for Math

## Overview

OPSD trains a **single model** as both teacher and student, using only **full-parameter RL** (no LoRA, no external teacher). The same model weights see two different prompts:

| Role    | Prompt                                         | Gradient |
|---------|------------------------------------------------|----------|
| Student | `Problem: <q>\nAnswer:`                        | ✅ yes   |
| Teacher | `Problem: <q>\n\nHere is a reference solution:\n<y*>\nAfter understanding the reference solution, please try to solve this problem using your own approach below:` | ❌ detached |

**Loss**: token-averaged symmetric JSD between the two distributions, computed over the **full vocabulary**:

```
JSD_β(pT ‖ pS) = β · KL(pT ‖ m) + (1-β) · KL(pS ‖ m),   m = β·pT + (1-β)·pS
```

Default `β = 0.5` (symmetric JSD). Computed in log-space for numerical stability.

## Architecture Changes

| File | Change |
|------|--------|
| `per_step_core_algos.py` | Added `compute_opsd_jsd_loss()` |
| `opentinker/backend_patch/verl/workers/opsd_worker.py` | New `OPSDActorRolloutRefWorker` subclass with `compute_opsd_vocab_log_probs` RPC (full vocab log-softmax at every response position) |
| `http_training_server.py` | Added `train_step_opsd()`, `_build_teacher_inputs()`, `/api/v1/train_step_opsd` endpoint; selects `OPSDActorRolloutRefWorker` when `algorithm.use_opsd=True` |
| `client/math_opsd_rl.py` | New client entry-point; overrides `train_step` to call `/api/v1/train_step_opsd` |
| `client/client_config/math_opsd_param.yaml` | New config (no teacher model path, adds `opsd_beta`) |

## Running OPSD Training

### Requirements

- A math dataset with `prompt` and `ground_truth` columns (same format as RLVR).
- The `raw_prompt` field is required in the batch for teacher prompt construction (automatically included by `DynamicGameDataset`).

### Quick Start

```bash
python opentinker/client/math_opsd_rl.py \
    tokenizer_path=/path/to/model \
    data_path=/path/to/train.parquet \
    val_data_path=/path/to/test.parquet \
    scheduler_url=http://localhost:8780 \
    num_gpus=8 \
    opsd_beta=0.5 \
    temperature=0.7 \
    max_new_tokens=8192 \
    num_steps=500
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `opsd_beta` | `0.5` | JSD mixture weight β (0.5 = symmetric) |
| `rollout_n` | `1` | Rollouts per prompt (1 is typical for OPSD) |
| `temperature` | `0.7` | Sampling temperature for student rollout |
| `max_prompt_tokens` | `4096` | Max tokens; teacher prompt is left-truncated to this |
| `val_n` | `1` | Rollouts per val problem (`>1` enables pass@k) |

### Monitoring

Key metrics logged to WandB:

| Metric | Description |
|--------|-------------|
| `opsd/jsd_per_token` | Mean JSD per response token (should decrease during training) |
| `opsd/jsd_beta` | β value (informational) |
| `actor/pg_loss` | Policy gradient loss via update_actor |
| `actor/grad_norm` | Gradient norm |

## Differences from `math_distill_rl` (external teacher)

| Aspect | `math_distill_rl` | `math_opsd_rl` (this) |
|--------|-------------------|----------------------|
| Teacher model | Separate checkpoint | Same model, different prompt |
| Loss | KL subtracted from GRPO advantage | Pure full-vocab JSD |
| RefPolicy worker | Yes (loads teacher weights) | No |
| Data required | `prompt` | `prompt` + `ground_truth` |
| LoRA support | Yes | Not required (full-param) |

## Data Format

Same parquet format as standard RLVR training:

```python
{
    "prompt": [{"role": "user", "content": "What is 2+2?"}],
    "ground_truth": "4"
}
```

The `ground_truth` is automatically used to construct the teacher prompt at training time.

## Testing

Unit test for the JSD loss function:

```bash
pytest opentinker/tests/test_per_step_core_algos.py -v -k "opsd"
```

运行方式
模式1：RL + JSD（需要 math env server）
Step 1：启动 scheduler
bash
bash opentinker/scripts/launch_scheduler.sh --scheduler-port 8780
Step 2：启动 math env server
bash
python opentinker/environment/math/math_server.py --port 8082
Step 3：训练
bash
python opentinker/client/math_opsd_rl.py \
    tokenizer_path=Qwen/Qwen3-8B \
    data_path=data/math/train.parquet \
    val_data_path=data/math/test_by_level/test_level_5.parquet \
    rollout_n=4 \
    opsd_jsd_coef=0.5 \
    opsd_beta=0.5 \
    scheduler_url=http://localhost:8780 \
    interaction.config.env_port=8082 \
    interaction.config.env_host=localhost
模式2：纯 JSD（不需要 env server）
bash
python opentinker/client/math_opsd_rl.py \
    tokenizer_path=Qwen/Qwen2.5-3B-Instruct \
    data_path=data/math/train.parquet \
    val_data_path=data/math/test_by_level/test_level_4.parquet \
    rollout_n=1 \
    opsd_beta=0.5 \
    scheduler_url=http://localhost:8780 \
    interaction.config.env_port=8082 \
    interaction.config.env_host=localhost
