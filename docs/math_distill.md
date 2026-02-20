# On-Policy Distillation — Math (Single-Turn)

**Author:** OpenTinker

This example trains a student model with combined RL + knowledge distillation, using a teacher model (reference policy) as an anchor.

## Algorithm

At every training step the per-token advantage is:

```
A_final = A_base - α × (log π_student − log π_teacher)
```

| Term | Meaning |
|------|---------|
| `A_base` | Standard RL advantage from any estimator (`grpo`, `gae`, `grpo_per_step`) |
| `log π_student − log π_teacher` | Per-token reverse KL divergence |
| `α` (`kl_penalty_coef`) | How strongly to penalise divergence from the teacher |

The KL term is added **after** advantage normalisation, so the task reward signal keeps its scale and the distillation signal is independent.

**Teacher model** — by default the teacher is the initial student checkpoint (standard RLVR). Set `teacher_model_path` to a stronger or larger model to distil from it.

---

## Prerequisites

1. Complete the [Installation](../README.md#-installation) steps.
2. Get the server IP: `hostname -I`.

---

## Step 1: Start the Scheduler (Server Side)

```bash
bash opentinker/scripts/launch_scheduler.sh --scheduler-port <scheduler_port>
```

---

## Step 2: Start the Math Environment (Client Side)

```bash
python opentinker/environment/math/math_server.py --port <env_port>
```

---

## Step 3: Generate Training Data

```bash
python opentinker/data_preprocess/math_multiturn_w_interaction.py \
    --local_save_dir=<local_save_dir>
```

---

## Step 4: Run On-Policy Distillation Training

```bash
python opentinker/client/math_distill_rl.py \
    tokenizer_path=Qwen/Qwen2.5-1.5B \
    teacher_model_path=Qwen/Qwen2.5-7B-Instruct \
    batch_size=16 \
    rollout_n=8 \
    num_epochs=5 \
    save_freq=100 \
    test_freq=5 \
    adv_estimator=grpo \
    use_kl_in_advantage=true \
    kl_penalty_coef=0.1 \
    data_path=data/math_agentloop/train.parquet \
    val_data_path=data/math_agentloop/test.parquet \
    scheduler_url=http://<server_endpoint>:<scheduler_port> \
    interaction.config.env_port=<env_port> \
    interaction.config.env_host=<client_endpoint>
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `tokenizer_path` | — | Student model (HuggingFace path or local directory) |
| `teacher_model_path` | `null` | Teacher model path. `null` → use the student's initial weights as teacher (equivalent to pure RLVR) |
| `use_kl_in_advantage` | `true` | Enable distillation. Set to `false` for pure RL |
| `kl_penalty_coef` | `0.1` | α — weight of KL term. Higher → stay closer to teacher |
| `adv_estimator` | `grpo` | RL advantage estimator: `grpo`, `gae`, or `grpo_per_step` |
| `rollout_n` | `16` | Number of rollouts per prompt (for GRPO) |
| `batch_size` | `64` | Prompt batch size per training step |
| `num_gpus` | `4` | Number of GPUs for tensor parallelism |

### Effect of `kl_penalty_coef`

| Value | Behaviour |
|-------|-----------|
| `0.0` | Pure RL — identical to `math_rl.py` with `use_kl_in_advantage=false` |
| `0.05–0.1` | Mild distillation: explores freely but stays close to teacher |
| `0.5+` | Strong distillation: model stays very close to teacher distribution |

---

## Step 5: Disable distillation (pure RLVR baseline)

To run the same script without distillation (for ablation):

```bash
python opentinker/client/math_distill_rl.py \
    tokenizer_path=Qwen/Qwen2.5-1.5B \
    use_kl_in_advantage=false \
    adv_estimator=grpo \
    data_path=data/math_agentloop/train.parquet \
    val_data_path=data/math_agentloop/test.parquet \
    scheduler_url=http://<server_endpoint>:<scheduler_port> \
    interaction.config.env_port=<env_port> \
    interaction.config.env_host=<client_endpoint>
```

> When `use_kl_in_advantage=false`, `teacher_model_path` is ignored and no reference policy worker is created.

---

## Monitoring

Training logs `distill/kl_per_token` and `distill/kl_coef` to WandB/console at every step, in addition to the standard RL metrics.

| Metric | Meaning |
|--------|---------|
| `distill/kl_per_token` | Mean per-token KL divergence between student and teacher |
| `distill/kl_coef` | The α coefficient in use |
| `training_step_reward` | Mean task reward (math correctness) |
| `actor/entropy` | Student policy entropy |

---

## Implementation Notes

The distillation logic lives entirely in **opentinker** (no verl changes):

| File | Role |
|------|------|
| `opentinker/backend_patch/verl/trainer/ppo/per_step_core_algos.py` | `incorporate_kl_penalty_in_advantage()` — computes and adds the KL term |
| `opentinker/server/http_training_server.py` | Calls the function after `compute_advantage`; enables the reference policy worker when `use_kl_in_advantage=true` |
| `opentinker/client/math_distill_rl.py` | Client entry-point |
| `opentinker/client/client_config/math_distill_param.yaml` | Default config |
