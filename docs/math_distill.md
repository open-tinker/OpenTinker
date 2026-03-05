# On-Policy Distillation — Math (Single-Turn)

**Author:** OpenTinker

This example trains a student model with combined RL + knowledge distillation, using a teacher model (reference policy) as an anchor.

## Algorithm

At every training step, token-level OPD uses:

```
A_teacher = log π_teacher(a_t) - log π_student(a_t)
A_final = A_base + A_teacher
L = L_pg + β_KL * L_KL
```

| Term | Meaning |
|------|---------|
| `A_base` | Standard RL advantage from any estimator (`grpo`, `gae`, `grpo_per_step`) |
| `A_teacher` | Token-level teacher signal used by policy gradient |
| `β_KL` (`kl_penalty_coef`) | External KL loss coefficient (`actor.kl_loss_coef`) |

When `disable_rl_reward=true`, `A_base` is zeroed and the policy-gradient term becomes teacher-signal-only (`A_final = A_teacher`).

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

**Example: train on retool_dapo, evaluate on AIME2024 with pass@32**

```bash
python opentinker/client/math_distill_rl.py \
    tokenizer_path=Qwen/Qwen3-4B-Instruct-2507 \
    teacher_model_path=Qwen/Qwen3-8B \
    batch_size=16 \
    rollout_n=4 \
    num_epochs=5 \
    save_freq=100 \
    test_freq=5 \
    adv_estimator=grpo \
    use_kl_in_advantage=true \
    kl_penalty_coef=1 \
    data_path=data/opsd_math/train_openthoughts_math.parquet \
    val_data_path=data/opsd_math/test_aime25.parquet \
    val_n=1 \
    val_batch_size=30 \
    scheduler_url=http://localhost:8781 \
    interaction.config.env_port=8083 \
    interaction.config.env_host=localhost
```

**Example: train/val on the original math dataset**

```bash
python opentinker/client/math_distill_rl.py \
    tokenizer_path=Qwen/Qwen2.5-1.5B \
    teacher_model_path=Qwen/Qwen2.5-7B-Instruct \
    batch_size=32 \
    rollout_n=1 \
    adv_estimator=gae \
    num_epochs=5 \
    save_freq=100 \
    test_freq=5 \
    use_kl_in_advantage=true \
    disable_rl_reward=true \
    kl_penalty_coef=1 \
    data_path=data/math/train.parquet \
    val_data_path=data/math/test.parquet \
    scheduler_url=http://localhost:8780 \
    interaction.config.env_port=8082 \
    interaction.config.env_host=localhost
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `tokenizer_path` | — | Student model (HuggingFace path or local directory) |
| `teacher_model_path` | `null` | Teacher model path. `null` → use the student's initial weights as teacher (equivalent to pure RLVR) |
| `use_kl_in_advantage` | `true` | Enable token-level OPD teacher signal in policy gradient. Set to `false` for pure RL. similar parameters in verl: https://github.com/verl-project/verl/issues/3276 |
| `disable_rl_reward` | `false` | Disable RL reward signal: zero out base advantages before adding teacher signal. When true (requires `use_kl_in_advantage=true`), `A_final = A_teacher`. |
| `kl_penalty_coef` | `0.1` | `β_KL` for external actor KL loss (outside policy gradient). Higher → stay closer to teacher |
| `adv_estimator` | `grpo` | RL advantage estimator: `grpo`, `gae`, or `grpo_per_step` |
| `rollout_n` | `16` | Number of rollouts per prompt (for GRPO) |
| `batch_size` | `64` | Prompt batch size per training step |
| `num_gpus` | `4` | Number of GPUs for tensor parallelism |
| `val_n` | `1` | Rollouts per validation prompt. Set to `32` for pass@32. When `>1`, the val dataset is automatically deduplified to unique prompts |
| `val_batch_size` | `30` | Number of unique validation prompts per evaluation |

### Effect of `kl_penalty_coef`

| Value | Behaviour |
|-------|-----------|
| `0.0` | No external KL regularizer (`L_KL` disabled), while teacher-signal PG still applies when `use_kl_in_advantage=true` |
| `0.05–0.1` | Mild external KL regularization: explores freely but stays close to teacher |
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
| `distill/kl_coef` | External KL loss coefficient (`β_KL`) in use |
| `training_step_reward` | Mean task reward (math correctness) |
| `actor/entropy` | Student policy entropy |
| `val/pass_at_32` | Fraction of problems solved by at least one of 32 rollouts (only logged when `val_n=32`) |
| `val/mean_score` | Mean reward across all validation rollouts |

---

## Implementation Notes

The distillation logic lives entirely in **opentinker** (no verl changes):

| File | Role |
|------|------|
| `opentinker/backend_patch/verl/trainer/ppo/per_step_core_algos.py` | `incorporate_teacher_signal_in_advantage()` — computes and adds token-level teacher signal |
| `opentinker/server/http_training_server.py` | Calls the function after `compute_advantage`; maps `kl_penalty_coef` to actor KL loss (`actor.kl_loss_coef`) |
| `opentinker/client/math_distill_rl.py` | Client entry-point |
| `opentinker/client/client_config/math_distill_param.yaml` | Default config |
