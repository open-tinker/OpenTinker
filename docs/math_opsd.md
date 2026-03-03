# On-Policy Self-Distillation (OPSD) — Math (Single-Turn)

This recipe enables OPSD teacher-context distillation in OpenTinker while keeping the existing RL pipeline.

## What `algorithm.opsd.enable` does

When `algorithm.opsd.enable=true`, the server computes `ref_log_prob` using a privileged teacher context:

- CoT source is fixed to `extra_info.answer`
- Student rollout and `old_log_probs` stay unchanged
- The update still uses the existing KL-in-advantage path
- Teacher parameter mode is controlled by `algorithm.opsd.teacher_mode` (`fixed` or `shared`)

When `algorithm.opsd.enable=false`, behavior is unchanged from existing distill/RL logic.

## Teacher Parameter Mode

- `algorithm.opsd.teacher_mode=fixed` (default): use a frozen teacher/reference model.
- `algorithm.opsd.teacher_mode=shared`: use current student parameters as teacher for OPSD ref log-prob.

Notes:

- `shared` mode only changes OPSD teacher-context ref computation.
- If `teacher_model_path` is set, it is used by `fixed` mode.
- `shared` mode avoids standalone RefPolicy model loading when no other KL path requires it.

## Training modes

1. Pure RL: `use_kl_in_advantage=false`
2. RL + OPSD: `use_kl_in_advantage=true`, `disable_rl_reward=false`, `algorithm.opsd.enable=true`
3. Pure OPSD-style distill: `use_kl_in_advantage=true`, `disable_rl_reward=true`, `algorithm.opsd.enable=true`

## Run

### Example A: RL + OPSD (supports `rollout_n>1`)

```bash
python opentinker/client/math_opsd_rl.py \
  tokenizer_path=Qwen/Qwen3-4B-Instruct-2507 \
  batch_size=16 \
  rollout_n=4 \
  num_epochs=5 \
  test_freq=5 \
  data_path=data/math_agentloop/train.parquet \
  val_data_path=data/math_agentloop/test.parquet \
  scheduler_url=http://localhost:8780 \
  interaction.config.env_port=8082 \
  interaction.config.env_host=localhost \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_advantage=true \
  algorithm.disable_rl_reward=false \
  algorithm.kl_penalty_coef=0.1 \
  algorithm.opsd.enable=true \
  algorithm.opsd.teacher_mode=fixed
```

### Example B: Pure OPSD-style distill (`rollout_n=1`)

```bash
python opentinker/client/math_opsd_rl.py \
  tokenizer_path=Qwen/Qwen3-4B-Instruct-2507 \
  batch_size=32 \
  rollout_n=1 \
  num_epochs=5 \
  data_path=data/math_agentloop/train.parquet \
  val_data_path=data/math_agentloop/test.parquet \
  scheduler_url=http://localhost:8780 \
  interaction.config.env_port=8082 \
  interaction.config.env_host=localhost \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_advantage=true \
  algorithm.disable_rl_reward=true \
  algorithm.kl_penalty_coef=0.1 \
  algorithm.opsd.enable=true \
  algorithm.opsd.teacher_mode=fixed
```

### Example C: Existing behavior (OPSD off)

```bash
python opentinker/client/math_opsd_rl.py \
  tokenizer_path=Qwen/Qwen3-4B-Instruct-2507 \
  batch_size=16 \
  rollout_n=4 \
  data_path=data/math_agentloop/train.parquet \
  val_data_path=data/math_agentloop/test.parquet \
  scheduler_url=http://localhost:8780 \
  interaction.config.env_port=8082 \
  interaction.config.env_host=localhost \
  algorithm.use_kl_in_advantage=true \
  algorithm.disable_rl_reward=false \
  algorithm.opsd.enable=false
```

### Example D: OPSD with shared teacher parameters

```bash
python opentinker/client/math_opsd_rl.py \
  tokenizer_path=Qwen/Qwen3-4B-Instruct-2507 \
  batch_size=16 \
  rollout_n=4 \
  data_path=data/math_agentloop/train.parquet \
  val_data_path=data/math_agentloop/test.parquet \
  scheduler_url=http://localhost:8780 \
  interaction.config.env_port=8082 \
  interaction.config.env_host=localhost \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_advantage=true \
  algorithm.disable_rl_reward=false \
  algorithm.kl_penalty_coef=0.1 \
  algorithm.opsd.enable=true \
  algorithm.opsd.teacher_mode=shared
```

## Metrics

In addition to existing metrics, OPSD logs:

- `distill/opsd_active`
- `distill/opsd_teacher_shared`
- `distill/opsd_teacher_prompt_len_mean`
- `distill/opsd_teacher_prompt_len_max`
- `distill/opsd_teacher_prompt_len_min`
- `distill/opsd_missing_cot_count`
- `val/pass_at_{k}` when `val_n > 1` (`k = val_n`)

`val/pass_at_{k}` is computed in GRPO style (`best@k`):

- Group validation rollouts by prompt `uid`
- Take the max score within each group of size `k`
- Report the mean of those per-prompt best scores

You do not need to manually duplicate validation data `k` times. Validation auto-repeats each sample by `val_n`.
Validation sampling temperature can be set via `val_temperature`. This is also effective when `val_n = 1` (sampling is enabled when `val_temperature > 0`).
