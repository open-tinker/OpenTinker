# RLTF-SD in OpenTinker

This document describes the RLTF-SD integration that is implemented **only in OpenTinker** (no edits in `verl/`).

## 1. Method Mapping

Implemented loop (strict same-state dual sampling) per decision step `t`:

1. Sample draft `y0` from current state `x_t`.
2. Generate critique `c_t` and draft score `r0` **without stepping environment**.
3. Build revised context `x'_t = f(x_t, y0, c_t)` and sample revised action `y1`.
4. Execute **only `y1`** in environment and get real reward `r1`.

Logged pair metadata (`rltf_sd_pair`) includes:

- `group_id`
- `turn_id`
- `r0`
- `r1`
- `x_t_token_ids`
- `y1_token_ids`

Auxiliary SD update:

- First-turn baseline per group: `b(0) = mean(r0)`
- SD advantage: `A_SD = r1 - b(0)` (with optional `gamma^(turn_id-1)` factor)
- `algorithm.rltf_sd.loss_type=awr`:
  - Auxiliary actor loss (AWR/NLL style):
  - `L_sd = -E[A_SD * log pi(y1 | x_t)]`
- `algorithm.rltf_sd.loss_type=kl`:
  - Pure distillation between student/teacher contexts:
  - student: `pi(· | x_t)`, teacher: `pi(· | x'_t)`
  - Distill mode: `topk_reverse_kl_tail` or `full_vocab_jsd`

The SD update is appended after normal actor update when enabled.

## 2. What Changed

Added files:

- `opentinker/server/rltf_sd_agent_loop.py`
- `opentinker/server/rltf_sd_feedback.py`
- `opentinker/server/rltf_sd_utils.py`
- `opentinker/client/client_config/math_rltf_sd_param.yaml`
- `opentinker/client/client_config/math_rltf_sd_kl_param.yaml`
- `opentinker/client/client_config/gomoku_rltf_sd_param.yaml`
- `opentinker/client/client_config/alfworld_rltf_sd_param.yaml`

Modified files:

- `opentinker/server/agent.yaml` (registers `rltf_sd_agent`)
- `opentinker/server/http_training_server.py` (optional SD branch)
- `opentinker/backend_patch/verl/workers/fsdp_workers.py` (adds `update_actor_rltf_sd`)
- `opentinker/client/utils/http_training_client.py` (supports `server_overrides` merge)

## 3. New Config Namespace

All new options are additive and default-off:

- `algorithm.rltf_sd.enable` (default `false`)
- `algorithm.rltf_sd.loss_type` (`awr` or `kl`, default `awr`)
- `algorithm.rltf_sd.main_pg_coef` (default `1.0`; set `0.0` for pure-KL)
- `algorithm.rltf_sd.use_stable_request_id` (default `false`; recommended `false` for rollout_n>1 to avoid vLLM request id collisions)
- `algorithm.rltf_sd.sd_coef` (default `1.0`)
- `algorithm.rltf_sd.gamma` (default `1.0`)
- `algorithm.rltf_sd.max_pairs_per_episode` (default `8`)
- `algorithm.rltf_sd.kl.*`:
  - `enable`, `teacher_mode`, `distill_mode`, `topk`, `beta`, `coef`
  - `vocab_chunk_size`, `token_chunk_size`, `teacher_logits_cpu_offload`
  - `pure_only`, `strict_pure`
- `algorithm.rltf_sd.feedback.*`

To activate RLTF-SD, use new config files or pass equivalent `server_overrides`.

## 4. Environment-Specific Feedback

- Math: LLM reviewer (`endpoint` required, `model` optional).
  - If reviewer is unavailable and `fail_fast=true`, training fails immediately.
  - Throughput controls (optional): `max_concurrency`, `max_tokens`, `retries`, `retry_backoff`, `acquire_timeout`.
  - For OpenAI-compatible endpoints, `enforce_json_response=true` is recommended.
- Gomoku: if `algorithm.rltf_sd.feedback.gomoku.endpoint` is set, call reviewer API; otherwise fallback to built-in rule-based critique.
- ALFWorld: if `algorithm.rltf_sd.feedback.alfworld.endpoint` is set, call reviewer API; otherwise fallback to built-in executable-action critique.

## 5. Run Commands

## 5.1 Start Scheduler

```bash
bash opentinker/scripts/launch_scheduler.sh --scheduler-port <scheduler_port>
```

## 5.2 Start Environments

Math:

```bash
python opentinker/environment/math/math_server.py --port <env_port>
```

Gomoku:

```bash
python opentinker/environment/gomoku/gomoku_server.py --port <env_port>
```

ALFWorld:

```bash
python -m opentinker.environment.alfworld.alfworld_server \
  --port <env_port> \
  --max_steps 50 \
  --split train \
  --num_games 5
```

## 5.3 Train with RLTF-SD

# launch the reviewer service if using math feedback (example with vLLM, adjust as needed for your reviewer):
vllm serve Qwen/Qwen3-4B --port 8084


Math (single-turn):

```bash
python opentinker/client/math_rl.py \
  --config-name math_rltf_sd_param \
  tokenizer_path=<model_path> \
  data_path=<train_data> \
  val_data_path=<val_data> \
  scheduler_url=http://<server_endpoint>:<scheduler_port> \
  interaction.config.env_host=<client_endpoint> \
  interaction.config.env_port=<env_port> \
  server_overrides.algorithm.rltf_sd.feedback.math.endpoint=http://<reviewer_host>:<reviewer_port>

# Optional if your reviewer service requires explicit model selection:
# server_overrides.algorithm.rltf_sd.feedback.math.model=<reviewer_model>

# Optional for slow reviewer endpoints:
# rollout_n=4 val_batch_size=16 validate_before_training=false \
# server_overrides.algorithm.rltf_sd.feedback.math.max_tokens=128 \
# server_overrides.algorithm.rltf_sd.feedback.math.max_concurrency=1 \
# server_overrides.algorithm.rltf_sd.feedback.math.timeout=60
```

Gomoku (multi-turn):

```bash
python opentinker/client/gomoku_rl.py \
  --config-name gomoku_rltf_sd_param \
  tokenizer_path=<model_path> \
  scheduler_url=http://<server_endpoint>:<scheduler_port> \
  interaction.config.env_host=<client_endpoint> \
  interaction.config.env_port=<env_port> \
  server_overrides.algorithm.rltf_sd.feedback.gomoku.endpoint=http://<reviewer_host>:<reviewer_port>

# Optional:
# server_overrides.algorithm.rltf_sd.feedback.gomoku.model=<reviewer_model>
```

ALFWorld (multi-turn):

```bash
python opentinker/client/alfworld_rl.py \
  --config-name alfworld_rltf_sd_param \
  tokenizer_path=<model_path> \
  scheduler_url=http://<server_endpoint>:<scheduler_port> \
  interaction.config.env_host=<client_endpoint> \
  interaction.config.env_port=<env_port> \
  server_overrides.algorithm.rltf_sd.feedback.alfworld.endpoint=http://<reviewer_host>:<reviewer_port>

# Optional:
# server_overrides.algorithm.rltf_sd.feedback.alfworld.model=<reviewer_model>
```

## 5.4 Train with RLTF-SD Pure KL (Math)

```bash
python opentinker/client/math_rl.py \
  --config-name math_rltf_sd_kl_param \
  tokenizer_path=<model_path> \
  data_path=<train_data> \
  val_data_path=<val_data> \
  scheduler_url=http://<server_endpoint>:<scheduler_port> \
  interaction.config.env_host=<client_endpoint> \
  interaction.config.env_port=<env_port> \
  server_overrides.algorithm.rltf_sd.feedback.math.endpoint=http://<reviewer_host>:<reviewer_port>
```

Pure-KL strict mode (`algorithm.rltf_sd.kl.pure_only=true` and `strict_pure=true`) will
override conflicting gradient sources:
- `algorithm.use_kl_in_reward=false`
- `algorithm.use_kl_in_advantage=false`
- `actor_rollout_ref.actor.use_kl_loss=false`

## 6. Expected Metrics

When RLTF-SD is enabled, training logs include:

- `rltf_sd/pair_count`
- `rltf_sd/b0_mean`
- `rltf_sd/adv_mean`
- `rltf_sd/loss`

Additional worker metrics may include:

- `rltf_sd/grad_norm`
- `rltf_sd/lr`

For `algorithm.rltf_sd.loss_type=kl`, additional metrics include:

- `rltf_sd/kl_loss`
- `rltf_sd/loss_type_kl`
- `rltf_sd/kl_coef`
- `rltf_sd/kl_topk` / `rltf_sd/kl_beta`
- `rltf_sd/kl_teacher_mode_fixed` / `rltf_sd/kl_teacher_mode_shared`
- `rltf_sd/kl_token_chunk_size`
- `rltf_sd/kl_auto_token_chunk_used` (1 means auto fallback chunking was used)
- `rltf_sd/kl_response_len_used`

## 7. Compatibility Notes

- Existing docs/use-cases remain unchanged when not using new config/overrides.
- Legacy commands continue to use `generic_agent` by default.
- RLTF-SD path currently supports FSDP/FSDP2 actor strategy.
- No changes were made to `/verl/**`.

## 8. Troubleshooting

Math reviewer connectivity:

- Symptom: immediate runtime error before/at rollout.
- Check:
  - `server_overrides.algorithm.rltf_sd.feedback.math.endpoint`
  - `server_overrides.algorithm.rltf_sd.feedback.math.model` (only if endpoint requires it)
  - reviewer API key (`api_key` or `api_key_env`)
  - If timeout occurs under high load, reduce `rollout_n` / `val_batch_size`, lower reviewer `max_tokens`, and tune `max_concurrency` + `timeout`.

No SD updates triggered:

- Check `rltf_sd/pair_count` is > 0.
- Confirm `default_agent_loop=rltf_sd_agent` and `algorithm.rltf_sd.enable=true` are both set.

vLLM `Request id ... already running`:

- Cause: duplicated stable `request_id` under parallel rollouts.
- Fix: keep `algorithm.rltf_sd.use_stable_request_id=false` (default in RLTF-SD configs).

Memory pressure / OOM:

- Lower `max_pairs_per_episode`.
- Reduce `max_new_tokens` or batch size.
- Keep `use_dynamic_bsz=true`.
- For `loss_type=kl`, reduce `algorithm.rltf_sd.kl.token_chunk_size` (e.g. `64` or `32`).

Numerical sanity:

- `A_SD` is computed as `r1 - b(0)` in `opentinker/server/rltf_sd_utils.py`.
- `b(0)` is group mean of collected `r0` by `group_id`.
