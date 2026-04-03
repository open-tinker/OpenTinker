# World Model Learning — Implementation Progress

## Method 1: WMC-ERC (World Model-Conditioned Entropy Regularized Co-evolution)
**Status: Complete**
- File: `opentinker/backend_patch/verl/trainer/ppo/wmc_erc.py`
- Uses the LLM's prediction entropy at env token positions as a World Model uncertainty signal
- Dynamically gates policy gradient updates to prevent entropy collapse

## Method 2: World Model SFT Loss
**Status: Complete**
- File: `opentinker/backend_patch/verl/trainer/ppo/world_model_loss.py`
- Auxiliary SFT loss on observation tokens (next-token prediction)
- Gated by `world_model_coeff` in actor config

## Method 3: RWML — Reinforcement World Model Learning (arxiv:2602.05842)
**Status: Complete**

### What it does
Per-turn RL reward based on text-level embedding similarity between model's predicted
next observation and actual environment observation, using an external embedding model.

```
d(pred, actual) = 1 - cos(E(pred), E(actual))  [external embedding model]
r^WM = 1.0 if d < tau_d else 0.0               [binary reward, paper default tau_d=0.2]
```

### How predictions are obtained
During `compute_log_prob`, logits at observation token positions are argmax-decoded
to get the model's predicted token IDs. These are decoded to text and compared with
actual observation tokens using an external sentence-transformers embedding model.

### Files modified
- [x] `opentinker/backend_patch/verl/trainer/ppo/world_model_rl.py` — Core RWML module
  - `EmbeddingSimilarityReward`: loads embedding model, computes cosine sim + binary reward
  - `decode_per_turn_texts()`: extracts observation texts from token IDs per turn
  - `compute_rwml_turn_rewards()`: batch reward computation with metrics
- [x] `verl/verl/workers/actor/dp_actor.py` — Predicted token ID extraction
  - `_forward_micro_batch`: `return_predicted_ids` → argmax on logits
  - `compute_log_prob`: returns `(log_probs, entropys, predicted_ids)`
- [x] `verl/verl/workers/fsdp_workers.py` — DataProto wrapping
  - Includes `predicted_ids` in returned DataProto when RWML is enabled
- [x] `opentinker/server/generic_agent_loop.py` — Observation text storage
  - Stores per-turn observation texts in `extra_fields["turn_observations"]`
- [x] `opentinker/server/http_training_server.py` — RWML integration
  - Initializes embedding model at startup
  - Computes RWML rewards from predicted_ids after compute_log_prob
  - Adds RWML rewards to turn_scores before advantage computation
- [x] `opentinker/client/client_config/alfworld_wmc_erc_param.yaml` — Config

### Configuration
```yaml
rwml:
  enable: false
  embedding_model: "Alibaba-NLP/gte-large-en-v1.5"
  tau_d: 0.2
  coeff: 1.0
```

### Data flow (separated from policy training)
```
Rollout → turn_observations stored in extra_fields
         ↓
compute_log_prob (rwml_enabled=True)
  → argmax on logits → predicted_ids
         ↓
RWML reward computation:
  → decode predicted obs texts from predicted_ids
  → decode actual obs texts from response tokens
  → cos_sim(E(predicted), E(actual)) via embedding model
  → binary reward: 1 if (1-sim) < tau_d
         ↓
RWML GRPO update (SEPARATE from policy):
  → compute_grpo_per_step_advantage(turn_scores=rwml_rewards)
  → update_actor(advantages=rwml_advantages)  ← world model GRPO
         ↓
Policy GRPO update (normal):
  → compute_advantage(turn_scores=task_rewards)
  → update_actor(advantages=policy_advantages) ← policy GRPO
```

### Note on previous hidden-state approach
The initial WM-RL implementation (hidden-state cosine similarity as auxiliary loss)
was superseded by this RWML implementation which correctly follows the paper:
- Text-level (not hidden-state) embedding similarity
- External embedding model (not training model's own representations)
- Per-turn RL reward (not differentiable auxiliary loss)
