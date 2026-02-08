# LLM SWE-Gym Agent (Multi-Turn)

**Author:** OpenTinker Contributors

This example trains an LLM to solve software engineering tasks in SWE-Gym using
multi-turn patch-and-test interaction.

## Prerequisites

1. Complete the [Installation](../README.md#-installation) steps
2. Install datasets: `pip install datasets`
3. Get your IP address: `hostname -I`

## Step 1: Start the Scheduler (Server Side)

```bash
bash opentinker/scripts/launch_scheduler.sh --scheduler-port <scheduler_port>
```

## Step 2: Start the SWE-Gym Environment (Client Side)

```bash
python opentinker/environment/swegym/swegym_server.py \
  --port <env_port> \
  --split train \
  --max-prompt-tokens 131072 \
  --tokenizer-path Qwen/Qwen2.5-1.5B
```

## Step 3: Run Training

```bash
python opentinker/client/swegym_rl.py \
  tokenizer_path=Qwen/Qwen2.5-1.5B \
  batch_size=4 \
  val_batch_size=50 \
  num_steps=500 \
  save_freq=20000 \
  test_freq=10 \
  scheduler_url=http://<server_endpoint>:<scheduler_port> \
  interaction.config.env_port=<env_port> \
  interaction.config.env_host=<client_endpoint>
```

## Notes

- The agent must output a unified diff patch as its action.
- `FAIL_TO_PASS` tests are run each step; `PASS_TO_PASS` is optional.
- Repos are cached under `/tmp/swegym/repos` by default.

