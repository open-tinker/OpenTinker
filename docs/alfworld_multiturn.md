# LLM Game Agent (ALFWorld Multi-Turn)

**Author:** Haofeiy

This example demonstrates training a language model to complete household tasks in the ALFWorld text-based environment.

## Overview

ALFWorld is a text-based interactive environment for training agents to complete household tasks. Tasks include:

- Pick and place objects
- Look at objects under light
- Clean objects and place them
- Heat/cool objects and place them
- Pick multiple objects

## Prerequisites

1. Complete the [Installation](../README.md#-installation) steps
2. Install ALFWorld: `pip install alfworld`
3. Download ALFWorld data: `alfworld-download`
4. Get your IP address: `hostname -I`

## Step 1: Start the Scheduler (Server Side)

```bash
bash opentinker/scripts/launch_scheduler.sh --scheduler-port <scheduler_port>
```

## Step 2: Start the ALFWorld Environment (Client Side)

```bash
python -m opentinker.environment.alfworld.alfworld_server \
    --port <env_port> \
    --max_steps 50 \
    --split train \
    --num_games 5 # can be larger or smaller
```

**Server Options:**

- `--port`: Server port (default: 8082)
- `--max_steps`: Max steps per episode (default: 50)
- `--split`: Dataset split (`train`, `eval_in_distribution`, `eval_out_of_distribution`)
- `--num_games`: Number of games to load (-1 = all, use smaller value for faster loading)

## Step 3: Run Training

```bash
python opentinker/client/alfworld_rl.py \
    tokenizer_path=Qwen/Qwen2.5-3B-Instruct \
    batch_size=4 \
    val_batch_size=50 \
    num_steps=1000 \
    save_freq=20000 \
    test_freq=10 \
    scheduler_url=http://<server_endpoint>:<scheduler_port> \
    interaction.config.env_port=<env_port> \
    interaction.config.env_host=<client_endpoint>
```

**Training Parameters:**

- `num_steps`: Total training steps (alternative: use `num_epochs`)
- `batch_size`: Training batch size
- `val_batch_size`: Validation samples per evaluation
- `test_freq`: Validation frequency (every N steps)
- `adv_estimator`: Advantage estimator (`gae`, `grpo`, `grpo_per_step`)

## Reward Structure

| Event            | Reward |
| ---------------- | ------ |
| Task Success     | +10.0  |
| Task Failure     | -1.0   |
| Per Step Penalty | -0.01  |
| Invalid Action   | -0.1   |

## Example Actions

The agent interacts with the environment using text commands:

- `go to desk 1` - Navigate to a location
- `take book 1 from desk 1` - Pick up an object
- `put book 1 in/on shelf 1` - Place an object
- `open drawer 1` - Open a container
- `use lamp 1` - Use a device
- `examine book 1` - Look at an object
- `inventory` - Check held items
- `look` - Look around

## Configuration Reference

See [`opentinker/client/client_config/alfworld_param.yaml`](../opentinker/client/client_config/alfworld_param.yaml) for full configuration options.
