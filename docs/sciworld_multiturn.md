# LLM Game Agent (ScienceWorld Multi-Turn)

**Author:** Haofeiy

This example demonstrates training a language model to complete science tasks in the ScienceWorld text-based environment.

## Overview

ScienceWorld is a text-based benchmark of grounded science tasks. Tasks include:

- Boiling / freezing / melting substances
- Identifying and classifying living things
- Using instruments (thermometer, microscope, etc.)
- Combining materials to produce reactions
- Navigating rooms to find and manipulate objects

OpenTinker support follows the same pattern as ALFWorld:

- `SciWorldGame` wraps the benchmark as an `AbstractGame`
- `sciworld_server.py` exposes the environment over the generic FastAPI server
- `sciworld_rl.py` trains against that server through `GameEnvironment`

## Prerequisites

1. Complete the [Installation](../README.md#-installation) steps
2. Install ScienceWorld: `pip install scienceworld`
3. Ensure **Java** is available (`java -version`), since ScienceWorld launches a JVM-backed server
4. Get your IP address if client and scheduler run on different machines: `hostname -I`

## Step 1: Start the Scheduler

```bash
bash opentinker/scripts/launch_scheduler.sh --scheduler-port <scheduler_port>
```

## Step 2: Start the ScienceWorld Environment

```bash
python -m opentinker.environment.sciworld.sciworld_server \
    --port <env_port> \
    --max_steps 30 \
    --split train \
    --shards 8 \
    --threads-per-shard 256
```

Optional task restriction:

```bash
python -m opentinker.environment.sciworld.sciworld_server \
    --port <env_port> \
    --split train \
    --task-name boil \
    --task-name find-animal
```

Useful server options:

- `--split`: ScienceWorld split to sample variations from (`train`, `dev`, `test`)
- `--task-name`: Repeat to restrict the task pool
- `--task-id`: Alternative to task names if you prefer numeric task ids
- `--variation`: Repeat to restrict to explicit variation ids
- `--simplification-str`: Pass-through simplification string for `env.load()`
- `--thread-base`: Base ScienceWorld thread number for this server group
- `--threads-per-shard`: Reserved thread-number block per shard

## Step 3: Run Training

```bash
python opentinker/client/sciworld_rl.py \
    tokenizer_path=Qwen/Qwen2.5-3B-Instruct \
    batch_size=4 \
    num_steps=1000 \
    test_freq=10 \
    scheduler_url=http://<server_endpoint>:<scheduler_port> \
    interaction.config.env_port=<env_port> \
    interaction.config.env_host=<client_endpoint> \
    interaction.config.split=train \
    interaction.config.local_thread_base=20000
```

## Notes

- Keep `num_workers=0` for the local prompt-generation dataloaders unless you
  explicitly manage non-overlapping ScienceWorld thread bases per worker.
- Use the same `split`, `task_names`, `task_ids`, and `variation_indices`
  settings on both the environment server and the client config so prompt
  generation matches the remote environment.
- If you already run ScienceWorld-backed processes on the same machine, move
  `--thread-base` and `interaction.config.local_thread_base` to disjoint ranges.

## Reward Structure

| Event            | Reward |
| ---------------- | ------ |
| Task Success     | +10.0  |
| Task Failure     | -1.0   |
| Per Step Penalty | -0.01  |
| Invalid Action   | -0.1   |

## Example Actions

The agent interacts with the environment using text commands:

- `look around` - Observe the current room
- `open door to kitchen` - Navigate between rooms
- `pick up thermometer` - Pick up an object
- `use thermometer on water` - Use an instrument
- `pour water into beaker` - Combine or transfer materials
- `focus on substance in microscope` - Examine with instruments
- `inventory` - Check held items
- `wait` - Wait one step (e.g. for a reaction)

## Configuration Reference

See [`opentinker/client/client_config/sciworld_param.yaml`](../opentinker/client/client_config/sciworld_param.yaml)
for the full configuration.
