# LLM Game Agent (Gomoku Multi-Turn)

**Author:** Siqi Zhu

This example demonstrates training a language model to play Gomoku in a multi-turn game environment.

## Prerequisites

1. Complete the [Installation](../README.md#-installation) steps
2. Get your IP address: `hostname -I`

## Step 1: Start the Scheduler (Server Side)

```bash
bash opentinker/scripts/launch_scheduler.sh --scheduler-port <scheduler_port>
```

## Step 2: Start the Gomoku Environment (Client Side)

```bash
python opentinker/environment/gomoku/gomoku_server.py --port <env_port>
```

## Step 3: Run Training

```bash
python opentinker/client/gomoku_rl.py \
    tokenizer_path=Qwen/Qwen2.5-3B-Instruct \
    batch_size=16 \
    val_batch_size=32 \
    num_epochs=5 \
    save_freq=1000 \
    test_freq=5 \
    scheduler_url=http://<server_endpoint>:<scheduler_port> \
    interaction.config.env_port=<env_port> \
    interaction.config.env_host=<client_endpoint>
```

## Step 4: Run Inference (Optional)

```bash
python opentinker/client/gomoku_inference.py \
    model_path=<model_name> \
    output_path=./tmp/results.jsonl \
    max_samples=5 \
    env_endpoint=http://<client_endpoint>:<env_port> \
    scheduler_url=http://<server_endpoint>:<scheduler_port>
```

## Performance

See [wandb run](https://wandb.ai/zsqzz/Open-Tinker/runs/7a7ggkw3?nw=nwuserzhusq20) for training metrics and results.
