# LLM Multi-Turn Math

**Author:** Siqi Zhu

This example demonstrates training a language model to solve mathematical problems with multi-turn tool calling (Code Interpreter).

## Prerequisites

1. Complete the [Installation](../README.md#-installation) steps
2. Get your IP address: `hostname -I`

## Step 1: Start the Scheduler (Server Side)

```bash
bash opentinker/scripts/launch_scheduler.sh --scheduler-port <scheduler_port>
```

## Step 2: Start the Math Tool Environment (Client Side)

```bash
python opentinker/environment/math/math_tool_server.py --port <env_port>
```

## Step 3: Generate Training Data

```bash
python opentinker/data_preprocess/math_multiturn_w_interaction.py \
    --local_save_dir=data/math_agentloop
```

## Step 4: Run Training

```bash
python opentinker/client/math_tool_rl.py \
    tokenizer_path=Qwen/Qwen2.5-1.5B \
    batch_size=16 \
    val_batch_size=64 \
    data_path=data/math_agentloop/train.parquet \
    val_data_path=data/math_agentloop/test.parquet \
    num_epochs=5 \
    save_freq=1000 \
    test_freq=5 \
    scheduler_url=http://<server_endpoint>:<scheduler_port> \
    interaction.config.env_port=<env_port> \
    interaction.config.env_host=<client_endpoint>
```

## Step 5: Run Inference (Optional)

```bash
python opentinker/client/math_tool_inference.py \
    model_path=<model_name> \
    data_path=data/math_agentloop/test.parquet \
    output_path=./tmp/results.jsonl \
    max_samples=5 \
    env_endpoint=http://<client_endpoint>:<env_port> \
    scheduler_url=http://<server_endpoint>:<scheduler_port>
```

## Performance

See [wandb run](https://wandb.ai/zsqzz/Open-Tinker/runs/f5pt6gcw?nw=nwuserzhusq20) for training metrics and results.
