# LLM Single-LoRA Single-Turn Math

**Author:** Siqi Zhu

This example demonstrates training a language model with LoRA (Low-Rank Adaptation) for mathematical problem solving.

## Prerequisites

1. Complete the [Installation](../README.md#-installation) steps
2. Get your IP address: `hostname -I`

## Step 1: Start the Scheduler (Server Side)

```bash
bash opentinker/scripts/launch_scheduler.sh --scheduler-port <scheduler_port>
```

## Step 2: Start the Math Environment (Client Side)

```bash
python opentinker/environment/math/math_server.py --port <env_port>
```

## Step 3: Generate Training Data

```bash
python opentinker/data_preprocess/math_multiturn_w_interaction.py \
    --local_save_dir=data/math_agentloop
```

## Step 4: Run LoRA Training

```bash
python opentinker/client/math_rl.py \
    --config-name math_lora_param.yaml \
    tokenizer_path=Qwen/Qwen2.5-0.5B \
    batch_size=64 \
    val_batch_size=100 \
    num_epochs=5 \
    save_freq=1000 \
    test_freq=5 \
    data_path=data/math_agentloop/train.parquet \
    val_data_path=data/math_agentloop/test.parquet \
    scheduler_url=http://<server_endpoint>:<scheduler_port> \
    interaction.config.env_port=<env_port> \
    interaction.config.env_host=<client_endpoint>
```

## Performance

TBA
