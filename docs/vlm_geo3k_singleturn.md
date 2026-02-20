# VLM Single-Turn Math (Geometry3K)

**Author:** Siqi Zhu

This example demonstrates training a vision-language model to solve geometry problems from the Geometry3K dataset.

## Prerequisites

1. Complete the [Installation](../README.md#-installation) steps
2. Get your IP address: `hostname -I`

## Step 1: Start the Scheduler (Server Side)

```bash
bash opentinker/scripts/launch_scheduler.sh --scheduler-port <scheduler_port>
```

## Step 2: Start the Geo3K Environment (Client Side)

```bash
python opentinker/environment/geo3k/geo3k_server.py --port <env_port>
```


## Step 3: Generate Training Data

```bash
python opentinker/data_preprocess/geo3k_multiturn_w_interaction.py \
    --local_save_dir=data/geo3k_multiturn_w_tool
```

## Step 4: Run Training

```bash
python opentinker/client/geo3k_rl.py \
    tokenizer_path=Qwen/Qwen2-VL-2B-Instruct \
    batch_size=16 \
    val_batch_size=64 \
    data_path=data/geo3k_multiturn_w_tool/train.parquet \
    val_data_path=data/geo3k_multiturn_w_tool/test.parquet \
    num_epochs=5 \
    save_freq=1000 \
    test_freq=5 \
    scheduler_url=http://<server_endpoint>:<scheduler_port> \
    interaction.config.env_port=<env_port> \
    interaction.config.env_host=<client_endpoint>
```

## Performance

See [wandb run](https://wandb.ai/zsqzz/Open-Tinker/runs/aidfc2y1?nw=nwuserzhusq20) for training metrics and results.
