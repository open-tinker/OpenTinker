# On-Policy Distillation 运行指南

本指南介绍如何在 OpenTinker 中运行 On-Policy Distillation。

## 1. 环境准备

确保你的 Python 环境中包含 `opentinker` 和 `verl` 路径。建议在 `OpenTinker` 仓库根目录下执行：

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/verl
```

## 2. 数据准备

Distillation 需要提供 Prompt 数据。你可以使用现有的数学数据集脚本生成 Parquet 文件：

```bash
python opentinker/data_preprocess/math_multiturn_w_interaction.py --local_save_dir=./data/math
```

**注意**：如果遇到 `ModuleNotFoundError: No module named 'verl.utils.hdfs_io'`，请确保已按照步骤 1 设置 `PYTHONPATH`。

## 3. 启动 Scheduler

在运行 Distillation 任务之前，需要启动 OpenTinker Scheduler：

```bash
# 假设你在 4 张 GPU 的机器上
python opentinker/scheduler/launch_scheduler.py --num_gpus=4
```

## 4. 运行 Distillation 客户端

使用 `distill_rl.py` 脚本启动训练。你需要指定学生模型、教师模型和数据路径。

### 示例命令：

```bash
python opentinker/client/distill_rl.py \
    tokenizer_path=Qwen/Qwen2.5-0.5B-Instruct \
    distillation.teacher_model_path=Qwen/Qwen2.5-3B-Instruct \
    data_path=./data/math/train.parquet \
    num_gpus=4
```

### 关键参数说明：

- `tokenizer_path`: 学生模型的路径或 HuggingFace ID。
- `distillation.teacher_model_path`: 教师模型的路径或 ID。
- `data_path`: 训练数据 Parquet 文件的路径。
- `distillation.distillation_mode`: 默认为 `pure`（仅执行 Distillation Loss）。
- `distillation.distillation_kl_type`: 默认为 `forward`（即 $KL(Teacher || Student)$）。

## 5. 监控与日志

- **控制台输出**：你会看到 `[Distillation] teacher_log_probs computed via ref_policy` 以及 `pg_loss`（即 Distillation Loss）的数值。
- **WandB**：如果配置了 `wandb`，可以在 WandB 仪表盘中查看 `actor/distill_loss` 和 `actor/entropy` 等指标。
- **Checkpoint**：模型会自动保存到 `checkpoints/opentinker/onpolicy_distillation`。

## 6. 常见问题

- **内存不足 (OOM)**：由于学生 (0.5B) 和教师 (3B) 模型通过 FSDP 同步加载在相同的 GPU 上，如果显存不足，请减小 `batch_size` 或 `max_new_tokens`。
- **教师模型加载失败**：确保 `teacher_model_path` 指向正确的 HuggingFace 模型或本地路径。
