# OPSD 复现实验偏差诊断（2601.18734，4卡，全参数，当前看 `pass@4`）

## 1. 当前默认口径（本仓库）

`opentinker/client/client_config/math_opsd_paper_param.yaml` 当前默认值：

- 全参数训练：`lora.lora_rank=0`
- 验证口径：`val_n=4`（`pass@4`）
- 生成长度：`max_new_tokens=2048`、`val_max_new_tokens=2048`
- 采样策略：`temperature=1.2`、`top_p=0.95`
- 系统开关：`enable_agent_loop=true`、`enable_thinking=false`
- 蒸馏目标：`distill_mode=topk_reverse_kl_tail`、`topk=32`

注：`topk_reverse_kl_tail` 模式下，`beta` 参数不参与损失计算。

其中：

- `enable_agent_loop` 已从客户端硬编码改为配置项（默认 `true`）。
- `enable_thinking` 已改为可配置（默认 `false`，可显式开 `true`）。

## 2. 基础启动（4卡）

```bash
# terminal A
bash opentinker/scripts/launch_scheduler.sh --scheduler-port 8781 --gpus "[0,1,2,3]"

# terminal B
python opentinker/environment/math/math_server.py --port 8083
```

```bash
# terminal C
python opentinker/client/math_opsd_rl.py \
  --config-name math_opsd_paper_param.yaml \
  tokenizer_path=Qwen/Qwen3-4B-Instruct-2507 \
  data_path=data/opsd_math/train_openthoughts_math.parquet \
  val_data_path=data/opsd_math/test_merged.parquet \
  scheduler_url=http://localhost:8781 \
  interaction.config.env_host=localhost \
  interaction.config.env_port=8083 \
  num_gpus=4
```

## 3. A/B 实验（每组 100 step）

### A 组（旧口径对照：8192 + top_p=1.0）

```bash
python opentinker/client/math_opsd_rl.py \
  --config-name math_opsd_paper_param.yaml \
  tokenizer_path=Qwen/Qwen3-4B-Instruct-2507 \
  data_path=data/opsd_math/train_openthoughts_math.parquet \
  val_data_path=data/opsd_math/test_merged.parquet \
  scheduler_url=http://localhost:8781 \
  interaction.config.env_host=localhost \
  interaction.config.env_port=8083 \
  num_gpus=4 \
  num_steps=100 \
  max_new_tokens=8192 \
  val_max_new_tokens=8192 \
  top_p=1.0
```

### B 组（对齐口径：2048 + top_p=0.95）

```bash
python opentinker/client/math_opsd_rl.py \
  --config-name math_opsd_paper_param.yaml \
  tokenizer_path=Qwen/Qwen3-4B-Instruct-2507 \
  data_path=data/opsd_math/train_openthoughts_math.parquet \
  val_data_path=data/opsd_math/test_merged.parquet \
  scheduler_url=http://localhost:8781 \
  interaction.config.env_host=localhost \
  interaction.config.env_port=8083 \
  num_gpus=4 \
  num_steps=100 \
  max_new_tokens=2048 \
  val_max_new_tokens=2048 \
  top_p=0.95
```

对比指标：

- `val/pass_at_4`
- `val/mean_score`
- `distill/full_vocab_jsd_loss`
- `response_length/clip_ratio`

## 4. Agent Loop / Thinking A/B

### C1（默认）：Agent Loop on，Thinking off

```bash
# 在第 2 节基础命令末尾追加：
enable_agent_loop=true enable_thinking=false
```

### C2：Agent Loop on，Thinking on

```bash
# 在第 2 节基础命令末尾追加：
enable_agent_loop=true enable_thinking=true
```

### C3：Agent Loop off（关键对照）

```bash
# 在第 2 节基础命令末尾追加：
enable_agent_loop=false
```

## 5. 统计稳定性测试（同 checkpoint 连续 3 次 validation）

建议固定同一 checkpoint，重复跑 3 次仅验证：

```bash
CKPT_PATH=/abs/path/to/ckpt
for i in 1 2 3; do
  python opentinker/client/math_opsd_rl.py \
    --config-name math_opsd_paper_param.yaml \
    tokenizer_path=Qwen/Qwen3-4B-Instruct-2507 \
    data_path=data/opsd_math/train_openthoughts_math.parquet \
    val_data_path=data/opsd_math/test_merged.parquet \
    scheduler_url=http://localhost:8781 \
    interaction.config.env_host=localhost \
    interaction.config.env_port=8083 \
    num_gpus=4 \
    num_steps=0 \
    val_before_train=true \
    server_overrides.trainer.resume_mode=resume_path \
    server_overrides.trainer.resume_from_path=$CKPT_PATH \
    experiment_name=math_opsd_val_repeat_${i}
done
```

若 `pass@4` 三次最大差值 `> 0.02`，先判为高噪声口径，不直接下“性能退化”结论。

## 6. 结论模板（按可归因性）

| 类别 | run-id | 关键设置 | `val/pass@4` | `val/mean_score` | `jsd_loss` | 结论 |
|---|---|---|---:|---:|---:|---|
| 参数口径问题 |  |  |  |  |  |  |
| 训练框架行为问题 |  |  |  |  |  |  |
| 真实性能退化 |  |  |  |  |  |  |
