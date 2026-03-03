# OPSD 论文复现实操（2601.18734）

本文档给出 OpenTinker 内的最小复现路径，并明确当前实现与论文主设定的差异。

## 1. 你当前 OPSD 与论文的关键差距

1. 训练目标不同（最关键）  
当前代码主路径是 sampled-token 的 `KL-in-advantage`（策略梯度形态）。  
论文主结果默认使用 **full-vocabulary logit distillation + JSD(beta=0.5)**（Eq.6/7）。

2. teacher 参数模式  
论文强调 teacher 固定为初始策略更稳定。  
本仓库已改为默认 `teacher_mode=fixed`，并提供 paper 配置。

3. 训练超参默认值  
原默认配置和论文有偏差（如 rollout 数、生成长度、LR 调度、有效 batch 等）。  
已新增 paper 配置文件对齐常用设定。

4. 数据与评测协议  
论文使用 OpenThoughts math 子集（最多 30k）训练，AIME24/AIME25/HMMT25/AMO-Bench 评测，`average@16`。  
仓库已提供对应数据转换脚本。

## 2. 本次改动（已完成）

1. 对齐 teacher prompt 到论文 Figure 2 文案风格（`opentinker/server/opsd_utils.py`）。  
2. `math_opsd_rl.py` 支持 `server_overrides` 合并，避免写死训练器底层参数。  
3. 客户端显式下发 LoRA 配置（可确保全参数训练：`lora_rank=0`）。  
4. 新增论文复现配置：`opentinker/client/client_config/math_opsd_paper_param.yaml`。  
5. 清理 `http_training_server.py` 中冗余 debug 日志，简化训练主流程。

## 3. 启动步骤

### Step 0: 环境准备

按仓库根目录 `README.md` 完成安装（OpenTinker + verl + 依赖）。

### Step 1: 准备论文同分布数据（30k train + benchmark test）

```bash
python data/prepare_opsd_math_data.py \
  --output_dir data/opsd_math \
  --max_train_samples 30000 \
  --seed 42
```

输出重点文件：

- `data/opsd_math/train_openthoughts_math.parquet`
- `data/opsd_math/test_merged.parquet`

### Step 2: 启动 scheduler 与环境服务

```bash
# terminal A
bash opentinker/scripts/launch_scheduler.sh --scheduler-port 8780 --gpus "[0,1,2,3,4,5,6,7]"

# terminal B
python opentinker/environment/math/math_server.py --port 8082
```

### Step 3: 启动 OPSD 训练（全参数、非 LoRA）

```bash
python opentinker/client/math_opsd_rl.py \
  --config-name math_opsd_paper_param.yaml \
  tokenizer_path=Qwen/Qwen3-4B-Instruct-2507 \
  data_path=data/opsd_math/train_openthoughts_math.parquet \
  val_data_path=data/opsd_math/test_merged.parquet \
  scheduler_url=http://localhost:8781 \
  interaction.config.env_host=localhost \
  interaction.config.env_port=8083
```

如果你不是 8 卡，把 `num_gpus` 和 `server_overrides.actor_rollout_ref.actor.*batch*` 适当改小。

## 4. 结果解读与“精准复现”说明

1. 上述流程已经对齐了论文的大部分可操作设定：  
`fixed teacher`、`rollout_n=1`、`2k generation`、`temperature=1.2`、`30k train`、`average@16`、`全参数训练`。

2. 仍未与论文“完全等价”的点：  
当前主训练目标仍是 sampled-token KL 形态，不是 full-vocab JSD 主目标。  
这会影响你是否能完全复现论文主表里的绝对数值。

3. 若要进一步逼近论文主结果：  
需要在 `opentinker/backend_patch/verl` 里补一版 full-vocab teacher-student divergence 的 actor 侧 loss（避免改 `verl/` 原目录）。
