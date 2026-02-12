docker exec -it opd_siqi bash
cd /workspace/dev/OpenTinker

bash opentinker/scripts/launch_scheduler.sh

python opentinker/client/distill_rl.py \
    tokenizer_path=Qwen/Qwen2.5-0.5B-Instruct \
    distillation.teacher_model_path=Qwen/Qwen2.5-3B-Instruct \
    data_path=./data/math/train.parquet \
    num_gpus=4