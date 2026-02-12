docker exec -it opd_siqi bash
cd /workspace/dev/OpenTinker

bash opentinker/scripts/launch_scheduler.sh

python opentinker/client/distill_rl.py \
    tokenizer_path=Qwen/Qwen2.5-0.5B-Instruct \
    distillation.teacher_model_path=Qwen/Qwen2.5-3B-Instruct \
    data_path=./data/math/train.parquet \
    val_data_path=./data/math/test.parquet \
    num_gpus=4


python opentinker/client/distill_rl.py \
    tokenizer_path=Qwen/Qwen2.5-1.5B \
    distillation.teacher_model_path=Qwen/Qwen2.5-7B-Instruct \
    data_path=./data/math/train.parquet \
    val_data_path=./data/math/test.parquet \
    num_gpus=4


python opentinker/client/math_tool_rl.py \
    tokenizer_path=Qwen/Qwen2.5-1.5B \
    batch_size=16 \
    val_batch_size=64 \
    data_path=data/math_agentloop/train.parquet \
    val_data_path=data/math_agentloop/test.parquet \
    num_epochs=5 \
    save_freq=1000 \
    test_freq=5
    # scheduler_url=http://<server_endpoint>:<scheduler_port> \
    # interaction.config.env_port=<env_port> \
    # interaction.config.env_host=<client_endpoint>