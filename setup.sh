docker exec -it opd_siqi bash
cd /workspace/dev/OpenTinker

bash opentinker/scripts/launch_scheduler.sh

python opentinker/client/distill_rl.py \
    tokenizer_path=Qwen/Qwen2.5-0.5B-Instruct \
    distillation.teacher_model_path=Qwen/Qwen2.5-3B-Instruct \
    data_path=./data/math/train.parquet \
    val_data_path=./data/math/test.parquet \
    num_gpus=4


# OPD

python opentinker/client/distill_rl.py \
    tokenizer_path=Qwen/Qwen2.5-1.5B \
    distillation.teacher_model_path=Qwen/Qwen2.5-7B-Instruct \
    data_path=./data/math/train.parquet \
    val_data_path=./data/math/test.parquet \
    num_gpus=4

# OPSD 
python opentinker/client/self_distill_rl.py \
      tokenizer_path=Qwen/Qwen2.5-1.5B-Instruct \
      data_path=./data/math/train.parquet \
      val_data_path=./data/math/test.parquet \
      self_distillation.loss_type=jsd \
      self_distillation.beta=0.5


python opentinker/client/math_tool_rl.py \
    tokenizer_path=Qwen/Qwen2.5-3B-Instruct \
    batch_size=16 \
    val_batch_size=100 \
    data_path=data/math_agentloop/train.parquet \
    val_data_path=data/math_agentloop/test.parquet \
    num_epochs=5 \
    save_freq=1000 \
    test_freq=5
    # scheduler_url=http://<server_endpoint>:<scheduler_port> \
    # interaction.config.env_port=<env_port> \
    # interaction.config.env_host=<client_endpoint>



python opentinker/client/self_distill_rl.py \
      tokenizer_path=Qwen/Qwen2.5-7B-Instruct \
      data_path=./data/math/train.parquet \
      val_data_path=data/Maxwell-Jia/AIME_2024/aime_2024_problems.parquet \
      self_distillation.loss_type=sampled_token \
      self_distillation.beta=0.5


python opentinker/client/self_distill_rl.py \
      tokenizer_path=Qwen/Qwen3-4B \
      data_path=./data/math/train.parquet \
      val_data_path=data/Maxwell-Jia/AIME_2024/aime_2024_problems.parquet \
      self_distillation.loss_type=sampled_token \
      self_distillation.beta=0.5


python opentinker/client/self_distill_rl.py \
      tokenizer_path=Qwen/Qwen2.5-7B-Instruct \
      data_path=./data/math/train.parquet \
      val_data_path=./data/math/test.parquet \
      self_distillation.loss_type=sampled_token \
      self_distillation.beta=0.5