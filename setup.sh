python verl/examples/data_preprocess/dapo_multiturn_w_tool.py --local_save_dir ./data/retool_dapo

python opentinker/data_preprocess/math_multiturn_w_interaction.py --local_save_dir ./data/math_agentloop

python verl/examples/data_preprocess/aime2024_multiturn_w_tool.py --local_save_dir ./data/aime2024

docker exec -it opsd_siqi bash
cd /workspace/dev/opsd_mar1

# kill program running on port 8082
kill -9 $(lsof -t -i:8781)


bash opentinker/scripts/launch_scheduler.sh --scheduler-port 8781 --gpus "[6,7,8,9]"
python opentinker/environment/math/math_server.py --port 8083

bash opentinker/scripts/launch_scheduler.sh --scheduler-port 8780 --gpus "[0,1,2,3]"
python opentinker/environment/math/math_server.py --port 8082


HYDRA_FULL_ERROR=1  python opentinker/client/math_opsd_rl.py \
      experiment_name=opsd_wo_kl_test \
      tokenizer_path=Qwen/Qwen3-8B \
      data_path=data/math/train.parquet \
      val_data_path=data/math/test_by_level/test_level_5.parquet \
      rollout_n=4 \
      disable_rl_reward=false \
      scheduler_url=http://localhost:8780 \
      interaction.config.env_port=8082 \
      interaction.config.env_host=localhost
      

python opentinker/client/math_opsd_rl.py \
      experiment_name=opsd_kl_as_loss \
      tokenizer_path=Qwen/Qwen3-8B \
      data_path=data/math/train.parquet \
      val_data_path=data/math/test_by_level/test_level_5.parquet \
      rollout_n=4 \
      opsd_jsd_coef=0.0 \
      opsd_kl_coef=0.1 \
      opsd_beta=0.5 \
      opsd_loss_mode=full_jsd \
      disable_rl_reward=false \
      scheduler_url=http://localhost:8781 \
      interaction.config.env_port=8083 \
      interaction.config.env_host=localhost

# pure opsd
python opentinker/client/math_opsd_rl.py   experiment_name=opsd_pure   tokenizer_path=Qwen/Qwen2.5-3B-Instruct   batch_size=32   rollout_n=1   num_epochs=5   data_path=data/math_agentloop/train.parquet   val_data_path=data/math_agentloop_by_level/test_level_5.parquet   scheduler_url=http://localhost:8781   interaction.config.env_port=8083   interaction.config.env_host=localhost   algorithm.adv_estimator=grpo   algorithm.use_kl_in_advantage=true   algorithm.disable_rl_reward=true   algorithm.kl_penalty_coef=0.1   algorithm.opsd.enable=true


# opsd with grpo
python opentinker/client/math_opsd_rl.py   save_freq=1000      experiment_name=opsd_pure_grpo    tokenizer_path=Qwen/Qwen2.5-1.5B   batch_size=16   rollout_n=16   num_epochs=5   data_path=data/math_agentloop/train.parquet   val_data_path=data/math_agentloop/test.parquet   scheduler_url=http://localhost:8780   interaction.config.env_port=8082   interaction.config.env_host=localhost   algorithm.adv_estimator=grpo   algorithm.use_kl_in_advantage=true   algorithm.disable_rl_reward=false   algorithm.kl_penalty_coef=0   algorithm.opsd.enable=true


data/math_agentloop/test.parquet


python opentinker/client/math_rl.py \
    tokenizer_path=Qwen/Qwen2.5-1.5B \
    batch_size=16 \
    val_batch_size=64 \
    num_epochs=5 \
    save_freq=1000 \
    test_freq=5 \
    data_path=data/math_agentloop/train.parquet \
    val_data_path=data/math_agentloop/test.parquet \
    scheduler_url=http://localhost:8781 \
    interaction.config.env_port=8083 \
    interaction.config.env_host=localhost