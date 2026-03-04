python verl/examples/data_preprocess/dapo_multiturn_w_tool.py --local_save_dir ./data/retool_dapo

python opentinker/data_preprocess/math_multiturn_w_interaction.py --local_save_dir ./data/math_agentloop

python verl/examples/data_preprocess/aime2024_multiturn_w_tool.py --local_save_dir ./data/aime2024

docker exec -it opsd_siqi bash
cd /workspace/dev/opsd_mar3

# kill program running on port 8082
kill -9 $(lsof -t -i:8781)


bash opentinker/scripts/launch_scheduler.sh --scheduler-port 8781 --gpus "[6,7,8,9]"
python opentinker/environment/math/math_server.py --port 8083

bash opentinker/scripts/launch_scheduler.sh --scheduler-port 8781 --gpus "[0,1,2,3]"
python opentinker/environment/math/math_server.py --port 8083


# paper-like OPSD (full-parameter, no LoRA)
python opentinker/client/math_opsd_rl.py \
    --config-name math_opsd_paper_param.yaml \
    tokenizer_path=Qwen/Qwen3-4B \
    data_path=data/opsd_math/train_openthoughts_math.parquet \
    val_data_path=data/opsd_math/test_merged.parquet \
    scheduler_url=http://localhost:8780 \
    interaction.config.env_port=8082 \
    interaction.config.env_host=localhost

# OPSD + RL mixed objective
python opentinker/client/math_opsd_rl.py \
    experiment_name=opsd_rl_mixed \
    tokenizer_path=Qwen/Qwen2.5-1.5B \
    batch_size=16 \
    rollout_n=16 \
    num_epochs=5 \
    data_path=data/math_agentloop/train.parquet \
    val_data_path=data/math_agentloop/test.parquet \
    scheduler_url=http://localhost:8780 \
    interaction.config.env_port=8082 \
    interaction.config.env_host=localhost \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_advantage=true \
    algorithm.disable_rl_reward=false \
    algorithm.kl_penalty_coef=0.1 \
    algorithm.opsd.enable=true \
    algorithm.opsd.teacher_mode=fixed


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


git push origin on-policy-distill-feb20:paper_repri


