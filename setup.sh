python verl/examples/data_preprocess/dapo_multiturn_w_tool.py --local_save_dir ./data/retool_dapo

python verl/examples/data_preprocess/math_multiturn_w_interaction.py --local_save_dir ./data/math_agentloop

python verl/examples/data_preprocess/aime2024_multiturn_w_tool.py --local_save_dir ./data/aime2024

docker exec -it opsd_siqi bash
cd /workspace/dev/OPSD

# kill program running on port 8082
kill -9 $(lsof -t -i:8082)
