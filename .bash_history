cd /workspace/dev
python -m pip install -e . --user
cd verl
python -m pip install -e . --user
cd ..
python opentinker/environment/swegym/swegym_server.py --port 8601
ray stop --force
exit
cd /workspace/dev
python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-1.5B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
echo "dev:x:1015:1016::/workspace/dev:/bin/bash" >> /etc/passwd
exit
cd /workspace/dev
python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-1.5B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-1.5B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
exit
cd /workspace/dev
python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-1.5B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-1.5B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
ls -lt /workspace/dev/.opentinker/logs | head
sed -n '1,200p'
ls
ls /tmp/opentinker/
ls /tmp/opentinker/logs
ls -lt /tmp/opentinker/logs/2026-02-07_09-44-15_job_502ca951_stderr.log
ls -lt /tmp/opentinker/logs/2026-02-07_09-44-15_job_502ca951_stderr.log | head
cat /tmp/opentinker/logs/2026-02-07_09-44-15_job_502ca951_stderr.log
exit
cd /workspace/dev
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-1.5B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
ls /tmp/opentinker/logs
cat /tmp/opentinker/logs/2026-02-07_09-49-10_job_a81c4ee9_stderr.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-1.5B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY

CUDA_VISIBLE_DEVICES=0 
dev@sn4622128200:~$ python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY

False
/usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py:829: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
0
dev@sn4622128200:~$ 



CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY

nvidia -si
nvidia-smi
exit
cd /workspace/dev
nvidia-smi
exit
nvidia-smi
eixt
exit
cd /workspace/dev
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-1.5B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
ls /tmp/opentinker/logs
cat /tmp/opentinker/logs/2026-02-07_09-58-58_job_f48af9bf_stderr.log
eixt
exit
sed -n '1,200p' /tmp/opentinker/logs/*_job_bcbbaa25_*err.log
nvitop
exit
cd /workspace/dev
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-1.5B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
pip install vllm
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-1.5B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
ray stop --force
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-1.5B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
ls /tmp/opentinker/logs
cat /tmp/opentinker/logs/2026-02-07_10-14-23_job_274c20a9_stderr.log
ray stop --force
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-1.5B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
ls /tmp/opentinker/logs
cat /tmp/opentinker/logs/2026-02-07_10-25-49_job_40ee6ea7_stderr.log
cat /tmp/opentinker/logs/2026-02-07_10-25-49_job_40ee6ea7_stdout.log
nvidia-smi
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-1.5B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
ray stop --force
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-1.5B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-32B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-32B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
CUDA_VISIBLE_DEVICES=0,1 python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-32B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
nvidia-smi
CUDA_VISIBLE_DEVICES=0,1 python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-32B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
ray stop --force
CUDA_VISIBLE_DEVICES=0,1 python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-32B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
CUDA_VISIBLE_DEVICES=0,1 python opentinker/client/swegym_rl.py   tokenizer_path=Qwen/Qwen2.5-32B   scheduler_url=http://localhost:8801   interaction.config.env_port=8601   interaction.config.env_host=localhost
exit
