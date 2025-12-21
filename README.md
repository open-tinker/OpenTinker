<h1 align="center">🔧 OpenTinker</h1>

<h3 align="center">Democratizing Agentic Reinforcement Learning as a Service</h3>

<p align="center">
  <a href="https://open-tinker.github.io/opentinker-page/"><img src="https://img.shields.io/badge/🌐_Project_Page-blue?style=for-the-badge" alt="Project Page"/></a>
  <a href="https://wandb.ai/zsqzz/Open-Tinker?nw=nwuserzhusq20"><img src="https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=weights-and-biases&logoColor=black" alt="WandB"/></a>




## 🌟 Key Features

1. **Separation of Programming and Execution**
   - Users can perform RL training and inference without local GPU resources.
   - Built-in Distributed Training and Job Scheduling manage resources transparently.

2. **Separation of Environment and Training Code**
   - Simplifies the design of various agentic task environments.
   - Includes support for any single-turn and multi-turn agentic tasks.

3. **Seamless Transition from Training to Inference**
   - Environments and agentic workflows can be seamlessly connected to inference, allowing trained models to be directly applied.



## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU(s)
- Docker (recommended)

### Option 1: Docker Installation (Recommended)

```bash
# Pull the verl Docker image
docker pull verlai/verl@sha256:3ce56ff018516b28ab9c4f4fc09d3aa67589074495ace75e2674b720aa4d0e5d

# Create and run container
docker run -dit \
  --gpus all \
  --restart=no \
  --entrypoint /bin/bash \
  --net=host \
  --shm-size=10g \
  --cap-add=SYS_ADMIN \
  -v .:/workspace/dev \
  --name tinker \
  verlai/verl@sha256:3ce56ff018516b28ab9c4f4fc09d3aa67589074495ace75e2674b720aa4d0e5d
```

### Option 2: From Source

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:open-tinker/OpenTinker.git
cd OpenTinker

# Install OpenTinker
pip install -e .

# Install verl dependency
cd verl
pip install -e .
cd ..
```



## 🚀 Quick Start

### 1. Get Your IP Address

```bash
hostname -I
```

### 2. Start the Scheduler (Server Side)

```bash
bash opentinker/scripts/launch_scheduler.sh --scheduler-port <scheduler_port>
```

### 3. Start the Environment Server (Client Side)

**For Math Environment:**
```bash
# single turn
python opentinker/environment/math/math_server.py --port <env_port>

# multi turn tool call
python opentinker/environment/math/code_interpreter_math_server.py --port <env_port>
```

**For Gomoku Environment:**
```bash
python opentinker/environment/gomoku/gomoku_server.py --port <env_port>
```

### 4. Run Training/Inference (Client Side)

**Math RL:**

generate data:
```bash
python opentinker/data_preprocess/math_multiturn_w_interaction.py \
    --local_save_dir=<local_save_dir>
```


```bash
# single turn
python opentinker/client/math_client_unified.py \
    tokenizer_path=Qwen/Qwen2.5-1.5B \
    batch_size=16 \
    val_batch_size=64 \
    num_epochs=5 \
    save_freq=1000 \
    test_freq=5 \
    data_path=data/math_agentloop/train.parquet \
    val_data_path=data/math_agentloop/test.parquet \
    scheduler_url=http://<server_endpoint>:<scheduler_port> \
    interaction.config.env_port=<env_port> \
    interaction.config.env_host=<client_endpoint>

# multi turn tool ca
python opentinker/client/math_code_interpreter_client.py \
    tokenizer_path=Qwen/Qwen2.5-1.5B \
    batch_size=16 \
    val_batch_size=64 \
    num_epochs=5 \
    save_freq=1000 \
    test_freq=5 \
    scheduler_url=http://<server_endpoint>:<scheduler_port> \
    interaction.config.env_port=<env_port> \
    interaction.config.env_host=<client_endpoint>
```

**Gomoku RL (Multi-turn):**
```bash
python opentinker/client/gomoku_client.py \
    tokenizer_path=Qwen/Qwen2.5-3B-Instruct \
    batch_size=16 \
    val_batch_size=32 \
    num_epochs=5 \
    save_freq=1000 \
    test_freq=5 \
    scheduler_url=http://<server_endpoint>:<scheduler_port> \
    interaction.config.env_port=<env_port> \
    interaction.config.env_host=<client_endpoint>
```

**Math Inference:**
```bash
# single turn
python opentinker/client/math_inference_with_scheduler.py \
    model_path=<model_name> \
    data_path=data/math/test.parquet \
    output_path=./tmp/results.jsonl \
    max_samples=5 \
    env_endpoint=http://<client_endpoint>:<env_port> \
    scheduler_url=http://<server_endpoint>:<scheduler_port>

# multi turn tool call
python opentinker/client/math_code_interpreter_inference.py \
    model_path=<model_name> \
    data_path=data/math/test.parquet \
    output_path=./tmp/results.jsonl \
    max_samples=5 \
    env_endpoint=http://<client_endpoint>:<env_port> \
    scheduler_url=http://<server_endpoint>:<scheduler_port>
```

**Gomoku Inference:**
```bash
python opentinker/client/gomoku_inference_with_scheduler.py \
    model_path=<model_name> \
    output_path=./tmp/results.jsonl \
    max_samples=5 \
    env_endpoint=http://<client_endpoint>:<env_port> \
    scheduler_url=http://<server_endpoint>:<scheduler_port>
```



## 🔐 Authentication

OpenTinker includes a built-in authentication system to secure access to the scheduler API.

### Configuration

Edit `opentinker/scheduler/config/scheduler.yaml`:

```yaml
enable_auth: true   # Set to true to enable authentication, false to disable authentication.
user_db_path: "scheduler_users.db"
```

### Quick Registration

Run the interactive script to register a user and get an API key:

```bash
python opentinker/scheduler/register_user_example.py
```


For advanced usage (REST API registration, using the key) and detailed configuration, see the [Scheduler & Dashboard Guide](opentinker/scheduler/SCHEDULER_GUIDE.md#authentication).




## 🎮 Environments

### Math Environment (Data-driven)

Single-turn math reasoning environment where the model solves mathematical problems. It serves as a key example of a **data-driven environment**, loading data from parquet files.

We also support **multi-turn tool call** mode (Code Interpreter), where the model can iteratively generate and execute Python code to solve math problems. This enables more complex reasoning through code execution feedback.

| Component | Description |
|-----------|-------------|
| Server (Single-turn) | `opentinker/environment/math/math_server.py` |
| Server (Multi-turn Tool Call) | `opentinker/environment/math/code_interpreter_math_server.py` |
| Client (Single-turn) | `opentinker/client/math_client_unified.py` |
| Client (Multi-turn Tool Call) | `opentinker/client/math_code_interpreter_client.py` |
| Data | Parquet files with math problems |
| Reward | Correctness of mathematical solutions |

### Gomoku Environment (Data-free)

Multi-turn game environment where the model plays Gomoku against an opponent. It serves as a key example of a **data-free environment**, where the model gets prompts directly from the simulated environment.

| Component | Description |
|-----------|-------------|
| Server | `opentinker/environment/gomoku/gomoku_server.py` |
| Data | Generated from simulated games |
| Reward | Win/loss/draw outcomes |


## 📚 Documentation

- [Scheduler & Dashboard Guide](opentinker/scheduler/SCHEDULER_GUIDE.md) - Configuration, Usage, and Web Dashboard




## 📖 Citation
```
@misc{opentinker2025,
  title        = {OpenTinker: Democratizing Agentic Reinforcement Learning as a Service},
  author       = {Siqi Zhu and Jiaxuan You},
  year         = {2025},
  howpublished = {\url{https://github.com/open-tinker/OpenTinker}},
  note         = {GitHub repository}
}
```
