#!/usr/bin/env python3
import hydra
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from torchdata.stateful_dataloader import StatefulDataLoader

from utils.http_training_client import ServiceClient, SchedulerClient
from opentinker.environment.base_game_environment import GameEnvironment
from opentinker.environment.base_data_generator import DynamicGameDataset, collate_fn
from opentinker.environment.math.math_tool_game import CodeInterpreterMathGame
from opentinker.environment.static_data_generator import StaticDatasetGenerator
from opentinker.environment.game_stats_client import GameStatsClient
from utils.utils import resolve_paths_in_config
from utils.scheduler_client_lifecycle import get_lifecycle_manager
from verl.trainer.main_ppo import create_rl_sampler
from opentinker.environment.math.math_tool_env import MathCodeInterpreterEnvironment

@hydra.main(config_path="client_config", config_name="math_code_interpreter_param.yaml")
def main(args):
    args = resolve_paths_in_config(args)
    lifecycle = get_lifecycle_manager()
    
    print("=" * 60)
    print("Math Training with Code Interpreter (Agent Loop)")
    print("=" * 60)
    
    # 1. Submit job to scheduler
    print("\n[1/4] Submitting job to scheduler...")
    scheduler_client = SchedulerClient(
        scheduler_url=args.get("scheduler_url", "http://localhost:8780"),
        api_key=args.get("scheduler_api_key")
    )
    
    job_result = scheduler_client.submit_job(
        config=OmegaConf.to_container(args, resolve=True),
        enable_agent_loop=True,
        wandb_key=args.get("wandb_key"),
        num_gpus=args.get("num_gpus"),
    )
    
    job_id = job_result["job_id"]
    server_url = job_result["server_url"]
    lifecycle.register_job(scheduler_client, job_id)
    
    print(f"✓ Job {job_id} allocated at {server_url}")
    
    # 2. Setup environment
    print("\n[2/4] Setting up environment...")
    # Use top-level env_url (with fallback to interaction.config.env_endpoint)
    env_url = args.get("env_url") or args.interaction.config.get("env_endpoint")
    env = MathCodeInterpreterEnvironment(
        game_class=CodeInterpreterMathGame,
        config=args,
        data_paths=[args.data_path],
        val_data_paths=[args.val_data_path] if args.val_data_path else None,
        job_id=job_id,
    )
    print(f"✓ Environment created")
    print(f"  - Interaction config: {env.get_interaction_config_path()}")
    print(f"  - Game server endpoint: {env_url}")
    
    # 3. Setup game stats client
    print("\n[3/4] Connecting to game server...")
    game_stats = GameStatsClient(env_url, job_id=env.job_id)
    if game_stats.health_check():
        game_stats.reset_all()
        print(f"✓ Connected to game server at {env_url}")
    else:
        game_stats = None
        print(f"⚠ Game server not responding at {env_url}")
        print(f"  Make sure to start: python opentinker/environment/math/code_interpreter_math_server.py")
    
    # 4. Connect to training server and train
    print("\n[4/4] Starting training...")
    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )
    client.set_config(args, env)
    
    print(f"\nTraining configuration:")
    print(f"  - Algorithm: {args.algorithm}")
    print(f"  - Steps: {args.get('num_steps')}")
    print(f"  - Epochs: {args.get('num_epochs')}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Max turns: {args.multi_turn.max_assistant_turns}")
    
    try:
        final_metrics = client.fit(
            env=env,
            num_epochs=args.get("num_epochs"),
            num_steps=args.get("num_steps"),
            save_freq=args.save_freq,
            test_freq=args.test_freq,
            verbose=True,
            validate_before_training=True,
            game_stats_client=game_stats,
        )
        print(f"\n✓ Training completed!")
        print(f"Final metrics: {final_metrics}")
    finally:
        env.cleanup()


if __name__ == "__main__":
    main()
