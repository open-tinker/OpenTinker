#!/usr/bin/env python3
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torchdata.stateful_dataloader import StatefulDataLoader

from utils.http_training_client import ServiceClient, SchedulerClient
from opentinker.environment.base_game_environment import GameEnvironment
from opentinker.environment.base_data_generator import DynamicGameDataset, collate_fn
from opentinker.environment.math import MathGame
from opentinker.environment.static_data_generator import StaticDatasetGenerator
from opentinker.environment.game_stats_client import GameStatsClient
from utils.utils import resolve_paths_in_config
from utils.scheduler_client_lifecycle import get_lifecycle_manager
from verl.trainer.main_ppo import create_rl_sampler
from opentinker.environment.math.math_env import MathGameEnvironment


@hydra.main(config_path="client_config", config_name="math_param.yaml")
def main(args):
    args = resolve_paths_in_config(args)
    lifecycle = get_lifecycle_manager()
    
    print("=" * 60)
    print("Math Training with GameEnvironment Pattern")
    print("=" * 60)
    
    # 1. Submit job to scheduler
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
    
    # 2. Setup environment (job_id is automatically handled)
    # Use top-level env_url (with fallback to interaction.config.env_endpoint)
    env_url = args.get("env_url") or args.interaction.config.get("env_endpoint")
    env = MathGameEnvironment(
        game_class=MathGame,
        config=args,
        data_paths=[args.data_path],
        val_data_paths=[args.val_data_path] if args.val_data_path else None,
        job_id=job_id,  # Pass job_id directly
    )
    print(f"✓ Environment created, interaction config: {env.get_interaction_config_path()}")
    
    # 3. Setup game stats client (use env.job_id for consistency)
    game_stats = GameStatsClient(env_url, job_id=env.job_id)
    if game_stats.health_check():
        game_stats.reset_all()
        print(f"✓ Connected to math server at {env_url}")
    else:
        game_stats = None
        print(f"⚠ Math server not responding at {env_url}")
    
    # 4. Connect to training server
    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )
    client.set_config(args, env)
    
    # 5. Train
    print(f"Starting training: steps={args.get('num_steps')}, epochs={args.get('num_epochs')}")
    
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
        print(f"Training completed! Metrics: {final_metrics}")
    finally:
        env.cleanup()


if __name__ == "__main__":
    main()
