#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import hydra

from utils.http_training_client import ServiceClient, SchedulerClient
from opentinker.environment.base_game_environment import GameEnvironment
from opentinker.environment.gomoku import GomokuGame
from opentinker.environment.game_stats_client import GameStatsClient
from utils.utils import resolve_paths_in_config
from utils.scheduler_client_lifecycle import get_lifecycle_manager


@hydra.main(config_path="client_config", config_name="gomoku_param.yaml")
def main(args):
    # Resolve paths to support both absolute and relative paths
    args = resolve_paths_in_config(args)
    
    # Get the lifecycle manager (this automatically enables cleanup handlers)
    lifecycle = get_lifecycle_manager()
    
    # Initialize Weave tracing (optional)
    enable_tracing = args.get("enable_tracing", False)
    if enable_tracing:
        try:
            from opentinker.utils.rollout_trace_saver import init_weave_tracing
            weave_project = args.get("weave_project", "gomoku-training")
            init_weave_tracing(
                project_name=weave_project,
                experiment_name=args.experiment_name,
                token2text=True,
            )
        except Exception as e:
            print(f"⚠ Failed to initialize Weave tracing: {e}")
    
    print("=" * 60)
    print("Training with Gomoku Environment")
    print("=" * 60)
    
    # 1. Connect to scheduler and submit job
    scheduler_url = args.get("scheduler_url", "http://localhost:8780")
    scheduler_api_key = args.get("scheduler_api_key", None)
    
    print(f"\nConnecting to scheduler at {scheduler_url}")
    if scheduler_api_key:
        print("✓ Using API key for authentication")
    else:
        print("⚠ No API key provided - authentication may fail if scheduler requires it")
    
    scheduler_client = SchedulerClient(
        scheduler_url=scheduler_url,
        api_key=scheduler_api_key
    )
    
    # Submit job with configuration
    print("\nSubmitting training job to scheduler...")
    job_result = scheduler_client.submit_job(
        config=OmegaConf.to_container(args, resolve=True),
        enable_agent_loop=True,  # REQUIRED for GenericAgentLoop
        wandb_key=args.get("wandb_key"),
        num_gpus=args.get("num_gpus"),
    )
    
    job_id = job_result["job_id"]
    server_url = job_result["server_url"]
    
    # Register job for automatic cleanup
    lifecycle.register_job(scheduler_client, job_id)
    
    print(f"\n✓ Job {job_id} allocated!")
    print(f"  Server URL: {server_url}")
    print(f"  GPUs: {job_result.get('gpu_ids')}")
    print(f"  Port: {job_result.get('port')}")
    print("=" * 60)
    
    # 2. Setup GameEnvironment with GomokuGame (job_id passed directly)
    interaction_config = args.interaction.config
    game_kwargs = {
        "board_size": interaction_config.get("board_size", 9),
        "max_total_steps": interaction_config.get("max_total_steps", 40),
    }
    
    # Use top-level env_url (with fallback to interaction.config.env_endpoint)
    env_url = args.get("env_url") or interaction_config.get("env_endpoint")
    
    print("\nSetting up GameEnvironment with GomokuGame...")
    print(f"  Environment URL: {env_url}")
    print(f"  Board size: {game_kwargs['board_size']}")
    print(f"  Job ID for stats: {job_id}")
    
    env = GameEnvironment(
        game_class=GomokuGame,
        config=args,
        game_kwargs=game_kwargs,
        job_id=job_id,  # Pass job_id directly
    )
    
    print(f"✓ Environment created")
    print(f"  Interaction config path: {env.get_interaction_config_path()}")
    
    # 3. Setup GameStatsClient for per-step metrics (use env.job_id for consistency)
    game_stats = GameStatsClient(env_url, job_id=env.job_id)
    if game_stats.health_check():
        print(f"✓ Connected to game server for metrics at {env_url}")
        game_stats.reset_all()  # Reset all stats before training
    else:
        print(f"⚠ Game server at {env_url} not responding - metrics disabled")
        game_stats = None
    
    # 4. Connect to allocated server
    print(f"\nConnecting to allocated server at {server_url}")
    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )
    
    # Set configuration on server
    client.set_config(args, env)
    
    # 5. Train with game stats tracking
    num_steps = args.get("num_steps", None)
    num_epochs = args.get("num_epochs", None)
    
    if num_steps:
        print(f"\nStarting training for {num_steps} steps...")
    elif num_epochs:
        print(f"\nStarting training for {num_epochs} epochs...")
    else:
        print("\nStarting training (1 epoch default)...")
        
    print(f"Checkpoint save frequency: {args.save_freq}")
    print(f"Validation frequency: {args.test_freq}")
    print("=" * 60)
    
    try:
        # Train with game stats tracking
        final_metrics = client.fit(
            env=env,
            num_epochs=num_epochs,
            num_steps=num_steps,
            save_freq=args.save_freq,
            test_freq=args.test_freq,
            verbose=True,
            validate_before_training=True,
            game_stats_client=game_stats,
        )
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Final training metrics: {final_metrics}")
        
        # Display final cumulative game stats
        if game_stats:
            print("\n" + "-" * 40)
            print("Final Game Statistics:")
            cumulative = game_stats.get_all_stats()
            if cumulative:
                print(f"  Total games: {cumulative.get('total_games', 0):.0f}")
                print(f"  Win rate: {cumulative.get('cumulative_win_rate', 0):.1%}")
                print(f"  Total wins: {cumulative.get('total_wins', 0):.0f}")
                print(f"  Total losses: {cumulative.get('total_losses', 0):.0f}")
        print("=" * 60)
        
    finally:
        # Clean up temporary files
        env.cleanup()


if __name__ == "__main__":
    main()
