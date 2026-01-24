#!/usr/bin/env python3
"""LLM User Simulator RL Training Client.

Train a conversational agent with LLM-based user simulation.

Usage:
    # Start the LLM user simulator server first:
    python -m opentinker.environment.llm_user_simulator.llm_user_server --port 8100 --shards 8

    # Run training:
    python llm_user_rl.py scheduler_url=http://localhost:8780 num_gpus=4
"""

from omegaconf import OmegaConf
import hydra

from utils.http_training_client import ServiceClient, SchedulerClient
from opentinker.environment.base_game_environment import GameEnvironment
from opentinker.environment.llm_user_simulator import LLMUserGame
from opentinker.environment.game_stats_client import GameStatsClient
from utils.utils import resolve_paths_in_config
from utils.scheduler_client_lifecycle import get_lifecycle_manager


@hydra.main(config_path="client_config", config_name="llm_user_param.yaml")
def main(args):
    args = resolve_paths_in_config(args)
    lifecycle = get_lifecycle_manager()

    print("=" * 60)
    print("Training with LLM User Simulator")
    print("=" * 60)

    # Connect to scheduler
    scheduler_url = args.get("scheduler_url", "http://localhost:8780")
    scheduler_api_key = args.get("scheduler_api_key", None)

    print(f"\nConnecting to scheduler at {scheduler_url}")
    scheduler_client = SchedulerClient(
        scheduler_url=scheduler_url, api_key=scheduler_api_key
    )

    # Submit job
    print("\nSubmitting training job...")
    job_result = scheduler_client.submit_job(
        config=OmegaConf.to_container(args, resolve=True),
        enable_agent_loop=True,
        wandb_key=args.get("wandb_key"),
        num_gpus=args.get("num_gpus"),
    )

    job_id = job_result["job_id"]
    server_url = job_result["server_url"]
    lifecycle.register_job(scheduler_client, job_id)

    print(f"\n✓ Job {job_id} allocated!")
    print(f"  Server URL: {server_url}")
    print("=" * 60)

    # Setup GameEnvironment
    interaction_config = args.interaction.config
    game_kwargs = {
        "max_turns": interaction_config.get("max_steps", 10),
    }

    env = GameEnvironment(
        game_class=LLMUserGame,
        config=args,
        game_kwargs=game_kwargs,
        job_id=job_id,
    )

    # Setup stats client
    env_endpoint = interaction_config.env_endpoint
    game_stats = GameStatsClient(env_endpoint, job_id=env.job_id)
    if game_stats.health_check():
        print(f"✓ Connected to LLM user simulator at {env_endpoint}")
        game_stats.reset_all()
    else:
        print(f"⚠ Server at {env_endpoint} not responding")
        game_stats = None

    # Connect to training server
    print(f"\nConnecting to server at {server_url}")
    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )

    client.set_config(args, env)

    # Train
    num_steps = args.get("num_steps", 1000)
    print(f"\nStarting training for {num_steps} steps...")
    print("=" * 60)

    try:
        final_metrics = client.fit(
            env=env,
            num_steps=num_steps,
            save_freq=args.save_freq,
            test_freq=args.test_freq,
            verbose=True,
            validate_before_training=True,
            game_stats_client=game_stats,
        )

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Final metrics: {final_metrics}")
        print("=" * 60)

    finally:
        env.cleanup()


if __name__ == "__main__":
    main()

