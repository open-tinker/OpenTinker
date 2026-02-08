#!/usr/bin/env python3
"""SWE-Gym RL Training Client."""

from omegaconf import OmegaConf
import hydra

from utils.http_training_client import ServiceClient, SchedulerClient
from opentinker.environment.base_game_environment import GameEnvironment
from opentinker.environment.swegym import SWEGymGame
from opentinker.environment.game_stats_client import GameStatsClient
from utils.utils import resolve_paths_in_config
from utils.scheduler_client_lifecycle import get_lifecycle_manager


@hydra.main(config_path="client_config", config_name="swegym_param.yaml")
def main(args):
    args = resolve_paths_in_config(args)
    lifecycle = get_lifecycle_manager()

    print("=" * 60)
    print("Training with SWE-Gym Environment")
    print("=" * 60)

    # 1. Connect to scheduler and submit job
    scheduler_url = args.get("scheduler_url", "http://localhost:8780")
    #scheduler_api_key = args.get("scheduler_api_key", None)
    scheduler_api_key = "otk_027be14b14d5257faee92a45cc30a611"
    print(f"\nConnecting to scheduler at {scheduler_url}")
    if scheduler_api_key:
        print("✓ Using API key for authentication")
    else:
        print("⚠ No API key provided - authentication may fail if required")

    scheduler_client = SchedulerClient(
        scheduler_url=scheduler_url, api_key=scheduler_api_key
    )

    print("\nSubmitting training job to scheduler...")
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
    print(f"  GPUs: {job_result.get('gpu_ids')}")
    print(f"  Port: {job_result.get('port')}")
    print("=" * 60)

    # 2. Setup GameEnvironment with SWEGymGame
    interaction_config = args.interaction.config
    game_kwargs = {
        "dataset_name": interaction_config.get("dataset_name", "SWE-Gym/SWE-Gym"),
        "split": interaction_config.get("split", "train"),
        "repo_cache_dir": interaction_config.get("repo_cache_dir", "/tmp/swegym/repos"),
        "timeout_s": interaction_config.get("timeout_s", 6000),
        "apply_test_patch": interaction_config.get("apply_test_patch", True),
        "run_pass_to_pass": interaction_config.get("run_pass_to_pass", False),
        "test_command": interaction_config.get("test_command", "pytest"),
        "max_steps": interaction_config.get("max_total_steps", 6),
        "max_prompt_tokens": args.get("max_prompt_tokens", None),
        "tokenizer_path": args.get("tokenizer_path", None),
    }

    env_endpoint = interaction_config.env_endpoint

    print("\nSetting up GameEnvironment with SWEGymGame...")
    print(f"  Environment endpoint: {env_endpoint}")
    print(f"  Dataset: {game_kwargs['dataset_name']}")
    print(f"  Split: {game_kwargs['split']}")
    print(f"  Repo cache: {game_kwargs['repo_cache_dir']}")
    print(f"  Job ID for stats: {job_id}")

    env = GameEnvironment(
        game_class=SWEGymGame,
        config=args,
        game_kwargs=game_kwargs,
        job_id=job_id,
    )

    print("✓ Environment created")
    print(f"  Interaction config path: {env.get_interaction_config_path()}")

    # 3. Setup GameStatsClient
    game_stats = GameStatsClient(env_endpoint, job_id=env.job_id)
    if game_stats.health_check():
        print(f"✓ Connected to SWE-Gym server for metrics at {env_endpoint}")
        game_stats.reset_all()
    else:
        print(f"⚠ SWE-Gym server at {env_endpoint} not responding - metrics disabled")
        game_stats = None

    # 4. Connect to allocated server and train
    print(f"\nConnecting to allocated server at {server_url}")
    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )

    client.set_config(args, env)

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
    finally:
        env.cleanup()


if __name__ == "__main__":
    main()

