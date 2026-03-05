#!/usr/bin/env python3
"""ScienceWorld RL training client."""

from omegaconf import OmegaConf
import hydra

from utils.http_training_client import ServiceClient, SchedulerClient
from opentinker.environment.base_game_environment import GameEnvironment
from opentinker.environment.game_stats_client import GameStatsClient
from opentinker.environment.sciworld import SciWorldGame
from utils.utils import resolve_paths_in_config
from utils.scheduler_client_lifecycle import get_lifecycle_manager


@hydra.main(config_path="client_config", config_name="sciworld_param.yaml")
def main(args):
    args = resolve_paths_in_config(args)
    lifecycle = get_lifecycle_manager()

    enable_tracing = args.get("enable_tracing", False)
    if enable_tracing:
        try:
            from opentinker.utils.rollout_trace_saver import init_weave_tracing

            weave_project = args.get("weave_project", "sciworld-training")
            init_weave_tracing(
                project_name=weave_project,
                experiment_name=args.experiment_name,
                token2text=True,
            )
        except Exception as e:
            print(f"Failed to initialize Weave tracing: {e}")

    print("=" * 60)
    print("Training with ScienceWorld Environment")
    print("=" * 60)

    scheduler_url = args.get("scheduler_url", "http://localhost:8780")
    scheduler_api_key = args.get("scheduler_api_key", None)

    print(f"\nConnecting to scheduler at {scheduler_url}")
    if scheduler_api_key:
        print("Using API key for authentication")
    else:
        print("No API key provided - authentication may fail if scheduler requires it")

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

    print(f"\nJob {job_id} allocated")
    print(f"  Server URL: {server_url}")
    print(f"  GPUs: {job_result.get('gpu_ids')}")
    print(f"  Port: {job_result.get('port')}")
    print("=" * 60)

    interaction_config = args.interaction.config
    game_kwargs = {
        "max_steps": interaction_config.get("max_total_steps", 30),
        "split": interaction_config.get("split", "train"),
        "task_names": interaction_config.get("task_names"),
        "task_ids": interaction_config.get("task_ids"),
        "variation_indices": interaction_config.get("variation_indices"),
        "simplification_str": interaction_config.get("simplification_str", ""),
        "jar_path": interaction_config.get("jar_path"),
        "thread_base": interaction_config.get(
            "local_thread_base", SciWorldGame.DEFAULT_LOCAL_THREAD_BASE
        ),
    }

    env_endpoint = interaction_config.env_endpoint

    print("\nSetting up GameEnvironment with SciWorldGame...")
    print(f"  Environment endpoint: {env_endpoint}")
    print(f"  Max steps: {game_kwargs['max_steps']}")
    print(f"  Split: {game_kwargs['split']}")
    print(f"  Job ID for stats: {job_id}")

    env = GameEnvironment(
        game_class=SciWorldGame,
        config=args,
        game_kwargs=game_kwargs,
        job_id=job_id,
    )

    print("Environment created")
    print(f"  Interaction config path: {env.get_interaction_config_path()}")

    game_stats = GameStatsClient(env_endpoint, job_id=env.job_id)
    if game_stats.health_check():
        print(f"Connected to ScienceWorld server for metrics at {env_endpoint}")
        game_stats.reset_all()
    else:
        print(
            f"ScienceWorld server at {env_endpoint} not responding - metrics disabled"
        )
        game_stats = None

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
        print("Training completed")
        print(f"Final training metrics: {final_metrics}")
        if game_stats:
            print("\n" + "-" * 40)
            print("Final Game Statistics:")
            cumulative = game_stats.get_all_stats()
            if cumulative:
                print(f"  Total episodes: {cumulative.get('total_games', 0):.0f}")
        print("=" * 60)
    finally:
        env.cleanup()


if __name__ == "__main__":
    main()
