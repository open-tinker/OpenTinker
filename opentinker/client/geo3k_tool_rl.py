#!/usr/bin/env python3
"""Geo3K Multi-Turn Vision-Language RL Training Client.

This script launches Geo3K geometry problem training using vision-language models
with multi-turn verification. The model can submit answers and receive feedback
before giving the final answer.

Usage:
    # First, start the scheduler:
    bash opentinker/scripts/launch_scheduler.sh

    # Then start the game server:
    python opentinker/environment/geo3k/geo3k_tool_server.py --port 8088

    # Finally, run this training script:
    python opentinker/client/geo3k_tool_rl.py
"""

import hydra
from omegaconf import OmegaConf

from opentinker.client.utils.http_training_client import ServiceClient, SchedulerClient
from opentinker.environment.geo3k import Geo3KToolEnvironment
from opentinker.environment.game_stats_client import GameStatsClient
from opentinker.client.utils.utils import resolve_paths_in_config
from opentinker.client.utils.scheduler_client_lifecycle import get_lifecycle_manager


@hydra.main(config_path="client_config", config_name="geo3k_tool_param.yaml")
def main(args):
    args = resolve_paths_in_config(args)
    lifecycle = get_lifecycle_manager()

    print("=" * 60)
    print("Geo3K Multi-Turn Vision-Language Training")
    print("=" * 60)

    # 1. Submit job to scheduler
    print("\n[1/4] Submitting job to scheduler...")
    scheduler_client = SchedulerClient(
        scheduler_url=args.get("scheduler_url", "http://localhost:8780"),
        api_key=args.get("scheduler_api_key"),
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

    # 2. Setup Geo3K multi-turn VL environment
    print("\n[2/4] Setting up environment...")
    env_endpoint = args.interaction.config.env_endpoint

    max_retries = (
        args.multi_turn.get("max_assistant_turns", 3) - 1
    )  # -1 for initial attempt
    env = Geo3KToolEnvironment(
        config=args,
        data_paths=[args.data_path],
        val_data_paths=[args.val_data_path] if args.val_data_path else None,
        job_id=job_id,
        max_retries=max_retries,
    )
    print("✓ Geo3K multi-turn VL environment created")
    print(f"  - Interaction config: {env.get_interaction_config_path()}")
    print(f"  - Max retries: {max_retries}")

    # 3. Setup game stats client
    print("\n[3/4] Connecting to game server...")
    game_stats = GameStatsClient(env_endpoint, job_id=env.job_id)
    if game_stats.health_check():
        game_stats.reset_all()
        print(f"✓ Connected to game server at {env_endpoint}")
    else:
        game_stats = None
        print(f"⚠ Game server not responding at {env_endpoint}")
        print(
            f"  Make sure to start: python opentinker/environment/geo3k/geo3k_tool_server.py --port {args.interaction.config.env_port}"
        )

    # 4. Connect to training server and train
    print("\n[4/4] Starting training...")
    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )
    client.set_config(args, env)

    print("\nTraining configuration:")
    print(f"  - Algorithm: {args.algorithm}")
    print(f"  - Epochs: {args.get('num_epochs')}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Max assistant turns: {args.multi_turn.max_assistant_turns}")
    print(f"  - ADV estimator: {args.adv_estimator}")
    print(f"  - Rollout N: {args.rollout_n}")

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
        print("\n✓ Training completed!")
        print(f"Final metrics: {final_metrics}")
    finally:
        env.cleanup()


if __name__ == "__main__":
    main()
