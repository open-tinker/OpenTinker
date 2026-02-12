#!/usr/bin/env python3
"""
On-Policy Distillation Client

Trains a student model to match a teacher model's output distribution
using forward KL divergence loss on student-generated rollouts.

Usage:
    python distill_rl.py \
        tokenizer_path=Qwen/Qwen2.5-0.5B-Instruct \
        data_path=/path/to/prompts.jsonl \
        distillation.teacher_model_path=Qwen/Qwen2.5-3B-Instruct
"""
import hydra
from omegaconf import OmegaConf

from utils.http_training_client import ServiceClient, SchedulerClient
from opentinker.environment.math import MathGame
from opentinker.environment.game_stats_client import GameStatsClient
from utils.utils import resolve_paths_in_config
from utils.scheduler_client_lifecycle import get_lifecycle_manager
from opentinker.environment.math.math_env import MathGameEnvironment


@hydra.main(config_path="client_config", config_name="distill_param.yaml")
def main(args):
    args = resolve_paths_in_config(args)
    lifecycle = get_lifecycle_manager()

    distill_cfg = args.get("distillation", {})

    print("=" * 60)
    print("On-Policy Distillation Training")
    print("=" * 60)
    print(f"  Student model:  {args.tokenizer_path}")
    print(f"  Teacher model:  {distill_cfg.get('teacher_model_path', 'N/A')}")
    print(f"  Distill mode:   {distill_cfg.get('distillation_mode', 'pure')}")
    print(f"  KL type:        {distill_cfg.get('distillation_kl_type', 'forward')}")
    print(f"  Validation:     every {args.test_freq} steps")
    print("=" * 60)

    # 1. Submit job to scheduler
    scheduler_client = SchedulerClient(
        scheduler_url=args.get("scheduler_url", "http://localhost:8780"),
        api_key=args.get("scheduler_api_key"),
    )

    job_result = scheduler_client.submit_job(
        config=OmegaConf.to_container(args, resolve=True),
        enable_agent_loop=False,  # No tool use needed for distillation
        wandb_key=args.get("wandb_key"),
        num_gpus=args.get("num_gpus"),
    )

    job_id = job_result["job_id"]
    server_url = job_result["server_url"]
    lifecycle.register_job(scheduler_client, job_id)

    print(f"✓ Job {job_id} allocated at {server_url}")

    # 2. Setup environment (use MathGame for prompt generation)
    env_endpoint = args.interaction.config.env_endpoint
    env = MathGameEnvironment(
        game_class=MathGame,
        config=args,
        data_paths=[args.data_path],
        val_data_paths=[args.val_data_path] if args.val_data_path else None,
        job_id=job_id,
    )
    print(
        f"✓ Environment created, interaction config: {env.get_interaction_config_path()}"
    )

    # 3. Setup game stats client
    game_stats = GameStatsClient(env_endpoint, job_id=env.job_id)
    if game_stats.health_check():
        game_stats.reset_all()
        print(f"✓ Connected to math server at {env_endpoint}")
    else:
        game_stats = None
        print(f"⚠ Math server not responding at {env_endpoint}")

    # 4. Connect to training server
    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )

    # 5. Set config with distillation overrides
    _set_distillation_config(client, args, env)

    # 6. Train
    print(
        f"Starting distillation: steps={args.get('num_steps')}, epochs={args.get('num_epochs')}"
    )

    try:
        # Validation uses distillation-specific metrics (KL divergence, cross-entropy)
        # instead of reward-based metrics when distillation mode is enabled on the server
        validate_before = args.val_data_path is not None
        final_metrics = client.fit(
            env=env,
            num_epochs=args.get("num_epochs"),
            num_steps=args.get("num_steps"),
            save_freq=args.save_freq,
            test_freq=args.test_freq,
            verbose=True,
            validate_before_training=validate_before,
            game_stats_client=game_stats,
        )
        print(f"Distillation completed! Metrics: {final_metrics}")
    finally:
        env.cleanup()


def _set_distillation_config(client, args, env):
    """Build and send distillation-aware config overrides to the server.

    Maps client-side distillation config to server-side config paths:
    - distillation.use_distillation → actor_rollout_ref.actor.use_distillation
    - distillation.distillation_mode → actor_rollout_ref.actor.distillation_mode
    - distillation.distillation_kl_type → actor_rollout_ref.actor.distillation_kl_type
    - distillation.teacher_model_path → actor_rollout_ref.ref.model.path
    """
    distill_cfg = args.get("distillation", {})

    # Build server config overrides
    server_cfg = OmegaConf.create(
        {
            "data": {
                "max_prompt_length": args.max_prompt_tokens,
                "max_response_length": args.max_new_tokens,
            },
            "actor_rollout_ref": {
                "model": {
                    "path": args.tokenizer_path,  # Student model
                },
                "actor": {
                    # Distillation flags
                    "use_distillation": distill_cfg.get("use_distillation", True),
                    "distillation_mode": distill_cfg.get("distillation_mode", "pure"),
                    "distillation_kl_type": distill_cfg.get("distillation_kl_type", "forward"),
                },
                "ref": {
                    "model": {
                        # Teacher model — loaded as RefPolicy with different weights
                        "path": distill_cfg.get("teacher_model_path", args.tokenizer_path),
                    },
                },
                "rollout": {
                    "tensor_model_parallel_size": 2 if args.num_gpus > 1 else 1,
                },
            },
            "critic": {
                "model": {
                    "path": args.tokenizer_path,
                },
            },
            "trainer": {
                "n_gpus_per_node": args.num_gpus,
            },
            # Force KL-related flags that trigger RefPolicy creation
            "algorithm": {
                "use_kl_in_reward": False,
            },
        }
    )

    # Add multi_turn config if present
    if hasattr(args, "multi_turn") and args.multi_turn:
        multi_turn_cfg = OmegaConf.to_container(args.multi_turn, resolve=True)
        server_cfg = OmegaConf.merge(
            server_cfg,
            OmegaConf.create(
                {"actor_rollout_ref": {"rollout": {"multi_turn": multi_turn_cfg}}}
            ),
        )

    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    client.client.set_generation_config(generation_config)
    client.client.set_config(server_cfg, env)

    print(f"✓ Distillation config sent to server:")
    print(f"  - use_distillation: {distill_cfg.get('use_distillation', True)}")
    print(f"  - distillation_mode: {distill_cfg.get('distillation_mode', 'pure')}")
    print(f"  - teacher_model_path: {distill_cfg.get('teacher_model_path')}")


if __name__ == "__main__":
    main()
