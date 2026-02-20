#!/usr/bin/env python3
"""On-Policy Distillation client for Math tasks.

Trains a student model with combined RL + KL-distillation advantage:

    A_final = A_base - kl_coef * (log π_student - log π_teacher)

where A_base is computed by the chosen adv_estimator (grpo / gae / grpo_per_step)
and the KL term is added per-token *after* advantage normalisation.

Key config knobs:
    adv_estimator    : "grpo" | "gae" | "grpo_per_step"
    use_kl_in_advantage: true
    kl_penalty_coef  : α  (default 0.1)
    teacher_model_path: path to teacher HF model (null → use initial student weights)
"""

import hydra
from omegaconf import OmegaConf, open_dict

from utils.http_training_client import ServiceClient, SchedulerClient
from opentinker.environment.math import MathGame
from opentinker.environment.game_stats_client import GameStatsClient
from utils.utils import resolve_paths_in_config
from utils.scheduler_client_lifecycle import get_lifecycle_manager
from opentinker.environment.math.math_env import MathGameEnvironment


@hydra.main(config_path="client_config", config_name="math_distill_param.yaml")
def main(args):
    args = resolve_paths_in_config(args)
    lifecycle = get_lifecycle_manager()

    print("=" * 60)
    print("On-Policy Distillation — Math")
    print(f"  adv_estimator     : {args.adv_estimator}")
    print(f"  use_kl_in_advantage: {args.get('use_kl_in_advantage', False)}")
    print(f"  kl_penalty_coef   : {args.get('kl_penalty_coef', 0.1)}")
    print(f"  teacher_model_path: {args.get('teacher_model_path', None)}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Submit job to scheduler
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Setup environment
    # ------------------------------------------------------------------
    env_endpoint = args.interaction.config.env_endpoint
    env = MathGameEnvironment(
        game_class=MathGame,
        config=args,
        data_paths=[args.data_path],
        val_data_paths=[args.val_data_path] if args.val_data_path else None,
        job_id=job_id,
    )

    # ------------------------------------------------------------------
    # 3. Game stats client (optional)
    # ------------------------------------------------------------------
    game_stats = GameStatsClient(env_endpoint, job_id=env.job_id)
    if game_stats.health_check():
        game_stats.reset_all()
        print(f"✓ Connected to math server at {env_endpoint}")
    else:
        game_stats = None
        print(f"⚠ Math server not responding at {env_endpoint}")

    # ------------------------------------------------------------------
    # 4. Build and send server config
    #
    # We call ServiceClient.set_config for the standard fields, then
    # send distillation-specific overrides via the underlying HTTP client.
    # Because set_config triggers server initialisation, ALL config must
    # be packed into a single call.  We therefore build the full server_cfg
    # here and call client.client.set_config() directly.
    # ------------------------------------------------------------------
    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )

    # Mirror what ServiceClient.set_config builds, plus distillation fields.
    server_cfg = OmegaConf.create(
        {
            "data": {
                "max_prompt_length": args.max_prompt_tokens,
                "max_response_length": args.max_new_tokens,
            },
            "actor_rollout_ref": {
                "model": {"path": args.tokenizer_path},
                "rollout": {
                    "tensor_model_parallel_size": 2 if args.num_gpus > 1 else 1,
                },
            },
            "critic": {"model": {"path": args.tokenizer_path}},
            "trainer": {"n_gpus_per_node": args.num_gpus},
            # Distillation algorithm config
            "algorithm": {
                "adv_estimator": args.adv_estimator,
                "use_kl_in_advantage": bool(args.get("use_kl_in_advantage", False)),
                "kl_ctrl": {
                    "type": "fixed",
                    "kl_coef": float(args.get("kl_penalty_coef", 0.1)),
                },
            },
        }
    )

    # Teacher model: if specified, point the reference policy at it.
    teacher_path = args.get("teacher_model_path", None)
    if teacher_path:
        with open_dict(server_cfg):
            server_cfg.actor_rollout_ref.ref = {"model": {"path": teacher_path}}
        print(f"✓ Teacher model: {teacher_path}")
    else:
        print("✓ Teacher model: initial student checkpoint (same model)")

    # multi_turn config
    if hasattr(args, "multi_turn") and args.multi_turn:
        multi_turn_cfg = OmegaConf.to_container(args.multi_turn, resolve=True)
        with open_dict(server_cfg):
            server_cfg.actor_rollout_ref.rollout.multi_turn = multi_turn_cfg

    # Send generation config first (separate endpoint)
    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    client.client.set_generation_config(generation_config)

    # Send the full merged config (single call — triggers server init)
    client.client.set_config(server_cfg, env)
    print("✓ Server config sent (distillation enabled)")

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print(
        f"Starting training: steps={args.get('num_steps')}, epochs={args.get('num_epochs')}"
    )
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
