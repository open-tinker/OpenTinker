#!/usr/bin/env python3
"""OPSD (On-Policy Self-Distillation) client for Math tasks.

Unified endpoint:
  always routes to /api/v1/train_step using standard ServiceClient.

Behavior switch:
  disable_rl_reward=True
      Pure JSD training (no RL reward signal).
  disable_rl_reward=False
      RL + JSD training.

Teacher prompt includes extra_info["answer"] as the reference solution.
Teacher source is configurable:
  - online (default): teacher uses current actor weights
  - initial_frozen: teacher uses initialization-time frozen reference weights
"""

import hydra
from omegaconf import OmegaConf, open_dict

from opentinker.client.utils.http_training_client import (
    ServiceClient,
)
from utils.http_training_client import SchedulerClient
from opentinker.environment.math import MathGame
from opentinker.environment.game_stats_client import GameStatsClient
from utils.utils import resolve_paths_in_config
from utils.scheduler_client_lifecycle import get_lifecycle_manager
from opentinker.environment.math.math_env import MathGameEnvironment
from opentinker.server.opsd_config import validate_opsd_modes


@hydra.main(config_path="client_config", config_name="math_opsd_param.yaml")
def main(args):
    args = resolve_paths_in_config(args)
    lifecycle = get_lifecycle_manager()

    # Prefer disable_rl_reward as the single switch.
    # Backward compatibility: if not explicitly provided, derive from legacy opsd_mode.
    legacy_mode = str(args.get("opsd_mode", "pure_jsd"))
    disable_rl_reward = bool(args.get("disable_rl_reward", legacy_mode == "pure_jsd"))
    opsd_teacher_source, opsd_loss_mode = validate_opsd_modes(args)

    print("=" * 60)
    print("OPSD — On-Policy Self-Distillation — Math")
    print(f"  disable_rl_reward: {disable_rl_reward}")
    print(f"  opsd_beta        : {args.get('opsd_beta', 0.5)}")
    print(f"  opsd_jsd_coef    : {args.get('opsd_jsd_coef', 1.0)}")
    print(f"  teacher_source   : {opsd_teacher_source}")
    print(f"  loss_mode        : {opsd_loss_mode}")
    print(f"  adv_estimator    : {args.adv_estimator}")
    print(f"  rollout_n        : {args.get('rollout_n', 1)}")
    if opsd_teacher_source == "initial_frozen":
        print("Teacher = initialization-time frozen reference model")
    else:
        print("Teacher = current student model (online, different prompt)")
    print("=" * 60)

    if disable_rl_reward:
        print("ℹ️  Pure JSD mode: math env server is NOT required")
    else:
        print("ℹ️  RL+JSD mode: math env server IS required for reward computation")

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
    env = MathGameEnvironment(
        game_class=MathGame,
        config=args,
        data_paths=[args.data_path],
        val_data_paths=[args.val_data_path] if args.val_data_path else None,
        job_id=job_id,
    )

    # ------------------------------------------------------------------
    # 3. Game stats client (optional, only meaningful in rl_jsd mode)
    # ------------------------------------------------------------------
    env_endpoint = args.interaction.config.env_endpoint
    game_stats = GameStatsClient(env_endpoint, job_id=env.job_id)
    if game_stats.health_check():
        game_stats.reset_all()
        print(f"✓ Connected to math server at {env_endpoint}")
    else:
        game_stats = None
        print(f"⚠ Math server not responding at {env_endpoint}")
        if not disable_rl_reward:
            print("  WARNING: RL+JSD mode requires the math server for reward computation!")

    # ------------------------------------------------------------------
    # 4. Build server config
    # ------------------------------------------------------------------
    # Build algorithm config block based on mode
    algo_cfg: dict = {
        "adv_estimator": args.adv_estimator,
        "use_opsd": True,           # select OPSDActorRolloutRefWorker
        "opsd_beta": float(args.get("opsd_beta", 0.5)),
        "use_opsd_jsd_in_advantage": True,
        "opsd_jsd_coef": float(args.get("opsd_jsd_coef", 1.0)),
        "opsd_teacher_source": opsd_teacher_source,
        "opsd_loss_mode": opsd_loss_mode,
        "use_kl_in_advantage": False,
        "disable_rl_reward": disable_rl_reward,
    }

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
                    "n": int(args.get("rollout_n", 1)),
                },
            },
            "critic": {"model": {"path": args.tokenizer_path}},
            "trainer": {"n_gpus_per_node": args.num_gpus},
            "algorithm": algo_cfg,
        }
    )

    val_n = int(args.get("val_n", 1))
    if val_n > 1:
        with open_dict(server_cfg):
            server_cfg.actor_rollout_ref.rollout.val_kwargs = OmegaConf.create(
                {"n": val_n, "do_sample": True}
            )
        print(f"✓ Validation: pass@{val_n} mode")

    if hasattr(args, "multi_turn") and args.multi_turn:
        multi_turn_cfg = OmegaConf.to_container(args.multi_turn, resolve=True)
        with open_dict(server_cfg):
            server_cfg.actor_rollout_ref.rollout.multi_turn = multi_turn_cfg

    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    # Always use standard client + /api/v1/train_step
    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )
    client.client.set_generation_config(generation_config)
    client.client.set_config(server_cfg, env)

    print(f"✓ Server config sent (disable_rl_reward={disable_rl_reward})")

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
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
