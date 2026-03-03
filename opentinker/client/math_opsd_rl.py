#!/usr/bin/env python3
"""On-Policy Self-Distillation (OPSD) client for Math tasks."""

import hydra
from omegaconf import OmegaConf, open_dict

from utils.http_training_client import ServiceClient, SchedulerClient
from opentinker.environment.math import MathGame
from opentinker.environment.game_stats_client import GameStatsClient
from utils.utils import resolve_paths_in_config
from utils.scheduler_client_lifecycle import get_lifecycle_manager
from opentinker.environment.math.math_env import MathGameEnvironment


@hydra.main(config_path="client_config", config_name="math_opsd_param.yaml")
def main(args):
    args = resolve_paths_in_config(args)
    lifecycle = get_lifecycle_manager()

    algo = args.algorithm
    print("=" * 60)
    print("On-Policy Self-Distillation (OPSD) — Math")
    print(f"  adv_estimator          : {algo.adv_estimator}")
    print(f"  rollout_n              : {args.rollout_n}")
    print(f"  use_kl_in_advantage    : {algo.get('use_kl_in_advantage', False)}")
    print(f"  disable_rl_reward      : {algo.get('disable_rl_reward', False)}")
    print(f"  kl_penalty_coef        : {algo.get('kl_penalty_coef', 0.1)}")
    print(f"  algorithm.opsd.enable  : {algo.get('opsd', {}).get('enable', False)}")
    print(f"  algorithm.opsd.teacher_mode : {algo.get('opsd', {}).get('teacher_mode', 'fixed')}")
    print(f"  teacher_model_path     : {args.get('teacher_model_path', None)}")
    print("=" * 60)

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

    env_endpoint = args.interaction.config.env_endpoint
    env = MathGameEnvironment(
        game_class=MathGame,
        config=args,
        data_paths=[args.data_path],
        val_data_paths=[args.val_data_path] if args.val_data_path else None,
        job_id=job_id,
    )

    game_stats = GameStatsClient(env_endpoint, job_id=env.job_id)
    if game_stats.health_check():
        game_stats.reset_all()
        print(f"✓ Connected to math server at {env_endpoint}")
    else:
        game_stats = None
        print(f"⚠ Math server not responding at {env_endpoint}")

    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )

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
                    "n": int(args.rollout_n),
                },
            },
            "critic": {"model": {"path": args.tokenizer_path}},
            "trainer": {"n_gpus_per_node": args.num_gpus},
            "algorithm": {
                "adv_estimator": algo.adv_estimator,
                "use_kl_in_advantage": bool(algo.get("use_kl_in_advantage", True)),
                "disable_rl_reward": bool(algo.get("disable_rl_reward", False)),
                "kl_ctrl": {
                    "type": "fixed",
                    "kl_coef": float(algo.get("kl_penalty_coef", 0.1)),
                },
                "opsd": {
                    "enable": bool(algo.get("opsd", {}).get("enable", False)),
                    "teacher_mode": str(
                        algo.get("opsd", {}).get("teacher_mode", "fixed")
                    ),
                },
            },
        }
    )

    teacher_path = args.get("teacher_model_path", None)
    if teacher_path:
        with open_dict(server_cfg):
            server_cfg.actor_rollout_ref.ref = {"model": {"path": teacher_path}}
        print(f"✓ Teacher model: {teacher_path}")
    else:
        print("✓ Teacher model: initial student checkpoint (same model)")

    val_n = int(args.get("val_n", 1))
    val_temperature = float(args.get("val_temperature", 0.0))
    val_do_sample = (val_n > 1) or (val_temperature > 0.0)
    with open_dict(server_cfg):
        server_cfg.actor_rollout_ref.rollout.val_kwargs = OmegaConf.create(
            {
                "n": val_n,
                "do_sample": val_do_sample,
                "temperature": val_temperature,
            }
        )
    if val_n > 1:
        print(
            f"✓ Validation: pass@{val_n} mode (k=val_n, do_sample={val_do_sample}, temperature={val_temperature})"
        )
    else:
        print(
            f"✓ Validation: n=1 mode (do_sample={val_do_sample}, temperature={val_temperature})"
        )

    if hasattr(args, "multi_turn") and args.multi_turn:
        multi_turn_cfg = OmegaConf.to_container(args.multi_turn, resolve=True)
        with open_dict(server_cfg):
            server_cfg.actor_rollout_ref.rollout.multi_turn = multi_turn_cfg

    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    client.client.set_generation_config(generation_config)
    client.client.set_config(server_cfg, env)
    print("✓ Server config sent (OPSD config included)")

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
