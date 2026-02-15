#!/usr/bin/env python3
"""
On-Policy Self-Distillation (OPSD) Client

Implements the Self-Distilled Reasoner framework (arXiv:2601.18734) where a single
model acts as both teacher and student:
  - Student: p_S(.|x) — observes only the problem
  - Teacher: p_T(.|x, y*) — conditions on problem + ground-truth solution

No separate teacher model is needed. The privileged solution is inserted into the
prompt to create the teacher conditioning.

Usage:
    python self_distill_rl.py \
        tokenizer_path=Qwen/Qwen2.5-7B-Instruct \
        data_path=/path/to/math_train.parquet \
        self_distillation.loss_type=jsd
"""
import hydra
from omegaconf import OmegaConf

from utils.http_training_client import ServiceClient, SchedulerClient
from opentinker.environment.math import MathGame
from opentinker.environment.game_stats_client import GameStatsClient
from utils.utils import resolve_paths_in_config
from utils.scheduler_client_lifecycle import get_lifecycle_manager
from opentinker.self_distillation.self_distill_env import SelfDistillMathEnvironment


@hydra.main(config_path="client_config", config_name="self_distill_param.yaml")
def main(args):
    args = resolve_paths_in_config(args)
    lifecycle = get_lifecycle_manager()

    sd_cfg = args.get("self_distillation", {})

    print("=" * 60)
    print("On-Policy Self-Distillation (OPSD)")
    print("  arXiv:2601.18734 - Self-Distilled Reasoner")
    print("=" * 60)
    print(f"  Model:           {args.tokenizer_path}")
    print(f"  Loss type:       {sd_cfg.get('loss_type', 'jsd')}")
    print(f"  JSD beta:        {sd_cfg.get('beta', 0.5)}")
    print(f"  Solution key:    {sd_cfg.get('solution_key', 'ground_truth')}")
    print(f"  Validation:      every {args.test_freq} steps")
    print("=" * 60)

    # 1. Submit job to scheduler
    scheduler_client = SchedulerClient(
        scheduler_url=args.get("scheduler_url", "http://localhost:8780"),
        api_key=args.get("scheduler_api_key"),
    )

    job_result = scheduler_client.submit_job(
        config=OmegaConf.to_container(args, resolve=True),
        enable_agent_loop=False,
        wandb_key=args.get("wandb_key"),
        num_gpus=args.get("num_gpus"),
    )

    job_id = job_result["job_id"]
    server_url = job_result["server_url"]
    lifecycle.register_job(scheduler_client, job_id)

    print(f"Job {job_id} allocated at {server_url}")

    # 2. Setup environment with solution text passthrough
    env_endpoint = args.interaction.config.env_endpoint
    env = SelfDistillMathEnvironment(
        game_class=MathGame,
        config=args,
        data_paths=[args.data_path],
        val_data_paths=[args.val_data_path] if args.val_data_path else None,
        job_id=job_id,
    )
    print(
        f"Environment created, interaction config: {env.get_interaction_config_path()}"
    )

    # 3. Setup game stats client
    game_stats = GameStatsClient(env_endpoint, job_id=env.job_id)
    if game_stats.health_check():
        game_stats.reset_all()
        print(f"Connected to math server at {env_endpoint}")
    else:
        game_stats = None
        print(f"Math server not responding at {env_endpoint}")

    # 4. Connect to training server
    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )

    # 5. Set config with self-distillation overrides
    _set_self_distillation_config(client, args, env)

    # 5.5. Set output directory for validation JSON files
    if args.get("output_dir"):
        client.set_output_dir(args.output_dir)

    # 6. Train
    print(
        f"Starting self-distillation: steps={args.get('num_steps')}, epochs={args.get('num_epochs')}"
    )

    try:
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
        print(f"Self-distillation completed! Metrics: {final_metrics}")
    finally:
        env.cleanup()


def _set_self_distillation_config(client, args, env):
    """Build and send self-distillation config overrides to the server.

    Self-distillation does NOT need a separate teacher model (no ref_policy).
    The actor model itself serves as both teacher and student with different
    input conditioning.
    """
    sd_cfg = args.get("self_distillation", {})

    solution_template = sd_cfg.get(
        "solution_template", "\n\nReference solution: {solution}\n\n After understanding the reference solution, please try to solve this problem using your own approach below:"
    )

    # Build server config overrides
    server_cfg = OmegaConf.create(
        {
            "data": {
                "max_prompt_length": args.max_prompt_tokens,
                "max_response_length": args.max_new_tokens,
                "max_solution_length": sd_cfg.get("max_solution_length", 512),
            },
            "actor_rollout_ref": {
                "model": {
                    "path": args.tokenizer_path,
                },
                "actor": {
                    # Self-distillation flags (OPSD)
                    "use_self_distillation": True,
                    "self_distillation_loss_type": sd_cfg.get("loss_type", "jsd"),
                    "self_distillation_beta": sd_cfg.get("beta", 0.5),
                    "self_distillation_clip_advantage": sd_cfg.get("clip_advantage", 0.0),
                    "self_distillation_solution_template": solution_template,
                    # No standard distillation
                    "use_distillation": False,
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

    print(f"Self-distillation config sent to server:")
    print(f"  - loss_type: {sd_cfg.get('loss_type', 'jsd')}")
    print(f"  - beta: {sd_cfg.get('beta', 0.5)}")
    print(f"  - solution_template: {solution_template}")
    print(f"  - No separate teacher model (uses actor as teacher)")


if __name__ == "__main__":
    main()
