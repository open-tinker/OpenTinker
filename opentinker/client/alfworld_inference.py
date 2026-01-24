#!/usr/bin/env python3
"""ALFWorld Inference Script.

This script runs inference/evaluation on trained ALFWorld models.

Usage:
    # Start ALFWorld environment server first (in another terminal):
    python -m opentinker.environment.alfworld.alfworld_server --port 8091 --split eval_in_distribution

    # Run inference with scheduler:
    python alfworld_inference.py \
        model_path=/path/to/checkpoint \
        scheduler_url=http://localhost:8089 \
        data_path=/path/to/eval_data.jsonl
"""

import hydra

from utils.http_training_client import InferenceSchedulerClient
from utils.scheduler_client_lifecycle import get_lifecycle_manager
from opentinker.environment.inference_pipeline import run_inference
from opentinker.environment.alfworld import ALFWorldGame
from opentinker.environment.game_stats_client import GameStatsClient


@hydra.main(
    config_path="client_config",
    config_name="alfworld_inference_config.yaml",
    version_base=None,
)
def main(args):
    """Run ALFWorld inference with scheduler-managed vLLM server."""
    lifecycle = get_lifecycle_manager()

    print("=" * 60)
    print("ALFWorld Inference with Scheduler")
    print("=" * 60)

    if not args.model_path:
        raise ValueError("model_path is required")

    # 1. Submit inference job to scheduler
    scheduler_client = InferenceSchedulerClient(
        scheduler_url=args.get("scheduler_url", "http://localhost:8089"),
        api_key=args.get("scheduler_api_key"),
    )

    print(f"\nModel: {args.model_path}")
    print(f"Scheduler: {args.scheduler_url}")
    print(f"Environment: {args.env_endpoint}")
    print(f"Split: {args.split}")

    print("\nSubmitting inference job to scheduler...")
    job_result = scheduler_client.submit_inference_job(
        model_path=args.model_path,
        tokenizer_path=args.get("tokenizer_path"),
        tensor_parallel_size=args.get("tensor_parallel_size", 1),
        num_gpus=args.get("num_gpus"),
        gpu_memory_utilization=args.get("gpu_memory_utilization", 0.9),
        max_model_len=args.get("max_model_len"),
        trust_remote_code=args.get("trust_remote_code", True),
    )

    job_id = job_result["job_id"]
    vllm_server_url = job_result["vllm_server_url"]

    # Register job for lifecycle cleanup
    lifecycle.register_job(scheduler_client, job_id)

    print(f"✓ Inference job {job_id} started at {vllm_server_url}")

    # 2. Setup GameStatsClient for per-step metrics (with job_id isolation)
    game_stats = GameStatsClient(args.env_endpoint, job_id=job_id)
    if game_stats.health_check():
        print(f"✓ Connected to ALFWorld server at {args.env_endpoint}")
        game_stats.reset_all()  # Reset stats for this job before inference
    else:
        print(
            f"⚠ ALFWorld server not available at {args.env_endpoint}, continuing without stats"
        )
        game_stats = None

    # 3. Run inference using the remote vLLM server
    data_path = args.get("data_path")
    if data_path:
        print(f"Running inference on {data_path}...")
    else:
        print(f"Running inference on ALFWorld {args.split} split...")

    results = run_inference(
        model_path=None,  # Not needed when using vllm_server_url
        vllm_server_url=vllm_server_url,
        tokenizer_path=args.get("tokenizer_path") or args.model_path,
        data_path=data_path,
        game_class=ALFWorldGame,
        env_endpoint=args.env_endpoint,
        job_id=job_id,  # Pass job_id for stats isolation
        output_path=args.get("output_path"),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        max_samples=args.get("max_samples"),
        max_user_turns=args.multi_turn.max_user_turns,
        max_assistant_turns=args.multi_turn.max_assistant_turns,
    )

    # 4. Log game stats after inference
    print("\n" + "=" * 60)
    print("Inference Results")
    print("=" * 60)

    if game_stats:
        stats = game_stats.get_all_stats()
        print(f"\nALFWorld Evaluation Stats (job_id={job_id}):")
        print(f"  Total episodes: {stats.get('total_games', 0)}")
        print(f"  Successes: {stats.get('total_wins', 0)}")
        print(f"  Failures: {stats.get('total_losses', 0)}")
        success_rate = stats.get('cumulative_win_rate', 0)
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Mean reward: {stats.get('mean_final_reward', 0):.4f}")
        print(f"  Mean steps: {stats.get('mean_steps', 0):.2f}")

    if results:
        print(f"\nProcessed {len(results)} samples")

    if args.get("output_path"):
        print(f"Results saved to: {args.output_path}")

    print(f"\n{'='*60}")
    print("Inference completed! vLLM server will be automatically cleaned up.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
