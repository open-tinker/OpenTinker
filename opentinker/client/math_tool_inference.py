#!/usr/bin/env python3
import hydra
from omegaconf import OmegaConf

from utils.http_training_client import InferenceSchedulerClient
from utils.scheduler_client_lifecycle import get_lifecycle_manager
from opentinker.environment.inference_pipeline import run_inference
from opentinker.environment.math.math_tool_game import CodeInterpreterMathGame
from opentinker.environment.game_stats_client import GameStatsClient


@hydra.main(config_path="client_config", config_name="math_code_interpreter_inference_config.yaml", version_base=None)
def main(args):
    """Run math code interpreter inference with scheduler-managed vLLM server."""
    lifecycle = get_lifecycle_manager()
    
    print("=" * 60)
    print("Math Code Interpreter Inference with Scheduler")
    print("=" * 60)
    
    if not args.model_path:
        raise ValueError("model_path is required")
    if not args.data_path:
        raise ValueError("data_path is required")
    
    # 1. Submit inference job to scheduler
    scheduler_client = InferenceSchedulerClient(
        scheduler_url=args.get("scheduler_url", "http://localhost:8780"),
        api_key=args.get("scheduler_api_key"),
    )
    
    print(f"Submitting inference job to scheduler...")
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
    game_stats = GameStatsClient(args.env_url, job_id=job_id)
    if game_stats.health_check():
        print(f"✓ Connected to code interpreter game server at {args.env_url}")
        game_stats.reset_all()  # Reset stats for this job before inference
    else:
        print(f"⚠ Game server not available at {args.env_url}, continuing without stats")
        print(f"  Make sure to start: python opentinker/environment/math/code_interpreter_math_server.py --port 8088")
        game_stats = None
    
    # 3. Run inference using the remote vLLM server
    print(f"\nRunning code interpreter inference on {args.data_path}...")
    print(f"  - Multi-turn: max_user_turns={args.multi_turn.max_user_turns}, max_assistant_turns={args.multi_turn.max_assistant_turns}")
    print(f"  - Max tokens: {args.max_new_tokens} total, {args.get('max_tokens_per_turn', 'unlimited')} per turn")
    
    results = run_inference(
        model_path=None,  # Not needed when using vllm_server_url
        vllm_server_url=vllm_server_url,
        tokenizer_path=args.get("tokenizer_path") or args.model_path,
        data_path=args.data_path,
        game_class=CodeInterpreterMathGame,
        env_endpoint=args.env_url,
        job_id=job_id,  # Pass job_id for stats isolation
        output_path=args.get("output_path"),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        max_tokens_per_turn=args.get("max_tokens_per_turn"),
        max_samples=args.get("max_samples"),
        max_user_turns=args.multi_turn.max_user_turns,
        max_assistant_turns=args.multi_turn.max_assistant_turns,
    )
    
    # 4. Log game stats after inference
    if game_stats:
        stats = game_stats.get_all_stats()
        print(f"\nCode Interpreter Stats (job_id={job_id}):")
        print(f"  Total samples: {stats.get('total_samples', 0)}")
        print(f"  Games completed: {stats.get('games_in_step', 0)}")
        print(f"  Mean reward: {stats.get('mean_final_reward', 0):.4f}")
        print(f"  Code executions: {stats.get('code_executions', 'N/A')}")
    
    if args.get("output_path"):
        print(f"\nResults saved to: {args.output_path}")
    
    print(f"\n{'='*60}")
    print("Inference completed! vLLM server will be automatically cleaned up.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
