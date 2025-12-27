#!/usr/bin/env python3
"""
Gomoku Environment Inference Script

Uses the shared InferencePipeline to run inference on Gomoku games.

Usage:
    1. Start the game server:
       python opentinker/environment/gomoku/gomoku_server.py

    2. Run inference:
       python gomoku_inference.py \
           model_path=/path/to/checkpoint \
           env_endpoint=http://localhost:8089
"""

import hydra
from opentinker.environment.inference_pipeline import run_inference
from opentinker.environment.gomoku import GomokuGame


@hydra.main(
    config_path="client_config",
    config_name="gomoku_inference_config.yaml",
    version_base=None,
)
def main(args):
    """Run inference on Gomoku games."""
    print("=" * 60)
    print("Gomoku Environment Inference")
    print("=" * 60)

    if not args.model_path and not args.get("vllm_server_url"):
        raise ValueError("model_path or vllm_server_url is required")

    # Gomoku is multi-turn: max_user_turns should be > 0
    max_user_turns = args.multi_turn.get("max_user_turns", 50)
    max_assistant_turns = args.multi_turn.get("max_assistant_turns", 50)

    run_inference(
        model_path=args.get("model_path"),
        vllm_server_url=args.get("vllm_server_url"),
        tokenizer_path=args.get("tokenizer_path"),
        data_path=args.get("data_path"),  # None for dynamic generation
        game_class=GomokuGame,
        env_endpoint=args.env_endpoint,
        output_path=args.get("output_path"),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        max_tokens_per_turn=args.multi_turn.get("max_tokens_per_turn"),
        max_samples=args.get("max_samples", 10),
        max_user_turns=max_user_turns,
        max_assistant_turns=max_assistant_turns,
        max_context_length=args.get("max_context_length", 30000),
        tensor_parallel_size=args.get("tensor_parallel_size", 1),
        # GomokuGame kwargs
        board_size=args.get("board_size", 9),
    )

    if args.get("output_path"):
        print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
