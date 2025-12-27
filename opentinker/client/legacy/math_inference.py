#!/usr/bin/env python3
"""
Math Environment Inference Script

Uses the shared InferencePipeline to run inference on math problems.

Usage:
    1. Start the game server:
       python opentinker/environment/math/math_server.py

    2. Run inference:
       python math_inference.py \
           model_path=/path/to/checkpoint \
           data_path=/ \
           output_path=/tmp/results.jsonl


"""

import hydra
from opentinker.environment.inference_pipeline import run_inference
from opentinker.environment.math import MathGame


@hydra.main(
    config_path="client_config",
    config_name="math_inference_config.yaml",
    version_base=None,
)
def main(args):
    """Run inference on math problems."""
    print("=" * 60)
    print("Math Environment Inference")
    print("=" * 60)

    if not args.model_path:
        raise ValueError("model_path is required")
    if not args.data_path:
        raise ValueError("data_path is required")

    run_inference(
        model_path=args.model_path,
        data_path=args.data_path,
        game_class=MathGame,
        env_endpoint=args.env_endpoint,
        output_path=args.get("output_path"),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        max_samples=args.get("max_samples"),
        max_user_turns=args.multi_turn.max_user_turns,
        max_assistant_turns=args.multi_turn.max_assistant_turns,
        tensor_parallel_size=args.get("tensor_parallel_size", 1),
    )

    if args.get("output_path"):
        print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
