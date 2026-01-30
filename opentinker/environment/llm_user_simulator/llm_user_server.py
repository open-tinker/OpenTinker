#!/usr/bin/env python3
"""LLM User Simulator Server.

This script starts an LLM user simulator server.

Usage:
    python llm_user_server.py --port 8100 --shards 8

    # With custom model:
    python llm_user_server.py --port 8100 --simulator_model gpt-4o-mini
"""

import argparse
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="LLM User Simulator Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8100, help="Server port")
    parser.add_argument(
        "--shards",
        type=int,
        default=4,
        help="Number of independent server processes on consecutive ports.",
    )
    parser.add_argument(
        "--simulator_model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for user simulation (e.g., gpt-4o-mini, gpt-4o)",
    )
    parser.add_argument(
        "--simulator_base_url",
        type=str,
        default=None,
        help="Custom API base URL (for local models like vLLM)",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="customer_service",
        choices=[
            "customer_service",
            "booking_assistant",
            "tech_support",
            "information_seeking",
        ],
        help="Type of user simulation task",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=10,
        help="Maximum conversation turns per episode",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for user simulator",
    )
    parser.add_argument(
        "--use_llm_judge",
        action="store_true",
        default=True,
        help="Use LLM-as-a-Judge for evaluation (default: True)",
    )
    parser.add_argument(
        "--no_llm_judge",
        action="store_true",
        help="Disable LLM-as-a-Judge, use keyword matching instead",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="Model for LLM judge (defaults to simulator_model)",
    )
    args = parser.parse_args()

    # Handle --no_llm_judge flag
    if args.no_llm_judge:
        args.use_llm_judge = False

    from opentinker.environment.llm_user_simulator.llm_user_game import LLMUserGame

    print("\nLLM User Simulator Configuration:")
    print(f"  Simulator model: {args.simulator_model}")
    print(f"  Base URL: {args.simulator_base_url or 'default (OpenAI)'}")
    print(f"  Task type: {args.task_type}")
    print(f"  Max turns: {args.max_turns}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Shards: {args.shards}")
    print(f"  LLM-as-a-Judge: {'enabled' if args.use_llm_judge else 'disabled'}")
    if args.use_llm_judge:
        print(f"  Judge model: {args.judge_model or args.simulator_model}")
    print("\nReward structure:")
    if args.use_llm_judge:
        print("  Using LLM Judge scoring (1-10 scale mapped to rewards)")
    else:
        print(f"  Success: +{LLMUserGame.REWARD_SUCCESS}")
        print(f"  Failure: {LLMUserGame.REWARD_FAILURE}")
    print(f"  Step penalty: {LLMUserGame.REWARD_STEP}")

    # Sharded mode
    if args.shards and args.shards > 1:
        print(
            f"\nStarting sharded mode: {args.shards} shards on ports {args.port}..{args.port + args.shards - 1}"
        )

        children: list[subprocess.Popen] = []
        try:
            for i in range(args.shards):
                port_i = args.port + i
                cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--host",
                    args.host,
                    "--port",
                    str(port_i),
                    "--shards",
                    "1",
                    "--simulator_model",
                    args.simulator_model,
                    "--task_type",
                    args.task_type,
                    "--max_turns",
                    str(args.max_turns),
                    "--temperature",
                    str(args.temperature),
                ]
                if args.simulator_base_url:
                    cmd.extend(["--simulator_base_url", args.simulator_base_url])
                if not args.use_llm_judge:
                    cmd.append("--no_llm_judge")
                if args.judge_model:
                    cmd.extend(["--judge_model", args.judge_model])

                children.append(subprocess.Popen(cmd))
                time.sleep(0.1)

            print("Shards started. Press Ctrl+C to stop all shards.")
            while True:
                for p in children:
                    rc = p.poll()
                    if rc is not None:
                        raise RuntimeError(
                            f"Shard exited early: pid={p.pid}, code={rc}"
                        )
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            for p in children:
                try:
                    p.terminate()
                except Exception:
                    pass
            for p in children:
                try:
                    p.wait(timeout=5)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
        return

    # Single shard mode
    from opentinker.environment.base_game_server import run_game_server

    run_game_server(
        game_class=LLMUserGame,
        host=args.host,
        port=args.port,
        stats_class=None,
        simulator_model=args.simulator_model,
        simulator_base_url=args.simulator_base_url,
        task_type=args.task_type,
        max_turns=args.max_turns,
        temperature=args.temperature,
        use_llm_judge=args.use_llm_judge,
        judge_model=args.judge_model,
    )


if __name__ == "__main__":
    main()
