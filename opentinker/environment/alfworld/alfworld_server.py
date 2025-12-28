#!/usr/bin/env python3
"""ALFWorld Environment Server - Simplified launcher.

This script starts an ALFWorld game server using the generic base_game_server.

Usage:
    python alfworld_server.py
    # Or with custom config:
    python alfworld_server.py --port 8082 --max_steps 50

    # For parallelism (recommended for training):
    python alfworld_server.py --port 8091 --shards 8
    # This launches 8 independent servers on ports 8091..8098
"""

import os

# ============================================================
# CRITICAL: Disable GPU before importing anything else!
# ALFWorld is a TEXT-based environment - no GPU needed.
# Without this, each worker process allocates GPU memory → OOM!
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="ALFWorld Game Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8082, help="Server port")
    parser.add_argument(
        "--shards",
        type=int,
        default=8,
        help="Number of independent server processes to launch on consecutive ports. "
        "Example: --port 8091 --shards 8 will launch 8 servers on 8091..8098. "
        "Each shard handles requests independently for true parallelism.",
    )
    parser.add_argument(
        "--config_path", type=str, default=None, help="Path to ALFWorld config file"
    )
    parser.add_argument(
        "--max_steps", type=int, default=50, help="Max steps per episode"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=-1,
        help="Number of games to load (-1 = all games, e.g. 64 for faster loading)",
    )
    args = parser.parse_args()

    # Import here to avoid issues with multiprocessing
    from opentinker.environment.alfworld.alfworld_game import ALFWorldGame

    print("\nALFWorld Game Configuration:")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Split: {args.split}")
    print(f"  Num games: {args.num_games if args.num_games > 0 else 'all'}")
    print(f"  Shards: {args.shards}")
    print(f"  Config: {args.config_path or 'default'}")
    print("\nReward structure:")
    print(f"  Success: +{ALFWorldGame.REWARD_SUCCESS}")
    print(f"  Failure: {ALFWorldGame.REWARD_FAILURE}")
    print(f"  Step penalty: {ALFWorldGame.REWARD_STEP}")
    print(f"  Invalid action: {ALFWorldGame.REWARD_INVALID_ACTION}")

    # Sharded mode: launch N separate single-worker servers on consecutive ports.
    # Each shard is an independent process with its own in-memory instance registry.
    # This avoids cross-process state issues and provides true parallelism.
    if args.shards and args.shards > 1:
        print(
            f"\nStarting sharded mode: {args.shards} shards on ports {args.port}..{args.port + args.shards - 1}"
        )
        print(
            "Each shard is an independent process with its own in-memory instance registry.\n"
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
                    "--max_steps",
                    str(args.max_steps),
                    "--split",
                    args.split,
                    "--num_games",
                    str(args.num_games),
                ]
                if args.config_path is not None:
                    cmd.extend(["--config_path", args.config_path])

                # Run each shard as a child process.
                children.append(subprocess.Popen(cmd))
                time.sleep(0.2)

            print("Shards started. Press Ctrl+C to stop all shards.")
            # Wait until all children exit (or until interrupted).
            while True:
                for p in children:
                    rc = p.poll()
                    if rc is not None:
                        raise RuntimeError(
                            f"Shard process exited early with code {rc}: pid={p.pid}"
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

    # Single shard mode: run one server process
    from opentinker.environment.base_game_server import run_game_server

    run_game_server(
        game_class=ALFWorldGame,
        host=args.host,
        port=args.port,
        stats_class=None,
        config_path=args.config_path,
        max_steps=args.max_steps,
        split=args.split,
        num_games=args.num_games,
    )


if __name__ == "__main__":
    main()
