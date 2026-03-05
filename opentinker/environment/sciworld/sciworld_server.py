#!/usr/bin/env python3
"""ScienceWorld environment server."""

import argparse
import os
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    parser = argparse.ArgumentParser(description="ScienceWorld Game Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8082, help="Server port")
    parser.add_argument(
        "--shards",
        type=int,
        default=8,
        help="Number of independent server processes to launch on consecutive ports.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=30,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "dev", "test"],
        help="ScienceWorld variation split to sample from.",
    )
    parser.add_argument(
        "--task-name",
        action="append",
        default=None,
        help="Restrict to one or more ScienceWorld task names. Repeat for multiple tasks.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        type=int,
        default=None,
        help="Restrict to one or more ScienceWorld task ids. Repeat for multiple ids.",
    )
    parser.add_argument(
        "--variation",
        action="append",
        type=int,
        default=None,
        help="Restrict to one or more explicit variation indices.",
    )
    parser.add_argument(
        "--simplification-str",
        type=str,
        default="",
        help="ScienceWorld simplification string passed directly to env.load().",
    )
    parser.add_argument(
        "--jar-path",
        type=str,
        default=None,
        help="Optional path to the ScienceWorld JAR if not using the packaged default.",
    )
    parser.add_argument(
        "--thread-base",
        type=int,
        default=0,
        help="Base ScienceWorld thread number for this server process.",
    )
    parser.add_argument(
        "--threads-per-shard",
        type=int,
        default=256,
        help="Reserved ScienceWorld thread-number block size per shard.",
    )
    args = parser.parse_args()

    from opentinker.environment.sciworld.sciworld_game import SciWorldGame

    print("\nScienceWorld Game Configuration:")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Split: {args.split}")
    print(f"  Shards: {args.shards}")
    print(f"  Task names: {args.task_name or 'all'}")
    print(f"  Task ids: {args.task_id or 'none'}")
    print(f"  Variations: {args.variation or 'split default'}")
    print(f"  Simplification: {args.simplification_str or '(none)'}")
    print(f"  Thread base: {args.thread_base}")
    print(f"  Threads per shard: {args.threads_per_shard}")
    print("\nReward structure:")
    print(f"  Success: +{SciWorldGame.REWARD_SUCCESS}")
    print(f"  Failure: {SciWorldGame.REWARD_FAILURE}")
    print(f"  Step penalty: {SciWorldGame.REWARD_STEP}")
    print(f"  Invalid action: {SciWorldGame.REWARD_INVALID_ACTION}")

    if args.shards and args.shards > 1:
        print(
            f"\nStarting sharded mode: {args.shards} shards on ports "
            f"{args.port}..{args.port + args.shards - 1}"
        )
        children: list[subprocess.Popen] = []
        try:
            for i in range(args.shards):
                port_i = args.port + i
                thread_base_i = args.thread_base + i * args.threads_per_shard
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
                    "--simplification-str",
                    args.simplification_str,
                    "--thread-base",
                    str(thread_base_i),
                    "--threads-per-shard",
                    str(args.threads_per_shard),
                ]
                if args.jar_path is not None:
                    cmd.extend(["--jar-path", args.jar_path])
                for task_name in args.task_name or []:
                    cmd.extend(["--task-name", task_name])
                for task_id in args.task_id or []:
                    cmd.extend(["--task-id", str(task_id)])
                for variation in args.variation or []:
                    cmd.extend(["--variation", str(variation)])

                children.append(subprocess.Popen(cmd))
                time.sleep(0.2)

            print("Shards started. Press Ctrl+C to stop all shards.")
            while True:
                for proc in children:
                    rc = proc.poll()
                    if rc is not None:
                        raise RuntimeError(
                            f"Shard process exited early with code {rc}: pid={proc.pid}"
                        )
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            for proc in children:
                try:
                    proc.terminate()
                except Exception:
                    pass
            for proc in children:
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        return

    from opentinker.environment.base_game_server import run_game_server

    run_game_server(
        game_class=SciWorldGame,
        host=args.host,
        port=args.port,
        max_steps=args.max_steps,
        split=args.split,
        task_names=args.task_name,
        task_ids=args.task_id,
        variation_indices=args.variation,
        simplification_str=args.simplification_str,
        jar_path=args.jar_path,
        thread_base=args.thread_base,
    )


if __name__ == "__main__":
    main()
