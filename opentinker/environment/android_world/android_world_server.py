#!/usr/bin/env python3
"""AndroidWorld Environment Server.

This script starts an AndroidWorld game server using the generic base_game_server.

Usage:
    python android_world_server.py
    python android_world_server.py --port 8082 --max_steps 50
    python android_world_server.py --port 8091 --shards 8
"""

import os
import argparse
import subprocess
import sys
import time

# Disable GPU if not needed, similar to ALFWorld
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    parser = argparse.ArgumentParser(description="AndroidWorld Game Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8082, help="Server port")
    parser.add_argument(
        "--shards",
        type=int,
        default=8,
        help="Number of independent server processes to launch on consecutive ports.",
    )
    parser.add_argument(
        "--config_path", type=str, default=None, help="Path to config file"
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
        help="Number of games to load",
    )
    # Multi-emulator support: each shard connects to a different emulator
    parser.add_argument(
        "--emulator_base_console_port",
        type=int,
        default=5556,
        help="Base console port for emulators. Shard i uses port base+i*2 (e.g., 5556, 5558, 5560, 5562)",
    )
    parser.add_argument(
        "--emulator_base_grpc_port",
        type=int,
        default=8554,
        help="Base gRPC port for emulators. Shard i uses port base+i (e.g., 8554, 8555, 8556, 8557)",
    )
    # Per-shard emulator ports (set automatically when launching shards)
    parser.add_argument("--emulator_console_port", type=int, default=None, help="(Internal) Console port for this shard")
    parser.add_argument("--emulator_grpc_port", type=int, default=None, help="(Internal) gRPC port for this shard")
    args = parser.parse_args()

    # Import here to avoid issues with multiprocessing
    from opentinker.environment.android_world.android_world_game import AndroidWorldGame

    print("\nAndroidWorld Game Configuration:")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Split: {args.split}")
    print(f"  Num games: {args.num_games if args.num_games > 0 else 'all'}")
    print(f"  Shards: {args.shards}")
    print(f"  Config: {args.config_path or 'default'}")
    if args.shards > 1:
        print(f"  Emulator base ports: console={args.emulator_base_console_port}, grpc={args.emulator_base_grpc_port}")

    if args.shards and args.shards > 1:
        print(
            f"\nStarting sharded mode: {args.shards} shards on ports {args.port}..{args.port + args.shards - 1}"
        )
        print("Each shard connects to a different emulator:")
        for i in range(args.shards):
            console_port = args.emulator_base_console_port + i * 2  # 5556, 5558, 5560, 5562
            grpc_port = args.emulator_base_grpc_port + i  # 8554, 8555, 8556, 8557
            print(f"  Shard {i}: server port {args.port + i}, emulator console={console_port}, grpc={grpc_port}")

        children: list[subprocess.Popen] = []
        try:
            for i in range(args.shards):
                port_i = args.port + i
                console_port_i = args.emulator_base_console_port + i * 2
                grpc_port_i = args.emulator_base_grpc_port + i
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
                    "--emulator_console_port",
                    str(console_port_i),
                    "--emulator_grpc_port",
                    str(grpc_port_i),
                ]
                if args.config_path is not None:
                    cmd.extend(["--config_path", args.config_path])

                children.append(subprocess.Popen(cmd))
                time.sleep(0.2)

            print("Shards started. Press Ctrl+C to stop all shards.")
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

    from opentinker.environment.base_game_server import run_game_server

    # For single shard mode, use explicit ports if provided, else fall back to base ports
    console_port = args.emulator_console_port or args.emulator_base_console_port
    grpc_port = args.emulator_grpc_port or args.emulator_base_grpc_port
    print(f"  Emulator: console={console_port}, grpc={grpc_port}")

    run_game_server(
        game_class=AndroidWorldGame,
        host=args.host,
        port=args.port,
        stats_class=None,
        config_path=args.config_path,
        max_steps=args.max_steps,
        split=args.split,
        num_games=args.num_games,
        emulator_console_port=console_port,
        emulator_grpc_port=grpc_port,
    )


if __name__ == "__main__":
    main()
