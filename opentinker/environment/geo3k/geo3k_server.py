#!/usr/bin/env python3
"""Geo3K Environment Server - Simplified launcher.

This script starts a Geo3K geometry problem server using the generic base_game_server.

Usage:
    python geo3k_server.py
    # Or with custom config:
    python geo3k_server.py --port 8082 --max_retries 0
"""

import argparse
from opentinker.environment.base_game_server import run_game_server
from opentinker.environment.geo3k.geo3k_game import Geo3KGame

# Geo3KGameStats is optional - falls back to BaseGameStats if not available
try:
    from opentinker.environment.geo3k.geo3k_stats import Geo3KGameStats
except ImportError:
    Geo3KGameStats = None


def main():
    parser = argparse.ArgumentParser(description="Geo3K Geometry Problem Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8082, help="Server port")
    parser.add_argument(
        "--max_retries",
        type=int,
        default=0,
        help="Max retry attempts (0 = single turn)",
    )
    args = parser.parse_args()

    print("\nGeo3K Game Configuration:")
    print(f"  Max retries: {args.max_retries}")
    print("\nReward structure:")
    print(f"  Correct: +{Geo3KGame.REWARD_CORRECT}")
    print(f"  Incorrect: {Geo3KGame.REWARD_INCORRECT}")

    if Geo3KGameStats:
        print("\nUsing Geo3KGameStats for tracking")
    else:
        print("\nUsing BaseGameStats (Geo3KGameStats not available)")

    run_game_server(
        game_class=Geo3KGame,
        host=args.host,
        port=args.port,
        stats_class=Geo3KGameStats,  # None falls back to BaseGameStats
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
