#!/usr/bin/env python3
"""Geo3K Multi-Turn Environment Server.

This script starts a Geo3K geometry problem server using Geo3KToolGame
for multi-turn verification-based interactions.

Usage:
    python geo3k_tool_server.py
    # Or with custom config:
    python geo3k_tool_server.py --port 8088 --max_retries 3
"""

import argparse
from opentinker.environment.base_game_server import run_game_server
from opentinker.environment.geo3k.geo3k_tool_game import Geo3KToolGame


def main():
    parser = argparse.ArgumentParser(description="Geo3K Multi-Turn Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8088, help="Server port")
    parser.add_argument(
        "--max_retries", type=int, default=3, help="Max verification attempts"
    )
    args = parser.parse_args()

    print("\nGeo3K Multi-Turn Server Configuration:")
    print(f"  Max retries: {args.max_retries}")
    print("\nFeedback format (verl-compatible):")
    print("  'Current parsed answer={answer} reward={0.0|1.0}'")
    print("\nReward structure:")
    print(f"  Correct: +{Geo3KToolGame.REWARD_CORRECT}")
    print(f"  Incorrect: {Geo3KToolGame.REWARD_INCORRECT}")
    print(f"  No improvement penalty: {Geo3KToolGame.PENALTY_NO_IMPROVEMENT}")

    run_game_server(
        game_class=Geo3KToolGame,
        host=args.host,
        port=args.port,
        stats_class=None,  # Use BaseGameStats
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
