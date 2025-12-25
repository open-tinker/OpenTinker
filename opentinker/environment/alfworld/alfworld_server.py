#!/usr/bin/env python3
"""ALFWorld Environment Server - Simplified launcher.

This script starts an ALFWorld game server using the generic base_game_server.

Usage:
    python alfworld_server.py
    # Or with custom config:
    python alfworld_server.py --port 8082 --max_steps 50
"""

import argparse
from opentinker.environment.base_game_server import run_game_server
from opentinker.environment.alfworld.alfworld_game import ALFWorldGame


def main():
    parser = argparse.ArgumentParser(description="ALFWorld Game Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8082, help="Server port")
    parser.add_argument("--config_path", type=str, default=None, 
                        help="Path to ALFWorld config file")
    parser.add_argument("--max_steps", type=int, default=50, 
                        help="Max steps per episode")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
                        help="Dataset split to use")
    parser.add_argument("--num_games", type=int, default=5,
                        help="Number of games to load (-1 = all games, e.g. 64 for faster loading)")
    args = parser.parse_args()
    
    print(f"\nALFWorld Game Configuration:")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Split: {args.split}")
    print(f"  Num games: {args.num_games if args.num_games > 0 else 'all'}")
    print(f"  Config: {args.config_path or 'default'}")
    print(f"\nReward structure:")
    print(f"  Success: +{ALFWorldGame.REWARD_SUCCESS}")
    print(f"  Failure: {ALFWorldGame.REWARD_FAILURE}")
    print(f"  Step penalty: {ALFWorldGame.REWARD_STEP}")
    print(f"  Invalid action: {ALFWorldGame.REWARD_INVALID_ACTION}")
    
    run_game_server(
        game_class=ALFWorldGame,
        host=args.host,
        port=args.port,
        config_path=args.config_path,
        max_steps=args.max_steps,
        split=args.split,
        num_games=args.num_games,
    )


if __name__ == "__main__":
    main()

