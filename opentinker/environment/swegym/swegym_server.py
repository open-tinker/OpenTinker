#!/usr/bin/env python3
"""SWE-Gym Environment Server - simplified launcher.

This server exposes /reset and /step endpoints via base_game_server,
compatible with GymEnvironmentInteraction.
"""

import argparse
import os

from opentinker.environment.base_game_server import run_game_server
from opentinker.environment.swegym.swegym_game import SWEGymGame


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Invalid int for {name}: {value}")


def main():
    env_max_prompt_tokens = _env_int("SWEGYM_MAX_PROMPT_TOKENS")
    env_tokenizer_path = os.environ.get("SWEGYM_TOKENIZER_PATH") or None
    parser = argparse.ArgumentParser(description="SWE-Gym Game Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8090, help="Server port")
    parser.add_argument("--dataset", default="SWE-Gym/SWE-Gym", help="Dataset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument(
        "--repo-cache-dir",
        default="/tmp/swegym/repos",
        help="Local cache directory for repos",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout (seconds) for each test command",
    )
    parser.add_argument(
        "--apply-test-patch",
        dest="apply_test_patch",
        action="store_true",
        help="Apply test_patch on reset when available",
    )
    parser.add_argument(
        "--no-apply-test-patch",
        dest="apply_test_patch",
        action="store_false",
        help="Disable applying test_patch on reset",
    )
    parser.set_defaults(apply_test_patch=True)
    parser.add_argument(
        "--run-pass-to-pass",
        action="store_true",
        help="Run PASS_TO_PASS tests after FAIL_TO_PASS passes",
    )
    parser.add_argument(
        "--test-command",
        default="pytest",
        help="Test command prefix (default: pytest)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Max steps per episode (for prompt only)",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=env_max_prompt_tokens,
        help=(
            "Truncate problem_statement/hints to this many tokens "
            "(or set SWEGYM_MAX_PROMPT_TOKENS env var)"
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=env_tokenizer_path,
        help=(
            "Tokenizer path used for token-based truncation "
            "(or set SWEGYM_TOKENIZER_PATH env var)"
        ),
    )
    args = parser.parse_args()

    print("\nSWE-Gym Game Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Split: {args.split}")
    print(f"  Repo cache: {args.repo_cache_dir}")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Apply test_patch: {args.apply_test_patch}")
    print(f"  Run PASS_TO_PASS: {args.run_pass_to_pass}")
    print(f"  Test command: {args.test_command}")
    print(f"  Max prompt tokens: {args.max_prompt_tokens}")
    print(f"  Tokenizer path: {args.tokenizer_path}")

    run_game_server(
        game_class=SWEGymGame,
        host=args.host,
        port=args.port,
        dataset_name=args.dataset,
        split=args.split,
        repo_cache_dir=args.repo_cache_dir,
        timeout_s=args.timeout,
        apply_test_patch=args.apply_test_patch,
        run_pass_to_pass=args.run_pass_to_pass,
        test_command=args.test_command,
        max_steps=args.max_steps,
        max_prompt_tokens=args.max_prompt_tokens,
        tokenizer_path=args.tokenizer_path,
    )


if __name__ == "__main__":
    main()
