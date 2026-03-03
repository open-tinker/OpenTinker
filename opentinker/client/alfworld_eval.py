#!/usr/bin/env python3
"""Standalone ALFWorld evaluation script.

This script reuses OpenTinker's existing inference pipeline and ALFWorld game
integration to run evaluation and export:
1. Per-sample JSONL records with full prompt/response and score (0/1)
2. Final aggregated scores for Pick/Look/Clean/Heat/Cool/Pick2 and All
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from opentinker.environment.alfworld import ALFWorldGame
from opentinker.environment.inference_pipeline import (
    InferencePipeline,
    RemoteEnvironmentClient,
)


TASK_TYPE_TO_SUBTASK = {
    "pick_and_place_simple": "Pick",
    "look_at_obj_in_light": "Look",
    "pick_clean_then_place_in_recep": "Clean",
    "pick_heat_then_place_in_recep": "Heat",
    "pick_cool_then_place_in_recep": "Cool",
    "pick_two_obj_and_place": "Pick2",
}

SUBTASK_ORDER = ["Pick", "Look", "Clean", "Heat", "Cool", "Pick2"]


def _extract_task_type_from_path(game_path: str) -> str:
    # .../<task_type>-xxx/trial_xxx
    task_dir = Path(game_path).parent.name
    return task_dir.split("-")[0]


def _build_index_to_seed(num_games: int) -> dict[int, int]:
    """Build deterministic seed mapping so each episode index is selected once.

    ALFWorldGame.reset() selects one game via random.choice(game_paths) after
    random.seed(seed). We find a seed for each target index so evaluation can
    run a fixed episode list instead of random sampling.
    """
    index_to_seed: dict[int, int] = {}
    seed = 0
    max_trials = max(10000, num_games * 2000)

    while len(index_to_seed) < num_games and seed < max_trials:
        idx = random.Random(seed).randrange(num_games)
        if idx not in index_to_seed:
            index_to_seed[idx] = seed
        seed += 1

    if len(index_to_seed) != num_games:
        raise RuntimeError(
            "Failed to construct full deterministic seed mapping for fixed-order "
            f"evaluation. covered={len(index_to_seed)}/{num_games}, trials={max_trials}"
        )
    return index_to_seed


def build_fixed_order_samples(
    args: argparse.Namespace,
    split: str,
) -> tuple[list[dict[str, Any]], int, int]:
    """Build fixed-order samples from the split's complete episode list."""
    game = ALFWorldGame(
        max_steps=args.max_steps,
        split=split,
        num_games=args.num_games,
    )
    game_paths = game._get_cached_game_paths()
    total_available = len(game_paths)
    if total_available <= 0:
        raise RuntimeError(
            f"No available ALFWorld games found for split={split}, "
            f"num_games={args.num_games}"
        )

    if args.max_samples is not None and args.max_samples > 0:
        effective_max_samples = min(args.max_samples, total_available)
    else:
        effective_max_samples = total_available

    index_to_seed = _build_index_to_seed(total_available)
    system_prompt = game.get_system_prompt()

    samples: list[dict[str, Any]] = []
    for episode_index in tqdm(
        range(effective_max_samples), desc="Build Fixed Episode List"
    ):
        seed = index_to_seed[episode_index]
        selected_idx = random.Random(seed).randrange(total_available)
        if selected_idx != episode_index:
            raise RuntimeError(
                f"Seed mapping mismatch: seed={seed}, expected={episode_index}, got={selected_idx}"
            )

        user_prompt = game.get_user_message_with_state(seed=seed)
        game_path = game_paths[episode_index]
        task_type = _extract_task_type_from_path(game_path)

        samples.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "env_kwargs": {
                    "seed": seed,
                    "task_type": task_type,
                },
                "episode_index": episode_index,
                "episode_seed": seed,
                "episode_path": game_path,
            }
        )

    print(
        "Built fixed-order ALFWorld episodes: "
        f"{effective_max_samples}/{total_available} "
        f"(split={split}, num_games={args.num_games})"
    )
    return samples, effective_max_samples, total_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on ALFWorld and export JSONL + summary scores."
    )

    # Model backend
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Model path for offline vLLM mode (required if --vllm-server-url is not set).",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default=None,
        help="Existing vLLM server URL for server mode (e.g., http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer path (defaults to model path).",
    )

    # Environment/eval setup
    parser.add_argument(
        "--env-endpoint",
        type=str,
        default="http://127.0.0.1:8092",
        help="ALFWorld environment server endpoint for single-split mode.",
    )
    parser.add_argument(
        "--seen-env-endpoint",
        type=str,
        default=None,
        help="ALFWorld environment endpoint for eval_in_distribution.",
    )
    parser.add_argument(
        "--unseen-env-endpoint",
        type=str,
        default=None,
        help="ALFWorld environment endpoint for eval_out_of_distribution.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default="alfworld_eval",
        help="Job id used for environment stats isolation.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "eval_in_distribution", "eval_out_of_distribution", "both"],
        help="ALFWorld split. Use 'both' to run seen+unseen in one command.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help=(
            "Number of evaluation samples to run. "
            "Use <=0 to auto-run all available samples for the split."
        ),
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=-1,
        help="Pass-through to ALFWorldGame num_games (-1 for all available).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum environment steps per episode.",
    )
    parser.add_argument(
        "--alfworld-data",
        type=str,
        default=None,
        help=(
            "ALFWorld data root directory (contains json_2.1.1/ and logic/). "
            "If set, this script exports ALFWORLD_DATA to this path."
        ),
    )

    # Generation settings
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Total generation budget across all assistant turns.",
    )
    parser.add_argument(
        "--max-tokens-per-turn",
        type=int,
        default=512,
        help="Per-turn generation budget.",
    )
    parser.add_argument("--max-user-turns", type=int, default=20)
    parser.add_argument("--max-assistant-turns", type=int, default=20)
    parser.add_argument("--max-context-length", type=int, default=30000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)

    # Output
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Output JSONL path for single-split mode.",
    )
    parser.add_argument(
        "--seen-output-jsonl",
        type=str,
        default="outputs/alfworld_eval_seen.jsonl",
        help="Output JSONL path for eval_in_distribution.",
    )
    parser.add_argument(
        "--unseen-output-jsonl",
        type=str,
        default="outputs/alfworld_eval_unseen.jsonl",
        help="Output JSONL path for eval_out_of_distribution.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional output path for summary JSON (default: <output-jsonl>.summary.json).",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip failed samples and continue evaluation.",
    )

    args = parser.parse_args()
    if not args.model_path and not args.vllm_server_url:
        parser.error("Either --model-path or --vllm-server-url must be provided.")
    if args.split == "both":
        if not args.seen_env_endpoint or not args.unseen_env_endpoint:
            parser.error(
                "--split both requires both --seen-env-endpoint and --unseen-env-endpoint."
            )
    else:
        if not args.output_jsonl:
            parser.error(
                "--output-jsonl is required for single-split mode (split != both)."
            )
    return args


def infer_subtask(task_type: str | None, task_description: str | None) -> str:
    if task_type and task_type in TASK_TYPE_TO_SUBTASK:
        return TASK_TYPE_TO_SUBTASK[task_type]

    text = (task_description or "").lower()
    if "pick two" in text or "two" in text:
        return "Pick2"
    if "clean" in text:
        return "Clean"
    if "heat" in text:
        return "Heat"
    if "cool" in text:
        return "Cool"
    if "look at" in text or "light" in text:
        return "Look"
    return "Pick"


def extract_task_description(
    sample: dict[str, Any], env_info_trace: list[dict[str, Any]]
) -> str:
    for info in env_info_trace:
        task = info.get("task")
        if isinstance(task, str) and task.strip():
            return task.strip()

    user_prompt = ""
    for msg in sample.get("prompt", []):
        if msg.get("role") == "user":
            user_prompt = msg.get("content", "")
            break

    # Typical format starts with "Task: ..."
    match = re.search(r"Task:\s*(.+?)(?:\n\n|$)", user_prompt, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    return user_prompt.strip()


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _coerce_optional_bool(value[0])
    return None


def extract_episode_success(env_info_trace: list[dict[str, Any]]) -> tuple[bool | None, str]:
    for info in reversed(env_info_trace):
        if not isinstance(info, dict):
            continue
        if "success" in info:
            success = _coerce_optional_bool(info.get("success"))
            if success is not None:
                return success, "success"
        if "won" in info:
            won = _coerce_optional_bool(info.get("won"))
            if won is not None:
                return won, "won"
    return None, "reward_fallback"


def compute_summary(records: list[dict[str, Any]], elapsed_sec: float) -> dict[str, Any]:
    stats = {k: {"total": 0, "correct": 0} for k in SUBTASK_ORDER}
    overall_total = 0
    overall_correct = 0
    errors = 0

    for rec in records:
        if rec.get("error"):
            errors += 1
            continue
        subtask = rec.get("subtask", "Pick")
        score = int(rec.get("score", 0))

        if subtask not in stats:
            stats[subtask] = {"total": 0, "correct": 0}

        stats[subtask]["total"] += 1
        stats[subtask]["correct"] += score
        overall_total += 1
        overall_correct += score

    per_subtask = {}
    for name in SUBTASK_ORDER:
        total = stats[name]["total"]
        correct = stats[name]["correct"]
        acc = (correct / total) if total > 0 else None
        per_subtask[name] = {
            "correct": correct,
            "total": total,
            "score": acc,
        }

    all_score = (overall_correct / overall_total) if overall_total > 0 else None
    return {
        "elapsed_sec": elapsed_sec,
        "num_records": len(records),
        "num_evaluated": overall_total,
        "num_errors": errors,
        "subtasks": per_subtask,
        "all": {
            "correct": overall_correct,
            "total": overall_total,
            "score": all_score,
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print("ALFWorld Evaluation Summary")
    print("=" * 64)

    for name in SUBTASK_ORDER:
        item = summary["subtasks"][name]
        correct = item["correct"]
        total = item["total"]
        score = item["score"]
        score_str = f"{score * 100:.2f}%" if score is not None else "N/A"
        print(f"{name:>6}: {score_str:>8}  ({correct}/{total})")

    all_item = summary["all"]
    all_score = all_item["score"]
    all_score_str = f"{all_score * 100:.2f}%" if all_score is not None else "N/A"
    print("-" * 64)
    print(f"{'All':>6}: {all_score_str:>8}  ({all_item['correct']}/{all_item['total']})")
    print("-" * 64)
    print(
        f"Samples={summary['num_records']}, Evaluated={summary['num_evaluated']}, "
        f"Errors={summary['num_errors']}, Elapsed={summary['elapsed_sec']:.1f}s"
    )
    print("=" * 64)


def build_combined_split_summary(
    seen_summary: dict[str, Any], unseen_summary: dict[str, Any]
) -> dict[str, Any]:
    subtasks: dict[str, Any] = {}
    for name in SUBTASK_ORDER:
        seen_item = seen_summary["subtasks"][name]
        unseen_item = unseen_summary["subtasks"][name]
        combined_correct = seen_item["correct"] + unseen_item["correct"]
        combined_total = seen_item["total"] + unseen_item["total"]
        combined_score = (
            combined_correct / combined_total if combined_total > 0 else None
        )
        subtasks[name] = {
            "seen": seen_item,
            "unseen": unseen_item,
            "combined": {
                "correct": combined_correct,
                "total": combined_total,
                "score": combined_score,
            },
        }

    seen_all = seen_summary["all"]
    unseen_all = unseen_summary["all"]
    combined_all_correct = seen_all["correct"] + unseen_all["correct"]
    combined_all_total = seen_all["total"] + unseen_all["total"]
    combined_all_score = (
        combined_all_correct / combined_all_total if combined_all_total > 0 else None
    )

    return {
        "subtasks": subtasks,
        "overall": {
            "seen": seen_all,
            "unseen": unseen_all,
            "combined": {
                "correct": combined_all_correct,
                "total": combined_all_total,
                "score": combined_all_score,
            },
        },
    }


def print_dual_split_summary(dual_summary: dict[str, Any]) -> None:
    print("\n" + "=" * 86)
    print("ALFWorld Seen/Unseen Combined Summary")
    print("=" * 86)
    print("Subtask     Seen         Unseen       Combined")
    print("-" * 86)
    for name in SUBTASK_ORDER:
        item = dual_summary["subtasks"][name]
        seen = item["seen"]
        unseen = item["unseen"]
        combined = item["combined"]
        seen_s = f"{seen['score'] * 100:.2f}%" if seen["score"] is not None else "N/A"
        unseen_s = (
            f"{unseen['score'] * 100:.2f}%" if unseen["score"] is not None else "N/A"
        )
        combined_s = (
            f"{combined['score'] * 100:.2f}%"
            if combined["score"] is not None
            else "N/A"
        )
        print(
            f"{name:>6}   {seen_s:>8} ({seen['correct']:>3}/{seen['total']:<3})   "
            f"{unseen_s:>8} ({unseen['correct']:>3}/{unseen['total']:<3})   "
            f"{combined_s:>8} ({combined['correct']:>3}/{combined['total']:<3})"
        )

    print("-" * 86)
    overall = dual_summary["overall"]
    seen_all = overall["seen"]
    unseen_all = overall["unseen"]
    combined_all = overall["combined"]
    seen_all_s = (
        f"{seen_all['score'] * 100:.2f}%" if seen_all["score"] is not None else "N/A"
    )
    unseen_all_s = (
        f"{unseen_all['score'] * 100:.2f}%"
        if unseen_all["score"] is not None
        else "N/A"
    )
    combined_all_s = (
        f"{combined_all['score'] * 100:.2f}%"
        if combined_all["score"] is not None
        else "N/A"
    )
    print(
        f"{'All':>6}   {seen_all_s:>8} ({seen_all['correct']:>3}/{seen_all['total']:<3})   "
        f"{unseen_all_s:>8} ({unseen_all['correct']:>3}/{unseen_all['total']:<3})   "
        f"{combined_all_s:>8} ({combined_all['correct']:>3}/{combined_all['total']:<3})"
    )
    print("=" * 86)


async def run_single_split_eval(
    args: argparse.Namespace,
    pipeline: InferencePipeline,
    split: str,
    env_endpoint: str,
    output_jsonl_path: str,
) -> tuple[dict[str, Any], Path]:
    samples, effective_max_samples, total_available = build_fixed_order_samples(
        args, split=split
    )

    pipeline.env_client = RemoteEnvironmentClient(env_endpoint, job_id=args.job_id)
    healthy = await pipeline.env_client.health_check()
    if not healthy:
        raise RuntimeError(f"ALFWorld server not available at {env_endpoint}")
    print(f"✓ Connected to ALFWorld server at {env_endpoint} (split={split})")

    output_jsonl = Path(output_jsonl_path)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("")

    start = time.time()
    records: list[dict[str, Any]] = []

    for idx, sample in enumerate(tqdm(samples, desc="ALFWorld Eval")):
        env_kwargs = sample.get("env_kwargs", {})
        task_type = env_kwargs.get("task_type")
        episode_index = sample.get("episode_index")
        episode_seed = sample.get("episode_seed")
        episode_path = sample.get("episode_path")

        try:
            result = await pipeline.run_single_inference(
                messages=sample["prompt"],
                env_kwargs=env_kwargs,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                max_tokens_per_turn=args.max_tokens_per_turn,
            )

            env_info_trace = result.info.get("env_info", [])
            task_description = extract_task_description(sample, env_info_trace)
            subtask = infer_subtask(task_type, task_description)
            episode_success, score_source = extract_episode_success(env_info_trace)
            if episode_success is None:
                episode_success = bool(result.reward > 0)
            score = int(episode_success)

            assistant_turns = [
                m.get("content", "") for m in result.messages if m.get("role") == "assistant"
            ]
            user_turns = [
                m.get("content", "") for m in result.messages if m.get("role") == "user"
            ]
            action_trace = [x.get("action_taken") for x in env_info_trace if x.get("action_taken")]
            raw_reward_trace = [x.get("raw_reward") for x in env_info_trace if "raw_reward" in x]
            success_trace = [x.get("success") for x in env_info_trace if "success" in x]
            won_trace = [x.get("won") for x in env_info_trace if "won" in x]

            record = {
                "index": idx,
                "sample_id": result.sample_id,
                "split": split,
                "task_type": task_type,
                "subtask": subtask,
                "question": {
                    "system_prompt": sample["prompt"][0]["content"] if sample.get("prompt") else "",
                    "initial_user_prompt": sample["prompt"][1]["content"] if len(sample.get("prompt", [])) > 1 else "",
                    "full_prompt_text": result.prompt_text,
                    "task_description": task_description,
                },
                "model_answer": {
                    "assistant_turns": assistant_turns,
                    "full_response_text": result.response_text,
                    "full_messages": result.messages,
                },
                "ground_truth": {
                    "task_description": task_description,
                    "task_type": task_type,
                    "success_definition": "Environment success/won field (fallback: reward > 0)",
                },
                "score": score,
                "success": bool(episode_success),
                "score_source": score_source,
                "final_reward": result.reward,
                "done": result.done,
                "num_turns": result.num_turns,
                "env_kwargs": env_kwargs,
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "episode_path": episode_path,
                "action_trace": action_trace,
                "raw_reward_trace": raw_reward_trace,
                "success_trace": success_trace,
                "won_trace": won_trace,
                "env_info_trace": env_info_trace,
            }
        except Exception as e:
            if not args.continue_on_error:
                raise
            record = {
                "index": idx,
                "split": split,
                "task_type": task_type,
                "subtask": infer_subtask(task_type, None),
                "score": 0,
                "error": str(e),
                "env_kwargs": env_kwargs,
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "episode_path": episode_path,
            }

        records.append(record)
        with output_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    summary = compute_summary(records, elapsed_sec=elapsed)
    summary.update(
        {
            "model_path": args.model_path,
            "vllm_server_url": args.vllm_server_url,
            "tokenizer_path": args.tokenizer_path,
            "env_endpoint": env_endpoint,
            "split": split,
            "max_samples": effective_max_samples,
            "total_available": total_available,
            "max_steps": args.max_steps,
            "alfworld_data": os.environ.get("ALFWORLD_DATA"),
            "sample_mode": "fixed_episode_order",
            "scoring_rule": "success -> won -> reward>0 fallback",
        }
    )

    summary_path = output_jsonl.with_suffix(output_jsonl.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print_summary(summary)
    print(f"Per-sample JSONL saved to: {output_jsonl}")
    print(f"Summary JSON saved to: {summary_path}")
    return summary, summary_path


async def run_eval(args: argparse.Namespace) -> None:
    if args.alfworld_data:
        alfworld_data = str(Path(args.alfworld_data).expanduser().resolve())
        os.environ["ALFWORLD_DATA"] = alfworld_data
        print(f"Using ALFWORLD_DATA={alfworld_data}")

    if args.split == "both":
        split_runs = [
            (
                "eval_in_distribution",
                args.seen_env_endpoint,
                args.seen_output_jsonl,
            ),
            (
                "eval_out_of_distribution",
                args.unseen_env_endpoint,
                args.unseen_output_jsonl,
            ),
        ]
    else:
        split_runs = [(args.split, args.env_endpoint, args.output_jsonl)]

    pipeline = InferencePipeline(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        vllm_server_url=args.vllm_server_url,
        env_endpoint=split_runs[0][1],
        job_id=args.job_id,
        max_user_turns=args.max_user_turns,
        max_assistant_turns=args.max_assistant_turns,
        max_context_length=args.max_context_length,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    split_summaries: dict[str, dict[str, Any]] = {}
    summary_paths: dict[str, str] = {}
    for split_name, endpoint, output_jsonl in split_runs:
        summary, summary_path = await run_single_split_eval(
            args=args,
            pipeline=pipeline,
            split=split_name,
            env_endpoint=endpoint,
            output_jsonl_path=output_jsonl,
        )
        split_summaries[split_name] = summary
        summary_paths[split_name] = str(summary_path)

    if args.split == "both":
        dual_summary = build_combined_split_summary(
            seen_summary=split_summaries["eval_in_distribution"],
            unseen_summary=split_summaries["eval_out_of_distribution"],
        )
        print_dual_split_summary(dual_summary)

        final_summary = {
            "model_path": args.model_path,
            "vllm_server_url": args.vllm_server_url,
            "tokenizer_path": args.tokenizer_path,
            "alfworld_data": os.environ.get("ALFWORLD_DATA"),
            "sample_mode": "fixed_episode_order",
            "splits": {
                "eval_in_distribution": split_summaries["eval_in_distribution"],
                "eval_out_of_distribution": split_summaries["eval_out_of_distribution"],
            },
            "combined": dual_summary,
            "split_summary_paths": summary_paths,
        }
        summary_path = (
            Path(args.summary_json)
            if args.summary_json
            else Path("outputs/alfworld_eval_both.summary.json")
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2))
        print(f"Combined summary JSON saved to: {summary_path}")
    else:
        # Single split: optionally copy to --summary-json path if user provided one.
        if args.summary_json:
            single_summary_path = Path(args.summary_json)
            single_summary_path.parent.mkdir(parents=True, exist_ok=True)
            single_summary_path.write_text(
                json.dumps(split_summaries[args.split], ensure_ascii=False, indent=2)
            )
            print(f"Summary JSON saved to: {single_summary_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
