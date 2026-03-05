#!/usr/bin/env python3
"""Standalone ScienceWorld evaluation script.

This script reuses OpenTinker's existing inference pipeline and ScienceWorld game
integration to run evaluation and export:
1. Per-sample JSONL records with full prompt/response and score (0/1)
2. Final aggregated scores per task name and overall
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from opentinker.environment.sciworld import SciWorldGame
from opentinker.environment.inference_pipeline import (
    InferencePipeline,
    RemoteEnvironmentClient,
)


def build_fixed_order_samples(
    args: argparse.Namespace,
    split: str,
) -> list[dict[str, Any]]:
    """Build fixed-order samples from the split's complete episode list."""
    game = SciWorldGame(
        max_steps=args.max_steps,
        split=split,
        task_names=args.task_name,
        task_ids=args.task_id,
        simplification_str=args.simplification_str,
        jar_path=args.jar_path,
        thread_base=50000,
    )

    try:
        task_pool = game._resolve_task_pool()
        pairs: list[tuple[str, int]] = []
        for task_name in sorted(task_pool):
            variations = game._resolve_variations_for_task(task_name)
            for variation in sorted(variations):
                pairs.append((task_name, variation))

        total_available = len(pairs)
        if total_available <= 0:
            raise RuntimeError(
                f"No available ScienceWorld episodes found for split={split}"
            )

        if args.max_samples is not None and args.max_samples > 0:
            effective = min(args.max_samples, total_available)
        else:
            effective = total_available

        pairs = pairs[:effective]
        system_prompt = game.get_system_prompt()
        samples: list[dict[str, Any]] = []

        for idx, (task_name, variation) in enumerate(
            tqdm(pairs, desc="Build Fixed Episode List")
        ):
            user_prompt = game.get_user_message_with_state(
                task_name=task_name, variation=variation
            )
            samples.append(
                {
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "env_kwargs": {
                        "task_name": task_name,
                        "variation": variation,
                    },
                    "episode_index": idx,
                    "task_name": task_name,
                    "variation": variation,
                }
            )

        print(
            f"Built fixed-order ScienceWorld episodes: "
            f"{effective}/{total_available} (split={split})"
        )
        return samples
    finally:
        game.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on ScienceWorld and export JSONL + summary scores."
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
        help="ScienceWorld environment server endpoint for single-split mode.",
    )
    parser.add_argument(
        "--dev-env-endpoint",
        type=str,
        default=None,
        help="ScienceWorld environment endpoint for dev split.",
    )
    parser.add_argument(
        "--test-env-endpoint",
        type=str,
        default=None,
        help="ScienceWorld environment endpoint for test split.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default="sciworld_eval",
        help="Job id used for environment stats isolation.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "dev", "test", "both"],
        help="ScienceWorld split. Use 'both' to run dev+test in one command.",
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
        "--max-steps",
        type=int,
        default=30,
        help="Maximum environment steps per episode.",
    )
    parser.add_argument(
        "--task-name",
        action="append",
        default=None,
        help="Restrict to specific task names. Repeat for multiple tasks.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        type=int,
        default=None,
        help="Restrict to specific task ids. Repeat for multiple ids.",
    )
    parser.add_argument(
        "--simplification-str",
        type=str,
        default="",
        help="ScienceWorld simplification string.",
    )
    parser.add_argument(
        "--jar-path",
        type=str,
        default=None,
        help="Optional path to the ScienceWorld JAR.",
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
    parser.add_argument("--max-user-turns", type=int, default=30)
    parser.add_argument("--max-assistant-turns", type=int, default=30)
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
        "--dev-output-jsonl",
        type=str,
        default="outputs/sciworld_eval_dev.jsonl",
        help="Output JSONL path for dev split.",
    )
    parser.add_argument(
        "--test-output-jsonl",
        type=str,
        default="outputs/sciworld_eval_test.jsonl",
        help="Output JSONL path for test split.",
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
        if not args.dev_env_endpoint or not args.test_env_endpoint:
            parser.error(
                "--split both requires both --dev-env-endpoint and --test-env-endpoint."
            )
    else:
        if not args.output_jsonl:
            parser.error(
                "--output-jsonl is required for single-split mode (split != both)."
            )
    return args


def extract_task_name(
    sample: dict[str, Any], env_info_trace: list[dict[str, Any]]
) -> str:
    for info in env_info_trace:
        task_name = info.get("task_name")
        if isinstance(task_name, str) and task_name.strip():
            return task_name.strip()
    return sample.get(
        "task_name", sample.get("env_kwargs", {}).get("task_name", "unknown")
    )


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


def extract_episode_success(
    env_info_trace: list[dict[str, Any]],
) -> tuple[bool | None, str]:
    for info in reversed(env_info_trace):
        if not isinstance(info, dict):
            continue
        if "success" in info:
            success = _coerce_optional_bool(info.get("success"))
            if success is not None:
                return success, "success"
        if "score" in info:
            score = info.get("score")
            if isinstance(score, (int, float)) and score >= 100.0:
                return True, "score"
    return None, "reward_fallback"


def compute_summary(
    records: list[dict[str, Any]], elapsed_sec: float
) -> dict[str, Any]:
    stats: dict[str, dict[str, int]] = {}
    overall_total = 0
    overall_correct = 0
    errors = 0

    for rec in records:
        if rec.get("error"):
            errors += 1
            continue
        task_name = rec.get("task_name", "unknown")
        score = int(rec.get("score", 0))

        if task_name not in stats:
            stats[task_name] = {"total": 0, "correct": 0}

        stats[task_name]["total"] += 1
        stats[task_name]["correct"] += score
        overall_total += 1
        overall_correct += score

    per_task = {}
    for name in sorted(stats.keys()):
        total = stats[name]["total"]
        correct = stats[name]["correct"]
        acc = (correct / total) if total > 0 else None
        per_task[name] = {
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
        "tasks": per_task,
        "all": {
            "correct": overall_correct,
            "total": overall_total,
            "score": all_score,
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print("ScienceWorld Evaluation Summary")
    print("=" * 64)

    for name in sorted(summary["tasks"].keys()):
        item = summary["tasks"][name]
        correct = item["correct"]
        total = item["total"]
        score = item["score"]
        score_str = f"{score * 100:.2f}%" if score is not None else "N/A"
        print(f"{name:>40}: {score_str:>8}  ({correct}/{total})")

    all_item = summary["all"]
    all_score = all_item["score"]
    all_score_str = f"{all_score * 100:.2f}%" if all_score is not None else "N/A"
    print("-" * 64)
    print(
        f"{'All':>40}: {all_score_str:>8}  ({all_item['correct']}/{all_item['total']})"
    )
    print("-" * 64)
    print(
        f"Samples={summary['num_records']}, Evaluated={summary['num_evaluated']}, "
        f"Errors={summary['num_errors']}, Elapsed={summary['elapsed_sec']:.1f}s"
    )
    print("=" * 64)


def build_combined_split_summary(
    dev_summary: dict[str, Any], test_summary: dict[str, Any]
) -> dict[str, Any]:
    all_task_names = sorted(
        set(dev_summary["tasks"].keys()) | set(test_summary["tasks"].keys())
    )

    empty = {"correct": 0, "total": 0, "score": None}
    tasks: dict[str, Any] = {}
    for name in all_task_names:
        dev_item = dev_summary["tasks"].get(name, empty)
        test_item = test_summary["tasks"].get(name, empty)
        combined_correct = dev_item["correct"] + test_item["correct"]
        combined_total = dev_item["total"] + test_item["total"]
        combined_score = (
            combined_correct / combined_total if combined_total > 0 else None
        )
        tasks[name] = {
            "dev": dev_item,
            "test": test_item,
            "combined": {
                "correct": combined_correct,
                "total": combined_total,
                "score": combined_score,
            },
        }

    dev_all = dev_summary["all"]
    test_all = test_summary["all"]
    combined_all_correct = dev_all["correct"] + test_all["correct"]
    combined_all_total = dev_all["total"] + test_all["total"]
    combined_all_score = (
        combined_all_correct / combined_all_total if combined_all_total > 0 else None
    )

    return {
        "tasks": tasks,
        "overall": {
            "dev": dev_all,
            "test": test_all,
            "combined": {
                "correct": combined_all_correct,
                "total": combined_all_total,
                "score": combined_all_score,
            },
        },
    }


def print_dual_split_summary(dual_summary: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("ScienceWorld Dev/Test Combined Summary")
    print("=" * 100)
    print(f"{'Task':>40}     Dev          Test         Combined")
    print("-" * 100)

    for name in sorted(dual_summary["tasks"].keys()):
        item = dual_summary["tasks"][name]
        dev = item["dev"]
        test = item["test"]
        combined = item["combined"]
        dev_s = f"{dev['score'] * 100:.2f}%" if dev["score"] is not None else "N/A"
        test_s = f"{test['score'] * 100:.2f}%" if test["score"] is not None else "N/A"
        combined_s = (
            f"{combined['score'] * 100:.2f}%"
            if combined["score"] is not None
            else "N/A"
        )
        print(
            f"{name:>40}   {dev_s:>8} ({dev['correct']:>3}/{dev['total']:<3})   "
            f"{test_s:>8} ({test['correct']:>3}/{test['total']:<3})   "
            f"{combined_s:>8} ({combined['correct']:>3}/{combined['total']:<3})"
        )

    print("-" * 100)
    overall = dual_summary["overall"]
    dev_all = overall["dev"]
    test_all = overall["test"]
    combined_all = overall["combined"]
    dev_all_s = (
        f"{dev_all['score'] * 100:.2f}%" if dev_all["score"] is not None else "N/A"
    )
    test_all_s = (
        f"{test_all['score'] * 100:.2f}%" if test_all["score"] is not None else "N/A"
    )
    combined_all_s = (
        f"{combined_all['score'] * 100:.2f}%"
        if combined_all["score"] is not None
        else "N/A"
    )
    print(
        f"{'All':>40}   {dev_all_s:>8} ({dev_all['correct']:>3}/{dev_all['total']:<3})   "
        f"{test_all_s:>8} ({test_all['correct']:>3}/{test_all['total']:<3})   "
        f"{combined_all_s:>8} ({combined_all['correct']:>3}/{combined_all['total']:<3})"
    )
    print("=" * 100)


async def run_single_split_eval(
    args: argparse.Namespace,
    pipeline: InferencePipeline,
    split: str,
    env_endpoint: str,
    output_jsonl_path: str,
) -> tuple[dict[str, Any], Path]:
    samples = build_fixed_order_samples(args, split=split)

    pipeline.env_client = RemoteEnvironmentClient(env_endpoint, job_id=args.job_id)
    healthy = await pipeline.env_client.health_check()
    if not healthy:
        raise RuntimeError(f"ScienceWorld server not available at {env_endpoint}")
    print(f"Connected to ScienceWorld server at {env_endpoint} (split={split})")

    output_jsonl = Path(output_jsonl_path)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("")

    start = time.time()
    records: list[dict[str, Any]] = []

    for idx, sample in enumerate(tqdm(samples, desc="ScienceWorld Eval")):
        env_kwargs = sample.get("env_kwargs", {})
        task_name_from_sample = sample.get("task_name", "unknown")

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
            task_name = extract_task_name(sample, env_info_trace)
            task_description = extract_task_description(sample, env_info_trace)
            episode_success, score_source = extract_episode_success(env_info_trace)
            if episode_success is None:
                episode_success = bool(result.reward > 0)
                score_source = "reward_fallback"
            score = int(episode_success)

            assistant_turns = [
                m.get("content", "")
                for m in result.messages
                if m.get("role") == "assistant"
            ]
            action_trace = [
                x.get("action_taken") for x in env_info_trace if x.get("action_taken")
            ]
            raw_reward_trace = [
                x.get("raw_reward") for x in env_info_trace if "raw_reward" in x
            ]
            success_trace = [x.get("success") for x in env_info_trace if "success" in x]
            score_trace = [x.get("score") for x in env_info_trace if "score" in x]

            record = {
                "index": idx,
                "sample_id": result.sample_id,
                "split": split,
                "task_name": task_name,
                "variation": env_kwargs.get("variation"),
                "question": {
                    "system_prompt": sample["prompt"][0]["content"]
                    if sample.get("prompt")
                    else "",
                    "initial_user_prompt": sample["prompt"][1]["content"]
                    if len(sample.get("prompt", [])) > 1
                    else "",
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
                    "task_name": task_name,
                    "success_definition": "Environment success field (fallback: score >= 100, then reward > 0)",
                },
                "score": score,
                "success": bool(episode_success),
                "score_source": score_source,
                "final_reward": result.reward,
                "done": result.done,
                "num_turns": result.num_turns,
                "env_kwargs": env_kwargs,
                "episode_index": sample.get("episode_index"),
                "action_trace": action_trace,
                "raw_reward_trace": raw_reward_trace,
                "success_trace": success_trace,
                "score_trace": score_trace,
                "env_info_trace": env_info_trace,
            }
        except Exception as e:
            if not args.continue_on_error:
                raise
            record = {
                "index": idx,
                "split": split,
                "task_name": task_name_from_sample,
                "variation": env_kwargs.get("variation"),
                "score": 0,
                "error": str(e),
                "env_kwargs": env_kwargs,
                "episode_index": sample.get("episode_index"),
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
            "max_samples": len(samples),
            "max_steps": args.max_steps,
            "sample_mode": "fixed_episode_order",
            "scoring_rule": "success -> score>=100 -> reward>0 fallback",
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
    if args.split == "both":
        split_runs = [
            ("dev", args.dev_env_endpoint, args.dev_output_jsonl),
            ("test", args.test_env_endpoint, args.test_output_jsonl),
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
            dev_summary=split_summaries["dev"],
            test_summary=split_summaries["test"],
        )
        print_dual_split_summary(dual_summary)

        final_summary = {
            "model_path": args.model_path,
            "vllm_server_url": args.vllm_server_url,
            "tokenizer_path": args.tokenizer_path,
            "sample_mode": "fixed_episode_order",
            "splits": {
                "dev": split_summaries["dev"],
                "test": split_summaries["test"],
            },
            "combined": dual_summary,
            "split_summary_paths": summary_paths,
        }
        summary_path = (
            Path(args.summary_json)
            if args.summary_json
            else Path("outputs/sciworld_eval_both.summary.json")
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2))
        print(f"Combined summary JSON saved to: {summary_path}")
    else:
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
