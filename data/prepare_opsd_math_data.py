#!/usr/bin/env python3
"""Prepare OPSD math train/test datasets in math_agentloop-compatible format.

Output schema (column order is fixed):
    ['level', 'type', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info']

This script downloads and converts:
    - Train: open-thoughts/OpenThoughts-114k (metadata config, math only)
    - Test:  math-ai/aime24, math-ai/aime25, FlagEval/HMMT_2025, meituan-longcat/AMO-Bench
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from datetime import datetime, timezone
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset

from verl.utils.reward_score.math_reward import (
    compute_score as math_compute_score,
    last_boxed_only_string,
    remove_boxed,
)


OUTPUT_COLUMNS = [
    "level",
    "type",
    "data_source",
    "prompt",
    "ability",
    "reward_model",
    "extra_info",
]

DEFAULT_DATA_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
DEFAULT_INSTRUCTION = (
    "Let's think step by step and output the final answer within \\boxed{}."
)


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def _has_boxed_instruction(question: str, instruction: str) -> bool:
    q = question.lower()
    if instruction.lower() in q:
        return True
    if "\\boxed" in q and "final answer" in q:
        return True
    if "output the final answer within \\boxed" in q:
        return True
    if "put your final answer in \\boxed" in q:
        return True
    return False


def build_prompt_text(question: str, instruction: str) -> str:
    question = _normalize_text(question)
    if not question:
        return instruction
    if _has_boxed_instruction(question, instruction):
        return question
    return f"{question} {instruction}"


def _extract_last_boxed_loose(text: str) -> str | None:
    """Extract content from the last '\\boxed' expression with loose parsing."""
    marker = "\\boxed"
    idx = text.rfind(marker)
    if idx < 0:
        return None

    j = idx + len(marker)
    n = len(text)
    while j < n and text[j].isspace():
        j += 1

    if j >= n:
        return None

    # Case 1: \boxed{...}
    if text[j] == "{":
        depth = 0
        start = j + 1
        k = j
        while k < n:
            ch = text[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:k].strip()
            k += 1
        return None

    # Case 2: \boxed 123 or \boxed8
    k = j
    while k < n and text[k] in " \t":
        k += 1
    start = k
    while k < n and text[k] not in " \t\r\n$.,;:)]}":
        k += 1
    token = text[start:k].strip()
    return token or None


def extract_ground_truth(text: Any, allow_raw_fallback: bool = True) -> str | None:
    """Extract final answer from solution text.

    Order:
      1) strict parser from existing reward utils
      2) loose boxed parser
      3) raw stripped text (optional)
    """
    content = _normalize_text(text)
    if not content:
        return None

    # Strict parser used by current math pipeline.
    try:
        boxed = last_boxed_only_string(content)
        if boxed is not None:
            gt = _normalize_text(remove_boxed(boxed))
            if gt:
                return gt
    except Exception:
        pass

    # Loose fallback for malformed boxed answers.
    loose = _extract_last_boxed_loose(content)
    if loose:
        return loose

    # Final fallback for datasets where answer is already plain text.
    if allow_raw_fallback:
        return content
    return None


def _build_record(
    *,
    level: str,
    type_name: str,
    prompt_text: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "level": level,
        "type": type_name,
        "data_source": DEFAULT_DATA_SOURCE,
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": extra_info,
    }


def _dataset_from_records(records: list[dict[str, Any]]) -> Dataset:
    if not records:
        return Dataset.from_dict({c: [] for c in OUTPUT_COLUMNS})
    ds = Dataset.from_list(records)
    return ds.select_columns(OUTPUT_COLUMNS)


def convert_train_openthoughts(
    *,
    instruction: str,
    cache_dir: str | None,
    max_train_samples: int | None,
) -> tuple[Dataset, dict[str, Any]]:
    ds = load_dataset(
        "open-thoughts/OpenThoughts-114k",
        "metadata",
        split="train",
        cache_dir=cache_dir,
    )

    records: list[dict[str, Any]] = []
    stats = {
        "total_rows": len(ds),
        "kept_rows": 0,
        "drop_non_math": 0,
        "drop_missing_question": 0,
        "drop_missing_teacher_hint": 0,
        "drop_missing_ground_truth": 0,
        "ground_truth_from_ground_truth_solution": 0,
        "ground_truth_from_deepseek_solution": 0,
    }

    for i, row in enumerate(ds):
        if row.get("domain") != "math":
            stats["drop_non_math"] += 1
            continue

        question = _normalize_text(row.get("problem"))
        if not question:
            stats["drop_missing_question"] += 1
            continue

        gt_source = "ground_truth_solution"
        primary_solution = _normalize_text(row.get("ground_truth_solution"))
        secondary_solution = _normalize_text(row.get("deepseek_solution"))
        teacher_hint = primary_solution or secondary_solution
        if not teacher_hint:
            stats["drop_missing_teacher_hint"] += 1
            continue

        ground_truth = extract_ground_truth(primary_solution, allow_raw_fallback=False)
        if ground_truth:
            stats["ground_truth_from_ground_truth_solution"] += 1
        else:
            gt_source = "deepseek_solution"
            ground_truth = extract_ground_truth(
                secondary_solution, allow_raw_fallback=False
            )
            if ground_truth:
                stats["ground_truth_from_deepseek_solution"] += 1

        if not ground_truth:
            stats["drop_missing_ground_truth"] += 1
            continue

        prompt_text = build_prompt_text(question, instruction)
        source = _normalize_text(row.get("source")) or "unknown"
        extra_info = {
            "answer": teacher_hint,
            "question": question,
            "index": i,
            "split": "train",
            "dataset_name": "open-thoughts/OpenThoughts-114k",
            "source": source,
            "domain": _normalize_text(row.get("domain")),
            "ground_truth_source": gt_source,
        }

        records.append(
            _build_record(
                level="OpenThoughts-Math",
                type_name=source,
                prompt_text=prompt_text,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        )

        if max_train_samples is not None and len(records) >= max_train_samples:
            break

    out = _dataset_from_records(records)
    stats["kept_rows"] = len(out)
    return out, stats


def _first_existing(row: dict[str, Any], keys: list[str]) -> str | None:
    for k in keys:
        if k in row and row[k] is not None:
            return str(row[k])
    return None


def convert_test_dataset(
    *,
    dataset_name: str,
    split: str,
    prompt_field: str,
    answer_field: str,
    type_name: str,
    instruction: str,
    cache_dir: str | None,
    keep_predicate: Any = None,
    orig_id_fields: list[str] | None = None,
) -> tuple[Dataset, dict[str, Any]]:
    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    orig_id_fields = orig_id_fields or ["id", "question_id"]

    records: list[dict[str, Any]] = []
    stats = {
        "dataset_name": dataset_name,
        "input_rows": len(ds),
        "kept_rows": 0,
        "drop_by_filter": 0,
        "drop_missing_question": 0,
        "drop_missing_ground_truth": 0,
    }

    for i, row in enumerate(ds):
        if keep_predicate is not None and not keep_predicate(row):
            stats["drop_by_filter"] += 1
            continue

        question = _normalize_text(row.get(prompt_field))
        if not question:
            stats["drop_missing_question"] += 1
            continue

        raw_answer = row.get(answer_field)
        ground_truth = extract_ground_truth(raw_answer, allow_raw_fallback=True)
        if not ground_truth:
            stats["drop_missing_ground_truth"] += 1
            continue

        prompt_text = build_prompt_text(question, instruction)
        answer_type = _normalize_text(row.get("answer_type")) or None
        extra_info = {
            "question": question,
            "index": i,
            "split": "test",
            "dataset_name": dataset_name,
            "orig_id": _first_existing(row, orig_id_fields),
            "answer_type": answer_type,
        }

        records.append(
            _build_record(
                level="Benchmark-Test",
                type_name=type_name,
                prompt_text=prompt_text,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        )

    out = _dataset_from_records(records)
    stats["kept_rows"] = len(out)
    return out, stats


def save_dataset(ds: Dataset, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.to_parquet(output_path)


def _load_parquet(path: str) -> Dataset:
    return load_dataset("parquet", data_files=path)["train"]


def validate_single_output(path: str, require_train_answer: bool) -> dict[str, Any]:
    ds = _load_parquet(path)
    errors: list[str] = []

    if ds.column_names != OUTPUT_COLUMNS:
        errors.append(
            f"Invalid columns: got {ds.column_names}, expected {OUTPUT_COLUMNS}"
        )

    # Data source consistency
    unique_sources = set(ds["data_source"]) if len(ds) > 0 else set()
    if any(src != DEFAULT_DATA_SOURCE for src in unique_sources):
        errors.append(f"Unexpected data_source values: {sorted(unique_sources)}")

    for i, row in enumerate(ds):
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            errors.append(f"row[{i}] invalid prompt format")
        elif not isinstance(prompt[0], dict) or prompt[0].get("role") != "user":
            errors.append(f"row[{i}] prompt[0] is not user message")

        reward_model = row.get("reward_model")
        gt = ""
        if isinstance(reward_model, dict):
            gt = _normalize_text(reward_model.get("ground_truth"))
        if not gt:
            errors.append(f"row[{i}] empty reward_model.ground_truth")

        if require_train_answer:
            answer = None
            extra_info = row.get("extra_info")
            if isinstance(extra_info, dict):
                answer = _normalize_text(extra_info.get("answer"))
            if not answer:
                errors.append(f"row[{i}] missing extra_info.answer for train sample")

        if len(errors) >= 20:
            break

    return {
        "path": path,
        "rows": len(ds),
        "valid": len(errors) == 0,
        "errors": errors,
    }


def validate_outputs(output_dir: str, paths: dict[str, str]) -> dict[str, Any]:
    results = {
        "files": {},
        "all_valid": True,
        "merged_size_check": {"valid": False, "expected": None, "actual": None},
    }

    for name, path in paths.items():
        require_train_answer = name == "train"
        res = validate_single_output(path, require_train_answer=require_train_answer)
        results["files"][name] = res
        if not res["valid"]:
            results["all_valid"] = False

    # Merged size check
    merged_rows = results["files"]["test_merged"]["rows"]
    expected = (
        results["files"]["test_aime24"]["rows"]
        + results["files"]["test_aime25"]["rows"]
        + results["files"]["test_hmmt2025"]["rows"]
        + results["files"]["test_amo_number"]["rows"]
    )
    size_ok = merged_rows == expected
    results["merged_size_check"] = {
        "valid": size_ok,
        "expected": expected,
        "actual": merged_rows,
    }
    if not size_ok:
        results["all_valid"] = False

    return results


def run_smoke_test(merged_test_path: str, sample_size: int, seed: int) -> dict[str, Any]:
    ds = _load_parquet(merged_test_path)
    if len(ds) == 0:
        return {"sample_size": 0, "ran": 0, "pass_count": 0, "errors": ["empty dataset"]}

    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: min(sample_size, len(ds))]

    pass_count = 0
    errors: list[str] = []
    for idx in idxs:
        row = ds[idx]
        gt = _normalize_text(row["reward_model"]["ground_truth"])
        pred = f"\\boxed{{{gt}}}"
        try:
            score = math_compute_score(pred, gt)
            if isinstance(score, (int, float)) and score >= 1.0:
                pass_count += 1
        except Exception as exc:
            errors.append(f"idx={idx}: {exc}")

    return {
        "sample_size": sample_size,
        "ran": len(idxs),
        "pass_count": pass_count,
        "errors": errors[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OPSD math train/test datasets.")
    parser.add_argument(
        "--output_dir",
        default="data/opsd_math",
        help="Output directory for converted parquet files.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional HuggingFace datasets cache directory.",
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="Instruction appended to question when not already present.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional cap on kept train samples (after filtering).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used in smoke test sampling.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    print("Converting OpenThoughts train split...")
    train_ds, train_stats = convert_train_openthoughts(
        instruction=args.instruction,
        cache_dir=args.cache_dir,
        max_train_samples=args.max_train_samples,
    )

    print("Converting AIME24 test split...")
    aime24_ds, aime24_stats = convert_test_dataset(
        dataset_name="math-ai/aime24",
        split="test",
        prompt_field="problem",
        answer_field="solution",
        type_name="aime24",
        instruction=args.instruction,
        cache_dir=args.cache_dir,
        orig_id_fields=["id"],
    )

    print("Converting AIME25 test split...")
    aime25_ds, aime25_stats = convert_test_dataset(
        dataset_name="math-ai/aime25",
        split="test",
        prompt_field="problem",
        answer_field="answer",
        type_name="aime25",
        instruction=args.instruction,
        cache_dir=args.cache_dir,
        orig_id_fields=["id"],
    )

    print("Converting HMMT2025 test split...")
    hmmt_ds, hmmt_stats = convert_test_dataset(
        dataset_name="FlagEval/HMMT_2025",
        split="train",
        prompt_field="question",
        answer_field="answer",
        type_name="hmmt2025",
        instruction=args.instruction,
        cache_dir=args.cache_dir,
        orig_id_fields=["id"],
    )

    print("Converting AMO-Bench (number-only) test split...")
    amo_ds, amo_stats = convert_test_dataset(
        dataset_name="meituan-longcat/AMO-Bench",
        split="test",
        prompt_field="prompt",
        answer_field="answer",
        type_name="amo_number",
        instruction=args.instruction,
        cache_dir=args.cache_dir,
        keep_predicate=lambda row: _normalize_text(row.get("answer_type")) == "number",
        orig_id_fields=["question_id"],
    )

    merged_test_ds = concatenate_datasets([aime24_ds, aime25_ds, hmmt_ds, amo_ds])

    paths = {
        "train": os.path.join(args.output_dir, "train_openthoughts_math.parquet"),
        "test_aime24": os.path.join(args.output_dir, "test_aime24.parquet"),
        "test_aime25": os.path.join(args.output_dir, "test_aime25.parquet"),
        "test_hmmt2025": os.path.join(args.output_dir, "test_hmmt2025.parquet"),
        "test_amo_number": os.path.join(args.output_dir, "test_amo_number.parquet"),
        "test_merged": os.path.join(args.output_dir, "test_merged.parquet"),
    }

    print("Saving parquet files...")
    save_dataset(train_ds, paths["train"])
    save_dataset(aime24_ds, paths["test_aime24"])
    save_dataset(aime25_ds, paths["test_aime25"])
    save_dataset(hmmt_ds, paths["test_hmmt2025"])
    save_dataset(amo_ds, paths["test_amo_number"])
    save_dataset(merged_test_ds, paths["test_merged"])

    print("Validating outputs...")
    validation = validate_outputs(args.output_dir, paths)

    print("Running smoke test...")
    smoke = run_smoke_test(paths["test_merged"], sample_size=10, seed=args.seed)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "output_dir": args.output_dir,
            "cache_dir": args.cache_dir,
            "instruction": args.instruction,
            "max_train_samples": args.max_train_samples,
            "seed": args.seed,
            "data_source": DEFAULT_DATA_SOURCE,
        },
        "stats": {
            "train_openthoughts": train_stats,
            "test_aime24": aime24_stats,
            "test_aime25": aime25_stats,
            "test_hmmt2025": hmmt_stats,
            "test_amo_number": amo_stats,
            "test_merged_rows": len(merged_test_ds),
        },
        "validation": validation,
        "smoke_test": smoke,
        "output_files": paths,
    }

    report_path = os.path.join(args.output_dir, "conversion_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Report: {report_path}")
    print(f"Train rows: {len(train_ds)}")
    print(
        "Test rows: "
        f"aime24={len(aime24_ds)}, "
        f"aime25={len(aime25_ds)}, "
        f"hmmt2025={len(hmmt_ds)}, "
        f"amo_number={len(amo_ds)}, "
        f"merged={len(merged_test_ds)}"
    )
    if validation["all_valid"]:
        print("Validation: PASS")
    else:
        print("Validation: FAIL (see conversion_report.json)")


if __name__ == "__main__":
    main()
