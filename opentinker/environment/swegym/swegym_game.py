#!/usr/bin/env python3
"""SWE-Gym Game Implementation.

This module implements AbstractGame for SWE-Gym style patch-and-test tasks.
The agent outputs a unified diff patch as its action. The environment applies
the patch, runs FAIL_TO_PASS tests, and returns rewards based on pass ratio.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from opentinker.environment.base_game import AbstractGame, StepResult

logger = logging.getLogger(__name__)


class SWEGymGame(AbstractGame):
    """SWE-Gym task environment using repo patch + tests workflow."""

    REWARD_SUCCESS = 1.0
    REWARD_APPLY_FAILED = -0.2
    REWARD_TEST_FAILED = -0.1

    _dataset_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    _dataset_index_cache: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    _dataset_lock = threading.Lock()

    def __init__(
        self,
        dataset_name: str = "SWE-Gym/SWE-Gym",
        split: str = "train",
        repo_cache_dir: str = "/tmp/swegym/repos",
        repo_url_template: str = "https://github.com/{repo}.git",
        apply_test_patch: bool = True,
        run_pass_to_pass: bool = False,
        timeout_s: int = 600,
        test_command: str = "pytest",
        max_steps: int = 6,
        max_prompt_tokens: Optional[int] = None,
        tokenizer_path: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.repo_cache_dir = Path(repo_cache_dir)
        self.repo_url_template = repo_url_template
        self.apply_test_patch = apply_test_patch
        self.run_pass_to_pass = run_pass_to_pass
        self.timeout_s = timeout_s
        self.test_command = test_command
        self.max_steps = max_steps
        self.max_prompt_tokens = max_prompt_tokens
        self.tokenizer_path = tokenizer_path
        self._tokenizer = None
        self._warned_missing_tokenizer = False
        self.seed = seed

        if seed is not None:
            random.seed(seed)

        self._ensure_dataset_loaded()
        self._prompt_sample: Optional[Dict[str, Any]] = None
        self._active_sample: Optional[Dict[str, Any]] = None
        self._repo_dir: Optional[Path] = None
        self._step_count = 0

    def get_system_prompt(self) -> str:
        return (
            "You are a software engineering agent. "
            "Given a repository bug report, produce a unified diff patch that fixes the issue. "
            "Return ONLY the patch text in unified diff format (starting with 'diff --git')."
        )

    def get_initial_user_message(self) -> str:
        return "Here is the task. Generate a patch to fix the failing tests."

    def generate_initial_state(self) -> Dict[str, Any]:
        sample = self._get_random_sample()
        self._prompt_sample = sample
        return {"sample_id": sample.get("instance_id")}

    def render_state_for_prompt(self, **state) -> str:
        sample_id = state.get("sample_id")
        sample = self._get_sample_by_id(sample_id) if sample_id else None
        if sample is None:
            sample = self._get_random_sample()
        return self._build_problem_prompt(sample)

    def reset(self, **kwargs) -> str:
        self._step_count = 0
        sample_id = kwargs.get("sample_id")
        sample = self._get_sample_by_id(sample_id) if sample_id else None
        if sample is None:
            sample = self._get_random_sample()

        self._active_sample = sample
        self._repo_dir = self._prepare_repo(sample)

        test_patch = sample.get("test_patch") or ""
        if self.apply_test_patch and test_patch.strip():
            ok, err = self._apply_patch(test_patch)
            if not ok:
                logger.warning("Failed to apply test_patch: %s", err)

        return self._build_problem_prompt(sample)

    def step(self, action: str) -> StepResult:
        if not self._active_sample or not self._repo_dir:
            return StepResult(
                observation="Environment not initialized. Call reset first.",
                reward=self.REWARD_TEST_FAILED,
                done=True,
                info={"error": "not_initialized"},
            )

        self._step_count += 1

        patch = self._extract_patch(action)
        ok, err = self._apply_patch(patch)
        if not ok:
            return StepResult(
                observation=f"Patch apply failed:\n{err}",
                reward=self.REWARD_APPLY_FAILED,
                done=False,
                info={"apply_error": err},
            )

        fail_to_pass = self._normalize_test_list(self._active_sample.get("FAIL_TO_PASS"))
        pass_to_pass = self._normalize_test_list(self._active_sample.get("PASS_TO_PASS"))

        fail_results = self._run_tests(fail_to_pass)
        all_fail_passed = fail_results["passed"] == fail_results["total"]

        pass_results = {"passed": 0, "total": 0, "outputs": []}
        if all_fail_passed and self.run_pass_to_pass and pass_to_pass:
            pass_results = self._run_tests(pass_to_pass)

        done = all_fail_passed and (
            not self.run_pass_to_pass or pass_results["passed"] == pass_results["total"]
        )

        if done:
            reward = self.REWARD_SUCCESS
        else:
            if fail_results["total"] > 0:
                reward = fail_results["passed"] / float(fail_results["total"])
            else:
                reward = self.REWARD_TEST_FAILED

        observation = self._format_test_observation(fail_results, pass_results)
        info = {
            "fail_to_pass": fail_results,
            "pass_to_pass": pass_results,
            "step_count": self._step_count,
        }

        return StepResult(observation=observation, reward=reward, done=done, info=info)

    def _ensure_dataset_loaded(self) -> None:
        cache_key = (self.dataset_name, self.split)
        with self._dataset_lock:
            if cache_key in self._dataset_cache:
                return
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError(
                    "datasets is required for SWE-Gym. Install with: pip install datasets"
                ) from exc

            dataset = load_dataset(self.dataset_name, split=self.split)
            samples = list(dataset)
            self._dataset_cache[cache_key] = samples
            self._dataset_index_cache[cache_key] = {
                str(s.get("instance_id")): s for s in samples if s.get("instance_id")
            }

    def _get_random_sample(self) -> Dict[str, Any]:
        samples = self._dataset_cache[(self.dataset_name, self.split)]
        return random.choice(samples)

    def _get_sample_by_id(self, sample_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not sample_id:
            return None
        index = self._dataset_index_cache.get((self.dataset_name, self.split), {})
        return index.get(str(sample_id))

    def _build_problem_prompt(self, sample: Dict[str, Any]) -> str:
        parts = []
        parts.append(f"Repository: {sample.get('repo', '')}")
        if sample.get("version") is not None:
            parts.append(f"Version: {sample.get('version')}")
        if sample.get("base_commit"):
            parts.append(f"Base commit: {sample.get('base_commit')}")

        problem_statement = sample.get("problem_statement") or ""
        problem_statement = self._truncate_text(problem_statement)
        parts.append("\nProblem Statement:\n" + problem_statement)

        hints = sample.get("hints_text") or ""
        if hints.strip():
            hints = self._truncate_text(hints)
            parts.append("\nHints:\n" + hints)

        fail_to_pass = self._normalize_test_list(sample.get("FAIL_TO_PASS"))
        pass_to_pass = self._normalize_test_list(sample.get("PASS_TO_PASS"))

        if fail_to_pass:
            parts.append("\nFailing Tests (FAIL_TO_PASS):\n" + "\n".join(fail_to_pass))
        if pass_to_pass:
            parts.append("\nRegression Tests (PASS_TO_PASS):\n" + "\n".join(pass_to_pass))

        parts.append(
            "\nYour task: provide a unified diff patch that fixes the failing tests."
        )
        return "\n".join(parts)

    def _prepare_repo(self, sample: Dict[str, Any]) -> Path:
        repo = sample.get("repo")
        if not repo:
            raise ValueError("Sample missing repo field")

        base_commit = sample.get("base_commit")
        if not base_commit:
            raise ValueError("Sample missing base_commit field")

        cache_dir = self.repo_cache_dir / "cache"
        work_dir = self.repo_cache_dir / "instances"
        cache_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        repo_key = repo.replace("/", "__")
        cached_repo = cache_dir / repo_key
        if not cached_repo.exists():
            url = self.repo_url_template.format(repo=repo)
            self._run_cmd(["git", "clone", "--no-checkout", url, str(cached_repo)])
        else:
            self._run_cmd(["git", "fetch", "--all"], cwd=cached_repo)

        instance_dir = work_dir / f"{repo_key}-{self._safe_id(sample.get('instance_id'))}"
        if instance_dir.exists():
            shutil.rmtree(instance_dir, ignore_errors=True)

        self._run_cmd(["git", "clone", "--shared", str(cached_repo), str(instance_dir)])
        self._run_cmd(["git", "checkout", base_commit], cwd=instance_dir)
        self._run_cmd(["git", "reset", "--hard"], cwd=instance_dir)

        return instance_dir

    def _safe_id(self, value: Optional[str]) -> str:
        if not value:
            return "unknown"
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", value)

    def _apply_patch(self, patch: str) -> Tuple[bool, str]:
        if not patch or not patch.strip():
            return False, "Empty patch"
        if not self._repo_dir:
            return False, "Repository not initialized"
        result = self._run_cmd(
            ["git", "apply", "--whitespace=fix", "-"],
            cwd=self._repo_dir,
            input_text=patch,
            check=False,
        )
        if result.returncode != 0:
            return False, result.stderr.strip() or "git apply failed"
        return True, ""

    def _run_tests(self, tests: List[str]) -> Dict[str, Any]:
        outputs = []
        passed = 0
        total = len(tests)

        if total == 0:
            return {"passed": 0, "total": 0, "outputs": []}

        for test in tests:
            cmd = self._build_test_command(test)
            result = self._run_cmd(
                cmd,
                cwd=self._repo_dir,
                check=False,
                timeout=self.timeout_s,
            )
            outputs.append(self._summarize_output(result))
            if result.returncode == 0:
                passed += 1

        return {"passed": passed, "total": total, "outputs": outputs}

    def _build_test_command(self, test_entry: str) -> List[str]:
        if test_entry.startswith("pytest") or test_entry.startswith("python -m pytest"):
            return test_entry.split()
        return [self.test_command, test_entry]

    def _normalize_test_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    data = json.loads(text)
                    if isinstance(data, list):
                        return [str(v) for v in data if str(v).strip()]
                except json.JSONDecodeError:
                    pass
            return [line.strip() for line in text.splitlines() if line.strip()]
        return [str(value)]

    def _extract_patch(self, action: str) -> str:
        if not action:
            return ""
        lines = action.splitlines()
        for idx, line in enumerate(lines):
            if line.startswith("diff --git "):
                return "\n".join(lines[idx:]) + "\n"
        return action.strip() + "\n"

    def _format_test_observation(
        self, fail_results: Dict[str, Any], pass_results: Dict[str, Any]
    ) -> str:
        parts = [
            f"FAIL_TO_PASS: {fail_results['passed']}/{fail_results['total']} passed."
        ]
        if self.run_pass_to_pass:
            parts.append(
                f"PASS_TO_PASS: {pass_results['passed']}/{pass_results['total']} passed."
            )
        for output in fail_results.get("outputs", [])[:3]:
            parts.append("\n" + output)
        return "\n".join(parts)

    def _summarize_output(self, result: subprocess.CompletedProcess) -> str:
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        combined = "\n".join(part for part in [stdout, stderr] if part)
        if not combined:
            return f"[exit={result.returncode}] (no output)"
        return f"[exit={result.returncode}]\n{combined[:2000]}"

    def _run_cmd(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        input_text: Optional[str] = None,
        check: bool = True,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            input=input_text,
            text=True,
            capture_output=True,
            check=check,
            timeout=timeout,
        )

    def _get_tokenizer(self):
        if self._tokenizer is not None or not self.tokenizer_path:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for token-based truncation. "
                "Install with: pip install transformers"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return self._tokenizer

    def _truncate_text(self, text: str) -> str:
        if not text:
            return text
        if not self.max_prompt_tokens:
            return text
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            if not self._warned_missing_tokenizer:
                logger.warning(
                    "max_prompt_tokens is set but tokenizer_path is missing; "
                    "prompt truncation is disabled."
                )
                self._warned_missing_tokenizer = True
            return self._truncate_bytes(text)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.max_prompt_tokens:
            return text
        truncated = tokenizer.decode(tokens[: self.max_prompt_tokens])
        return truncated + "\n...[truncated]"

    def _truncate_bytes(self, text: str) -> str:
        """Fallback truncation without tokenizer (byte-level, conservative)."""
        if not self.max_prompt_tokens:
            return text
        data = text.encode("utf-8")
        if len(data) <= self.max_prompt_tokens:
            return text
        truncated = data[: self.max_prompt_tokens].decode("utf-8", errors="ignore")
        return truncated + "\n...[truncated]"

