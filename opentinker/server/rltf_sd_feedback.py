# Copyright 2026 OpenTinker
#
# Licensed under the Apache License, Version 2.0.
"""Feedback generator for RLTF-SD agent loops.

This module implements environment-specific critique and draft scoring (r0)
without changing environment HTTP APIs.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter


@dataclass
class FeedbackResult:
    critique: str
    r0: float
    metadata: dict[str, Any]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_math_score(raw_score: float) -> float:
    """Normalize reviewer score to [0, 1].

    Accepts either [0,1] or [0,10] conventions from reviewer models.
    """
    score = float(raw_score)
    if 1.0 < score <= 10.0:
        score = score / 10.0
    return _clamp01(score)


class RLTFSdFeedbackGenerator:
    """Environment-aware critique + draft scoring for RLTF-SD."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self._session = requests.Session()
        self._logger = logging.getLogger(__name__)
        self._math_locks = threading.Lock()
        self._math_semaphores: dict[str, threading.BoundedSemaphore] = {}
        self._math_model_cache: dict[str, Optional[str]] = {}

        math_cfg = self.config.get("math", {}) or {}
        pool_size = int(math_cfg.get("pool_maxsize", 64))
        if pool_size > 0:
            adapter = HTTPAdapter(
                pool_connections=pool_size,
                pool_maxsize=pool_size,
                max_retries=0,
                pool_block=True,
            )
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

    def generate_feedback(
        self,
        *,
        interaction_name: str,
        messages: list[dict[str, Any]],
        draft_response: str,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ) -> FeedbackResult:
        name = str(interaction_name or "").lower()
        interaction_kwargs = interaction_kwargs or {}

        if name == "math":
            return self._math_feedback(
                messages=messages,
                draft_response=draft_response,
                interaction_kwargs=interaction_kwargs,
            )
        if name == "gomoku":
            return self._gomoku_feedback(
                messages=messages,
                draft_response=draft_response,
                interaction_kwargs=interaction_kwargs,
            )
        if name == "alfworld":
            return self._alfworld_feedback(
                messages=messages,
                draft_response=draft_response,
                interaction_kwargs=interaction_kwargs,
            )

        # Safe fallback for unknown environments.
        return FeedbackResult(
            critique="No environment-specific critique configured.",
            r0=0.5,
            metadata={"source": "fallback"},
        )

    def build_revision_prompt(self, critique: str, interaction_name: str) -> str:
        """Build a revision prompt injected before revised sampling."""
        name = str(interaction_name or "").lower()
        if name == "math":
            action_hint = "Provide a corrected final answer."
        elif name == "gomoku":
            action_hint = "Provide exactly one legal move using <move>row,col</move>."
        elif name == "alfworld":
            action_hint = "Provide exactly one executable action using <action>...</action>."
        else:
            action_hint = "Provide a revised response only."

        return (
            "Critique on your previous draft:\n"
            f"{critique}\n\n"
            "Revise your previous draft based on the critique.\n"
            f"{action_hint}"
        )

    # --------------------------- Math feedback ---------------------------

    def _math_feedback(
        self,
        *,
        messages: list[dict[str, Any]],
        draft_response: str,
        interaction_kwargs: dict[str, Any],
    ) -> FeedbackResult:
        math_cfg = self.config.get("math", {})
        endpoint = math_cfg.get("endpoint")
        model = math_cfg.get("model")
        timeout = float(math_cfg.get("timeout", 30.0))
        fail_fast = bool(math_cfg.get("fail_fast", True))
        max_concurrency = max(1, int(math_cfg.get("max_concurrency", 4)))
        acquire_timeout = float(math_cfg.get("acquire_timeout", timeout))
        max_tokens = int(math_cfg.get("max_tokens", 192))
        retries = max(0, int(math_cfg.get("retries", 1)))
        retry_backoff = max(0.0, float(math_cfg.get("retry_backoff", 1.0)))
        auto_pick_model = bool(math_cfg.get("auto_pick_model", True))
        enforce_json_response = bool(math_cfg.get("enforce_json_response", False))

        if not endpoint:
            msg = (
                "RLTF-SD math reviewer requires "
                "algorithm.rltf_sd.feedback.math.endpoint."
            )
            if fail_fast:
                raise RuntimeError(msg)
            return FeedbackResult(critique=msg, r0=0.0, metadata={"source": "math_reviewer"})

        question = self._extract_last_user_message(messages)
        env_kwargs = interaction_kwargs.get("env_kwargs", {}) or {}
        ground_truth = env_kwargs.get("ground_truth", "")

        system_prompt = math_cfg.get(
            "system_prompt",
            "You are a strict math reviewer. Return JSON only with keys: critique (string), score (0 to 1).",
        )
        user_prompt_template = math_cfg.get(
            "user_prompt_template",
            "Problem:\n{question}\n\nDraft solution:\n{draft}\n\nGround truth:\n{ground_truth}\n\n"
            "Evaluate correctness and reasoning quality. Return JSON.",
        )
        user_prompt = user_prompt_template.format(
            question=question,
            draft=draft_response,
            ground_truth=ground_truth,
        )

        api_key = math_cfg.get("api_key")
        api_key_env = math_cfg.get("api_key_env")
        if not api_key and api_key_env:
            import os

            api_key = os.getenv(str(api_key_env))

        url = endpoint.rstrip("/")
        if "/chat/completions" not in url:
            url = f"{url}/v1/chat/completions"

        payload = {
            "temperature": float(math_cfg.get("temperature", 0.0)),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if enforce_json_response:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resolved_model = self._resolve_math_model(
            endpoint=endpoint,
            headers=headers,
            timeout=min(timeout, 10.0),
            explicit_model=model,
            auto_pick_model=auto_pick_model,
        )
        # model is optional. Some reviewer endpoints route to a single default model.
        if resolved_model:
            payload["model"] = resolved_model

        semaphore = self._get_math_semaphore(url, max_concurrency=max_concurrency)
        acquired = semaphore.acquire(timeout=max(0.0, acquire_timeout))
        if not acquired:
            msg = (
                f"Math reviewer semaphore acquisition timed out after {acquire_timeout:.1f}s "
                f"(max_concurrency={max_concurrency}, endpoint={endpoint})"
            )
            if fail_fast:
                raise RuntimeError(msg)
            return FeedbackResult(critique=msg, r0=0.0, metadata={"source": "math_reviewer"})

        try:
            start_time = time.time()
            last_exc: Optional[Exception] = None
            for attempt in range(retries + 1):
                try:
                    response = self._session.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    raw = response.json()
                    content = (
                        raw.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    critique, score = self._parse_math_reviewer_response(content)
                    return FeedbackResult(
                        critique=critique,
                        r0=_normalize_math_score(score),
                        metadata={
                            "source": "math_reviewer",
                            "raw": content,
                            "model": resolved_model,
                            "latency_s": round(time.time() - start_time, 4),
                            "attempt": attempt + 1,
                        },
                    )
                except Exception as exc:  # noqa: PERF203
                    last_exc = exc
                    if attempt >= retries:
                        break
                    sleep_s = retry_backoff * (2**attempt)
                    self._logger.warning(
                        "Math reviewer request failed (attempt %s/%s): %s. Retrying in %.2fs",
                        attempt + 1,
                        retries + 1,
                        exc,
                        sleep_s,
                    )
                    time.sleep(sleep_s)

            assert last_exc is not None
            raise last_exc
        except Exception as exc:
            msg = f"Math reviewer call failed: {exc} (endpoint={url})"
            if not resolved_model:
                msg += (
                    " (If your reviewer endpoint requires a model name, set "
                    "algorithm.rltf_sd.feedback.math.model)"
                )
            if fail_fast:
                raise RuntimeError(msg) from exc
            return FeedbackResult(critique=msg, r0=0.0, metadata={"source": "math_reviewer"})
        finally:
            semaphore.release()

    def _parse_math_reviewer_response(self, content: str) -> tuple[str, float]:
        if not content:
            raise ValueError("empty reviewer response")

        # First try strict JSON.
        try:
            obj = json.loads(content)
            critique = str(obj.get("critique", "")).strip()
            score = float(obj.get("score", 0.0))
            if critique:
                return critique, score
        except Exception:
            pass

        # Then try fenced JSON or loose extraction.
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                obj = json.loads(json_match.group(0))
                critique = str(obj.get("critique", "")).strip()
                score = float(obj.get("score", 0.0))
                if critique:
                    return critique, score
            except Exception:
                pass

        score_match = re.search(r"(?:score|r0)\s*[:=]\s*(-?\d+(?:\.\d+)?)", content, flags=re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 0.0
        critique = content.strip()
        return critique, _normalize_math_score(score)

    # -------------------------- Gomoku feedback --------------------------

    def _gomoku_feedback(
        self,
        *,
        messages: list[dict[str, Any]],
        draft_response: str,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ) -> FeedbackResult:
        gomoku_cfg = self.config.get("gomoku", {}) or {}
        llm_feedback = self._optional_llm_feedback(
            task_name="gomoku",
            task_cfg=gomoku_cfg,
            messages=messages,
            draft_response=draft_response,
            system_prompt_default=(
                "You are a strict Gomoku move reviewer. Return JSON only with keys: "
                "critique (string), score (0 to 1)."
            ),
            user_prompt_template_default=(
                "Current observation:\n{observation}\n\n"
                "Draft move response:\n{draft}\n\n"
                "Evaluate legality, format correctness, and tactical quality. "
                "Return JSON only."
            ),
            fail_fast_default=False,
        )
        if llm_feedback is not None:
            return llm_feedback

        observation = self._extract_last_user_message(messages)
        board = self._parse_gomoku_board(observation)
        move = self._parse_gomoku_move(draft_response)

        if move is None:
            return FeedbackResult(
                critique="Missing or invalid move format. Use <move>row,col</move>.",
                r0=0.0,
                metadata={"source": "gomoku_rule"},
            )

        row, col = move
        if board is None:
            return FeedbackResult(
                critique="Could not parse board from observation; keep move format strict and legal.",
                r0=0.35,
                metadata={"source": "gomoku_rule"},
            )

        size = len(board)
        if row < 0 or col < 0 or row >= size or col >= size:
            return FeedbackResult(
                critique=f"Move ({row},{col}) is out of bounds for board size {size}.",
                r0=0.0,
                metadata={"source": "gomoku_rule", "size": size},
            )

        cell = board[row][col]
        if cell != ".":
            return FeedbackResult(
                critique=f"Move ({row},{col}) is occupied by '{cell}'. Choose an empty position.",
                r0=0.05,
                metadata={"source": "gomoku_rule", "size": size},
            )

        # Lightweight tactical heuristic.
        center = size // 2
        dist = abs(row - center) + abs(col - center)
        near_center_bonus = 0.08 if dist <= max(1, size // 3) else 0.0

        x_neighbors = 0
        o_neighbors = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = row + dr
                cc = col + dc
                if 0 <= rr < size and 0 <= cc < size:
                    if board[rr][cc] == "X":
                        x_neighbors += 1
                    elif board[rr][cc] == "O":
                        o_neighbors += 1

        connectivity_bonus = 0.12 if x_neighbors >= 1 else 0.0
        blocking_bonus = 0.1 if o_neighbors >= 2 else 0.0
        r0 = _clamp01(0.45 + near_center_bonus + connectivity_bonus + blocking_bonus)

        critique = (
            f"Move ({row},{col}) is legal. "
            f"Prefer moves that extend your X chains or block immediate O threats. "
            f"Neighbor stats: X={x_neighbors}, O={o_neighbors}."
        )
        return FeedbackResult(
            critique=critique,
            r0=r0,
            metadata={
                "source": "gomoku_rule",
                "x_neighbors": x_neighbors,
                "o_neighbors": o_neighbors,
            },
        )

    def _parse_gomoku_board(self, observation: str) -> Optional[list[list[str]]]:
        lines = observation.splitlines()
        board: list[list[str]] = []
        for line in lines:
            m = re.match(r"^\s*\d+\s+([XO\.\s]+)$", line.strip())
            if not m:
                continue
            cells = [tok for tok in m.group(1).split() if tok in {"X", "O", "."}]
            if cells:
                board.append(cells)

        if not board:
            return None

        width = len(board[0])
        for row in board:
            if len(row) != width:
                return None
        return board

    def _parse_gomoku_move(self, draft_response: str) -> Optional[tuple[int, int]]:
        match = re.search(r"<move>\s*(\d+)\s*,\s*(\d+)\s*</move>", draft_response, flags=re.IGNORECASE)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    # -------------------------- ALFWorld feedback ------------------------

    def _alfworld_feedback(
        self,
        *,
        messages: list[dict[str, Any]],
        draft_response: str,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ) -> FeedbackResult:
        alf_cfg = self.config.get("alfworld", {}) or {}
        llm_feedback = self._optional_llm_feedback(
            task_name="alfworld",
            task_cfg=alf_cfg,
            messages=messages,
            draft_response=draft_response,
            system_prompt_default=(
                "You are a strict ALFWorld action reviewer. Return JSON only with keys: "
                "critique (string), score (0 to 1)."
            ),
            user_prompt_template_default=(
                "Current observation:\n{observation}\n\n"
                "Draft action response:\n{draft}\n\n"
                "Evaluate executability under current Available Actions, correctness, and relevance. "
                "Return JSON only."
            ),
            fail_fast_default=False,
        )
        if llm_feedback is not None:
            return llm_feedback

        observation = self._extract_last_user_message(messages)
        action = self._parse_alfworld_action(draft_response)
        available_actions = self._parse_alfworld_available_actions(observation)

        if not action:
            return FeedbackResult(
                critique="Missing action. Use <action>...</action> with exactly one executable command.",
                r0=0.0,
                metadata={"source": "alfworld_rule"},
            )

        if not available_actions:
            return FeedbackResult(
                critique="No action list found. Keep the action concrete and executable.",
                r0=0.4,
                metadata={"source": "alfworld_rule"},
            )

        norm_action = action.strip().lower()
        norm_choices = [a.strip().lower() for a in available_actions]

        if norm_action in norm_choices:
            return FeedbackResult(
                critique="Action is executable under current Available Actions.",
                r0=0.9,
                metadata={"source": "alfworld_rule", "action": action},
            )

        close = get_close_matches(norm_action, norm_choices, n=1, cutoff=0.75)
        if close:
            return FeedbackResult(
                critique=(
                    "Action is close to a valid command but not exact. "
                    f"Use the exact action string: {close[0]}"
                ),
                r0=0.45,
                metadata={"source": "alfworld_rule", "action": action},
            )

        return FeedbackResult(
            critique="Action is not in Available Actions. Pick an executable action from the list.",
            r0=0.1,
            metadata={"source": "alfworld_rule", "action": action},
        )

    def _parse_alfworld_action(self, draft_response: str) -> str:
        match = re.search(r"<action>\s*(.*?)\s*</action>", draft_response, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

        lines = [ln.strip() for ln in draft_response.splitlines() if ln.strip()]
        return lines[-1] if lines else ""

    def _parse_alfworld_available_actions(self, observation: str) -> list[str]:
        actions: list[str] = []
        in_actions = False
        for line in observation.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("=== available actions"):
                in_actions = True
                continue
            if not in_actions:
                continue
            if not stripped:
                continue
            if stripped.startswith("==="):
                break
            if stripped.startswith("-"):
                candidate = stripped[1:].strip()
                if candidate:
                    actions.append(candidate)

        return actions

    # ----------------------------- Utilities -----------------------------

    def _optional_llm_feedback(
        self,
        *,
        task_name: str,
        task_cfg: dict[str, Any],
        messages: list[dict[str, Any]],
        draft_response: str,
        system_prompt_default: str,
        user_prompt_template_default: str,
        fail_fast_default: bool,
    ) -> Optional[FeedbackResult]:
        endpoint = task_cfg.get("endpoint")
        if not endpoint:
            return None

        model = task_cfg.get("model")
        timeout = float(task_cfg.get("timeout", 30.0))
        fail_fast = bool(task_cfg.get("fail_fast", fail_fast_default))
        max_concurrency = max(1, int(task_cfg.get("max_concurrency", 4)))
        acquire_timeout = float(task_cfg.get("acquire_timeout", timeout))
        max_tokens = int(task_cfg.get("max_tokens", 192))
        retries = max(0, int(task_cfg.get("retries", 1)))
        retry_backoff = max(0.0, float(task_cfg.get("retry_backoff", 1.0)))
        auto_pick_model = bool(task_cfg.get("auto_pick_model", True))
        enforce_json_response = bool(task_cfg.get("enforce_json_response", False))

        observation = self._extract_last_user_message(messages)
        system_prompt = task_cfg.get("system_prompt", system_prompt_default)
        user_prompt_template = task_cfg.get(
            "user_prompt_template", user_prompt_template_default
        )
        user_prompt = user_prompt_template.format(
            observation=observation,
            draft=draft_response,
        )

        api_key = task_cfg.get("api_key")
        api_key_env = task_cfg.get("api_key_env")
        if not api_key and api_key_env:
            import os

            api_key = os.getenv(str(api_key_env))

        url = str(endpoint).rstrip("/")
        if "/chat/completions" not in url:
            url = f"{url}/v1/chat/completions"

        payload: dict[str, Any] = {
            "temperature": float(task_cfg.get("temperature", 0.0)),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if enforce_json_response:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resolved_model = self._resolve_math_model(
            endpoint=endpoint,
            headers=headers,
            timeout=min(timeout, 10.0),
            explicit_model=model,
            auto_pick_model=auto_pick_model,
        )
        if resolved_model:
            payload["model"] = resolved_model

        semaphore = self._get_math_semaphore(url, max_concurrency=max_concurrency)
        acquired = semaphore.acquire(timeout=max(0.0, acquire_timeout))
        if not acquired:
            msg = (
                f"{task_name} reviewer semaphore acquisition timed out after "
                f"{acquire_timeout:.1f}s (max_concurrency={max_concurrency}, endpoint={endpoint})"
            )
            if fail_fast:
                raise RuntimeError(msg)
            self._logger.warning(msg)
            return None

        try:
            start_time = time.time()
            last_exc: Optional[Exception] = None
            for attempt in range(retries + 1):
                try:
                    response = self._session.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    raw = response.json()
                    content = (
                        raw.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    critique, score = self._parse_structured_feedback_response(content)
                    return FeedbackResult(
                        critique=critique,
                        r0=score,
                        metadata={
                            "source": f"{task_name}_reviewer",
                            "raw": content,
                            "model": resolved_model,
                            "latency_s": round(time.time() - start_time, 4),
                            "attempt": attempt + 1,
                            "endpoint": endpoint,
                        },
                    )
                except Exception as exc:
                    last_exc = exc
                    if attempt >= retries:
                        break
                    sleep_s = retry_backoff * (2**attempt)
                    self._logger.warning(
                        "%s reviewer request failed (attempt %s/%s): %s. Retrying in %.2fs",
                        task_name,
                        attempt + 1,
                        retries + 1,
                        exc,
                        sleep_s,
                    )
                    time.sleep(sleep_s)

            assert last_exc is not None
            raise last_exc
        except Exception as exc:
            msg = f"{task_name} reviewer call failed: {exc} (endpoint={url})"
            if not resolved_model:
                msg += (
                    " (If your reviewer endpoint requires a model name, set "
                    f"algorithm.rltf_sd.feedback.{task_name}.model)"
                )
            if fail_fast:
                raise RuntimeError(msg) from exc
            self._logger.warning(
                "%s Falling back to built-in heuristic feedback.", msg
            )
            return None
        finally:
            semaphore.release()

    def _parse_structured_feedback_response(self, content: str) -> tuple[str, float]:
        if not content:
            raise ValueError("empty reviewer response")

        # strict JSON
        try:
            obj = json.loads(content)
            critique = str(obj.get("critique", "")).strip()
            score = float(obj.get("score", 0.0))
            if critique:
                return critique, _normalize_math_score(score)
        except Exception:
            pass

        # fenced JSON / loose JSON block
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                obj = json.loads(json_match.group(0))
                critique = str(obj.get("critique", "")).strip()
                score = float(obj.get("score", 0.0))
                if critique:
                    return critique, _normalize_math_score(score)
            except Exception:
                pass

        score_match = re.search(
            r"(?:score|r0)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
            content,
            flags=re.IGNORECASE,
        )
        score = float(score_match.group(1)) if score_match else 0.0
        return content.strip(), _normalize_math_score(score)

    def _get_math_semaphore(
        self,
        endpoint: str,
        *,
        max_concurrency: int,
    ) -> threading.BoundedSemaphore:
        key = str(endpoint or "").rstrip("/")
        with self._math_locks:
            sem = self._math_semaphores.get(key)
            if sem is None:
                sem = threading.BoundedSemaphore(value=max_concurrency)
                self._math_semaphores[key] = sem
            return sem

    def _resolve_math_model(
        self,
        *,
        endpoint: str,
        headers: dict[str, str],
        timeout: float,
        explicit_model: Optional[str],
        auto_pick_model: bool,
    ) -> Optional[str]:
        if explicit_model:
            return str(explicit_model)
        if not auto_pick_model:
            return None

        key = str(endpoint or "").rstrip("/")
        with self._math_locks:
            if key in self._math_model_cache:
                return self._math_model_cache[key]

        models_url = key
        if "/v1/models" not in models_url:
            models_url = f"{models_url}/v1/models"
        try:
            response = self._session.get(models_url, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            items = data.get("data", []) if isinstance(data, dict) else []
            model_id = None
            if isinstance(items, list) and items:
                first = items[0]
                if isinstance(first, dict):
                    model_id = first.get("id")
            resolved = str(model_id) if model_id else None
        except Exception:
            resolved = None

        with self._math_locks:
            self._math_model_cache[key] = resolved
        return resolved

    def _extract_last_user_message(self, messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""
