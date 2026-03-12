#!/usr/bin/env python3
"""ScienceWorld game implementation for OpenTinker.

This wrapper follows the same AbstractGame contract as ALFWorld and exposes
ScienceWorld through the generic FastAPI game server + GameEnvironment stack.
"""

from __future__ import annotations

import heapq
import logging
import random
import re
import threading
from typing import Any, Dict, Iterable, List, Optional

from opentinker.environment.base_game import AbstractGame, StepResult

try:
    from scienceworld import ScienceWorldEnv

    SCIWORLD_AVAILABLE = True
except ImportError:
    ScienceWorldEnv = None
    SCIWORLD_AVAILABLE = False

logger = logging.getLogger(__name__)


class SciWorldGame(AbstractGame):
    """ScienceWorld text environment wrapper."""

    REWARD_SUCCESS = 10.0
    REWARD_FAILURE = -1.0
    REWARD_STEP = -0.01
    REWARD_INVALID_ACTION = -0.1

    DEFAULT_MAX_STEPS = 30
    DEFAULT_MAX_ACTIONS = 50
    DEFAULT_LOCAL_THREAD_BASE = 20000

    _thread_lock = threading.Lock()
    _free_thread_offsets: list[int] = []
    _next_thread_offset = 0

    def __init__(
        self,
        max_steps: int = DEFAULT_MAX_STEPS,
        split: str = "train",
        task_names: Optional[List[str]] = None,
        task_ids: Optional[List[int]] = None,
        variation_indices: Optional[List[int]] = None,
        simplification_str: str = "",
        jar_path: Optional[str] = None,
        thread_base: int = 0,
        max_action_candidates: int = DEFAULT_MAX_ACTIONS,
    ):
        if not SCIWORLD_AVAILABLE:
            raise ImportError(
                "scienceworld package not installed. Install with: pip install scienceworld"
            )

        self.max_steps = max_steps
        self.split = (split or "train").strip().lower()
        self.task_names = self._normalize_string_list(task_names)
        self.task_ids = list(task_ids or [])
        self.variation_indices = [int(v) for v in (variation_indices or [])]
        self.simplification_str = simplification_str or ""
        self.jar_path = jar_path
        self.thread_base = int(thread_base or 0)
        self.max_action_candidates = max(1, int(max_action_candidates or 1))

        self._env: Optional[ScienceWorldEnv] = None
        self._thread_num: Optional[int] = None
        self._all_task_names: Optional[List[str]] = None
        self._variation_cache: Dict[str, List[int]] = {}

        self._current_obs = ""
        self._task_desc = ""
        self._current_task_name = ""
        self._current_variation = -1
        self._current_info: Dict[str, Any] = {}
        self._admissible_actions: List[str] = []
        self._action_templates: List[str] = []
        self._objects: List[str] = []
        self._done = False
        self._step_count = 0
        self._score = 0.0

    @staticmethod
    def _normalize_string_list(values: Optional[Iterable[str]]) -> List[str]:
        if values is None:
            return []
        if isinstance(values, str):
            values = [v.strip() for v in values.split(",")]
        return [str(v).strip() for v in values if str(v).strip()]

    @classmethod
    def _allocate_thread_offset(cls) -> int:
        with cls._thread_lock:
            if cls._free_thread_offsets:
                return heapq.heappop(cls._free_thread_offsets)
            offset = cls._next_thread_offset
            cls._next_thread_offset += 1
            return offset

    @classmethod
    def _release_thread_offset(cls, offset: int) -> None:
        with cls._thread_lock:
            heapq.heappush(cls._free_thread_offsets, offset)

    def _ensure_env(self) -> None:
        if self._env is not None:
            return

        offset = self._allocate_thread_offset()
        self._thread_num = self.thread_base + offset
        logger.info(
            "Initializing ScienceWorldEnv(threadNum=%s, max_steps=%s, split=%s)",
            self._thread_num,
            self.max_steps,
            self.split,
        )
        kwargs = {"envStepLimit": self.max_steps}
        if self.jar_path:
            kwargs["serverPath"] = self.jar_path
        self._env = ScienceWorldEnv("", **kwargs)

    def close(self) -> None:
        if self._env is not None:
            shutdown_fn = getattr(self._env, "shutdown", None)
            if callable(shutdown_fn):
                try:
                    shutdown_fn()
                except Exception as exc:
                    logger.warning("Failed to shutdown ScienceWorldEnv: %s", exc)
            self._env = None

        if self._thread_num is not None:
            offset = self._thread_num - self.thread_base
            if offset >= 0:
                self._release_thread_offset(offset)
            self._thread_num = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_all_task_names(self) -> List[str]:
        self._ensure_env()
        if self._all_task_names is None:
            task_names = list(self._env.getTaskNames())
            self._all_task_names = [str(name) for name in task_names]
        return self._all_task_names

    def _resolve_task_pool(self) -> List[str]:
        task_pool = self._get_all_task_names()
        if self.task_names:
            allowed = set(self.task_names)
            filtered = [name for name in task_pool if name in allowed]
            if not filtered:
                raise ValueError(
                    f"No ScienceWorld tasks matched task_names={self.task_names!r}"
                )
            return filtered

        if self.task_ids:
            resolved = []
            for task_id in self.task_ids:
                if task_id < 0 or task_id >= len(task_pool):
                    raise ValueError(
                        f"ScienceWorld task_id out of range: {task_id} (num_tasks={len(task_pool)})"
                    )
                resolved.append(task_pool[task_id])
            return resolved

        return task_pool

    @staticmethod
    def _extract_variations_from_value(
        value: Any, task_name: str, split: str
    ) -> Optional[List[int]]:
        if value is None:
            return None
        if isinstance(value, dict):
            for key in (task_name, split):
                if key in value:
                    extracted = SciWorldGame._extract_variations_from_value(
                        value[key], task_name, split
                    )
                    if extracted:
                        return extracted
            return None
        if isinstance(value, range):
            return list(value)
        if isinstance(value, (list, tuple, set)):
            result = []
            for item in value:
                try:
                    result.append(int(item))
                except Exception:
                    return None
            return result
        if isinstance(value, int):
            return list(range(value))
        return None

    def _call_variation_method(
        self, method_name: str, task_name: str
    ) -> Optional[List[int]]:
        method = getattr(self._env, method_name, None)
        if not callable(method):
            return None

        for args in ((task_name,), tuple()):
            try:
                value = method(*args)
            except TypeError:
                continue
            except Exception:
                return None

            variations = self._extract_variations_from_value(
                value=value,
                task_name=task_name,
                split=self.split,
            )
            if variations:
                return variations
        return None

    def _resolve_variations_for_task(self, task_name: str) -> List[int]:
        if task_name in self._variation_cache:
            return self._variation_cache[task_name]

        # 1. Get the ground-truth valid variations from the environment for this task/split
        split_method_candidates = {
            "train": [
                "getVariationsTrain",
                "getTrainVariations",
                "getTaskTrainVariations",
            ],
            "dev": [
                "getVariationsDev",
                "getDevVariations",
                "getTaskDevVariations",
            ],
            "test": [
                "getVariationsTest",
                "getTestVariations",
                "getTaskTestVariations",
            ],
        }
        generic_candidates = [
            "getVariationsForTask",
            "getTaskVariations",
            "getVariations",
            "getVariationIndices",
            "getNumVariations",
        ]

        env_variations = []
        for method_name in split_method_candidates.get(self.split, []):
            env_variations = self._call_variation_method(method_name, task_name) or []
            if env_variations:
                break

        if not env_variations:
            for method_name in generic_candidates:
                env_variations = (
                    self._call_variation_method(method_name, task_name) or []
                )
                if env_variations:
                    break

        if not env_variations:
            env_variations = [0]

        # 2. If user provided a specific set of indices, take the intersection
        if self.variation_indices:
            user_indices = set(self.variation_indices)
            env_indices = set(env_variations)
            variations = sorted(list(user_indices.intersection(env_indices)))

            # If intersection is empty, fall back to environment defaults to prevent crash
            if not variations:
                logger.warning(
                    f"Task {task_name!r} has no overlap with provided variation_indices. "
                    f"Falling back to environment's variations."
                )
                variations = sorted(env_variations)
        else:
            variations = sorted(env_variations)

        self._variation_cache[task_name] = variations
        return variations

    def _select_task_and_variation(
        self,
        task_name: Optional[str] = None,
        variation: Optional[int] = None,
    ) -> tuple[str, int]:
        task_pool = self._resolve_task_pool()
        selected_task = task_name or random.choice(task_pool)
        if selected_task not in task_pool:
            raise ValueError(
                f"Task {selected_task!r} is not in the configured ScienceWorld task pool"
            )

        variations = self._resolve_variations_for_task(selected_task)

        # 1. Determine initial selection
        if variation is not None:
            v_int = int(variation)
            if v_int in variations:
                selected_variation = v_int
            else:
                # Fallback to a valid one from the list via modulo
                selected_variation = variations[v_int % len(variations)]
        else:
            selected_variation = random.choice(variations)

        # 2. MANDATORY HARD LIMIT CHECK
        # ScienceWorld often has a task-specific maximum variations.
        # We try several ways to find the absolute limit to prevent the Java-side error.
        limit = 1000000  # Default huge

        # Try to get the count from various scienceworld metadata sources
        for method_name in ["getMaxVariations", "getNumVariations", "getVariationCount"]:
            try:
                m = getattr(self._env, method_name, None)
                if callable(m):
                    # Try with task name, then without
                    for args in ((selected_task,), ()):
                        try:
                            res = m(*args)
                            if isinstance(res, int) and res > 0:
                                limit = min(limit, res)
                                break
                        except Exception:
                            continue
            except Exception:
                pass

        # If we have a list of variations from _resolve, its max is also a limit
        if variations:
            limit = min(limit, max(variations) + 1)

        # If our selection still exceeds the limit, force it down
        if selected_variation >= limit:
            old_v = selected_variation
            selected_variation = old_v % limit
            logger.warning(
                f"Forced fix: variation {old_v} exceeds limit {limit} for task {selected_task!r}. "
                f"New variation: {selected_variation}"
            )

        return selected_task, selected_variation

    def reset(
        self,
        task_name: Optional[str] = None,
        variation: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> str:
        del kwargs
        self._ensure_env()

        if seed is not None:
            random.seed(seed)

        selected_task, selected_variation = self._select_task_and_variation(
            task_name=task_name,
            variation=variation,
        )

        self._env.load(selected_task, selected_variation, self.simplification_str)
        observation, info = self._env.reset()

        self._current_task_name = selected_task
        self._current_variation = selected_variation
        self._current_obs = str(observation)
        self._current_info = info if isinstance(info, dict) else {}
        self._task_desc = self._get_task_description()
        self._admissible_actions = self._extract_valid_actions_from_info(self._current_info)
        self._action_templates = self._extract_action_templates()
        self._objects = self._extract_objects()
        self._done = False
        self._step_count = 0
        self._score = self._extract_score(self._current_info)

        return self._format_observation(self._current_obs)

    def step(self, action: str) -> StepResult:
        if self._done:
            return StepResult(
                observation="Episode already finished.",
                reward=0.0,
                done=True,
                info={"error": "episode_finished"},
            )

        self._ensure_env()
        self._step_count += 1
        parsed_action = self._parse_action(action)

        observation, reward, done, info = self._env.step(parsed_action)
        info = info if isinstance(info, dict) else {}

        self._current_obs = str(observation)
        self._current_info = info
        self._admissible_actions = self._extract_valid_actions_from_info(self._current_info)
        self._action_templates = self._extract_action_templates()
        self._objects = self._extract_objects()

        # Enforce step limit
        if self._step_count >= self.max_steps and not done:
            done = True
            observation = (
                f"TIMEOUT: Maximum steps ({self.max_steps}) reached.\n\n{observation}"
            )
            self._current_obs = str(observation)

        score = self._extract_score(info, default=self._score)
        score_delta = score - self._score
        success = self._extract_success(
            done=done, reward=reward, info=info, score=score
        )
        valid_action = self._extract_valid_action(
            info=info, observation=self._current_obs
        )

        # Use ScienceWorld's score delta as reward signal (0-100 scale → 0-1)
        if score_delta > 0:
            final_reward = score_delta / 100.0
        elif valid_action is False:
            final_reward = self.REWARD_INVALID_ACTION
        else:
            final_reward = self.REWARD_STEP

        self._done = bool(done)
        self._score = score

        return StepResult(
            observation=self._format_observation(self._current_obs),
            reward=final_reward,
            done=self._done,
            info={
                "action_taken": parsed_action,
                "task": self._task_desc,
                "task_name": self._current_task_name,
                "variation": self._current_variation,
                "score": score,
                "raw_reward": float(reward)
                if isinstance(reward, (int, float))
                else reward,
                "success": success,
                "valid_action": valid_action,
            },
        )

    @staticmethod
    def _coerce_optional_bool(value: Any) -> Optional[bool]:
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
        return None

    @staticmethod
    def _normalize_reward(reward: Any) -> float:
        if not isinstance(reward, (int, float)):
            return 0.0
        reward = float(reward)
        if abs(reward) > 10.0:
            reward = reward / 10.0
        return max(min(reward, 10.0), -10.0)

    @staticmethod
    def _extract_score(info: Dict[str, Any], default: float = 0.0) -> float:
        if not isinstance(info, dict):
            return float(default)
        for key in ("score", "normalizedScore", "taskScore"):
            value = info.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return float(default)

    def _extract_success(
        self,
        done: bool,
        reward: Any,
        info: Dict[str, Any],
        score: float,
    ) -> bool:
        for key in ("success", "taskCompleted", "completed", "isCompleted"):
            value = self._coerce_optional_bool(info.get(key))
            if value is not None:
                return value
        if done and score >= 100.0:
            return True
        if done and isinstance(reward, (int, float)) and float(reward) >= 10.0:
            return True
        return False

    def _extract_valid_action(
        self, info: Dict[str, Any], observation: str
    ) -> Optional[bool]:
        for key in ("valid", "action_valid", "validAction"):
            value = info.get(key)
            # ScienceWorld stores valid action *list* under "valid" — skip it
            if isinstance(value, (list, dict)):
                continue
            coerced = self._coerce_optional_bool(value)
            if coerced is not None:
                return coerced

        lowered = observation.lower()
        invalid_markers = [
            "not a valid action",
            "invalid action",
            "i don't understand",
            "nothing happens",
            "you can't do that",
            "no known action matches that input",
            "ambiguous request",
        ]
        if any(marker in lowered for marker in invalid_markers):
            return False
        return None

    def _call_optional_text_method(self, method_name: str) -> str:
        self._ensure_env()
        method = getattr(self._env, method_name, None)
        if not callable(method):
            return ""
        try:
            value = method()
        except Exception:
            return ""
        return str(value).strip() if value is not None else ""

    def _get_task_description(self) -> str:
        task_desc = self._call_optional_text_method("getTaskDescription")
        if task_desc:
            return task_desc
        task_desc = self._call_optional_text_method("taskdescription")
        if task_desc:
            return task_desc
        return f"Complete ScienceWorld task {self._current_task_name}."

    def _extract_valid_actions_from_info(self, info: Dict[str, Any]) -> List[str]:
        """Extract valid action list from ScienceWorld's info['valid']."""
        valid = info.get("valid", [])
        if not isinstance(valid, list):
            return []
        actions = []
        seen = set()
        for v in valid:
            a = str(v).strip()
            if a and a not in seen:
                seen.add(a)
                actions.append(a)
        return actions

    def _extract_action_templates(self) -> List[str]:
        """Get possible action templates from ScienceWorld (e.g. 'open OBJ')."""
        self._ensure_env()
        method = getattr(self._env, "getPossibleActions", None)
        if not callable(method):
            return []
        try:
            templates = method()
        except Exception:
            return []
        if not isinstance(templates, (list, tuple)):
            return []
        return [str(t).strip() for t in templates if str(t).strip()]

    def _extract_objects(self) -> List[str]:
        """Get possible objects from ScienceWorld (e.g. 'door to kitchen')."""
        self._ensure_env()
        method = getattr(self._env, "getPossibleObjects", None)
        if not callable(method):
            return []
        try:
            objects = method()
        except Exception:
            return []
        if not isinstance(objects, (list, tuple)):
            return []
        return [str(o).strip() for o in objects if str(o).strip()]

    def _format_observation(self, observation: str) -> str:
        parts = [
            "=== Current State ===",
            observation.strip() or "(empty observation)",
        ]

        score_line = f"Score: {self._score:.2f}"
        if self._current_variation >= 0:
            score_line += f" | Variation: {self._current_variation}"
        parts.extend(["", score_line])

        inventory = self._call_optional_text_method("inventory")
        if inventory:
            parts.extend(["", "=== Inventory ===", inventory])

        look = self._call_optional_text_method("look")
        if look and look != observation:
            parts.extend(["", "=== Look ===", look])

        if self._action_templates:
            parts.extend(["", "=== Action Templates ==="])
            parts.extend(f"- {template}" for template in self._action_templates)

        if self._objects:
            parts.extend(["", "=== Objects ==="])
            parts.extend(f"- {obj}" for obj in self._objects)

        return "\n".join(parts)

    def _parse_action(self, raw_action: str) -> str:
        match = re.search(
            r"<action>\s*(.*?)\s*</action>", raw_action, re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()

        lines = [
            line.strip() for line in raw_action.strip().splitlines() if line.strip()
        ]
        return lines[-1] if lines else "look around"

    def get_system_prompt(self) -> str:
        return (
            "You are an AI assistant playing ScienceWorld, a text-based science environment.\n"
            "Your goal is to complete the task by issuing grounded text actions.\n\n"
            "IMPORTANT: You MUST respond in the following format:\n"
            "1. Think briefly in <thinking></thinking>\n"
            "2. Output exactly one environment action in <action></action>\n\n"
            "Examples of actions:\n"
            "- look around\n"
            "- inventory\n"
            "- examine beaker\n"
            "- open door to art studio\n"
            "- pour cup containing blue paint in glass cup\n\n"
            "Example response:\n"
            "<thinking>I should inspect the room before manipulating objects.</thinking>\n"
            "<action>look around</action>"
        )

    def get_initial_user_message(self) -> str:
        return (
            f"Task: {self._task_desc}\n\n"
            "Interact with the environment and complete the task."
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "task": self._task_desc,
            "task_name": self._current_task_name,
            "variation": self._current_variation,
            "score": self._score,
            "step_count": self._step_count,
            "max_steps": self.max_steps,
            "done": self._done,
            "candidate_actions": self._admissible_actions,
            "action_templates": self._action_templates,
            "objects": self._objects,
        }

    def generate_initial_state(self) -> Dict[str, Any]:
        task_name, variation = self._select_task_and_variation()
        return {
            "task_name": task_name,
            "variation": variation,
            "seed": random.randint(0, 1_000_000),
        }

    def get_user_message_with_state(
        self,
        task_name: Optional[str] = None,
        variation: Optional[int] = None,
        **kwargs,
    ) -> str:
        self.reset(task_name=task_name, variation=variation, **kwargs)
        return (
            f"Task: {self._task_desc}\n\n"
            f"{self._format_observation(self._current_obs)}\n\n"
            "What will you do next?"
        )

    def get_interaction_name(self) -> str:
        return "sciworld"
