# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gym Environment Interaction for LLM-Environment RL Training.

This module provides an interaction class for interfacing with OpenAI Gym-like
environments. It implements the BaseInteraction interface and handles:
- Environment initialization (reset)
- Step execution (action -> observation, reward, done)
- Session management for multiple concurrent trajectories

The environment can be either local (imported directly) or remote (via HTTP API).
"""

import logging
import os
import zlib
from urllib.parse import urlparse, urlunparse
from typing import Any, Optional, Callable
from uuid import uuid4

import aiohttp

from verl.interactions.base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class GymEnvironmentInteraction(BaseInteraction):
    """Interaction class for OpenAI Gym-like environments.

    This class wraps a Gym environment and provides the BaseInteraction interface
    for use with GenericAgentLoop. It supports both local environments (passed
    directly) and remote environments (accessed via HTTP API).

    Configuration options:
        - env_endpoint: HTTP endpoint for remote environment API
        - env_factory: Callable that creates a local environment instance
        - max_steps: Maximum number of steps per episode
        - observation_template: Template for formatting observations as messages

    Gym API Format (old version):
        step(action) -> (observation, reward, done, info)
        reset() -> observation

    Example usage in config:
        - name: gym_env
          class: verl.interactions.gym_environment_interaction.GymEnvironmentInteraction
          config:
            env_endpoint: "http://localhost:8080"
            max_steps: 100
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # Remote environment endpoint(s).
        #
        # - env_endpoint: single URL string (backward compatible)
        # - env_endpoints: list of URL strings (recommended for sharded servers)
        # - env_shards: if set (>1) and env_endpoint has a port, we derive env_endpoints
        self.env_endpoint: Optional[str] = config.get("env_endpoint")
        self.env_endpoints: Optional[list[str]] = config.get("env_endpoints")
        self.env_shards: int = int(config.get("env_shards", 1) or 1)
        if self.env_endpoints is not None and isinstance(self.env_endpoints, str):
            # Allow comma-separated string for convenience.
            self.env_endpoints = [
                s.strip() for s in self.env_endpoints.split(",") if s.strip()
            ]

        # Normalize endpoint host. "0.0.0.0" is a bind-all address and is not a
        # reliable destination address for clients (can be flaky / invalid).
        # If the user configured env_host=0.0.0.0, rewrite to 127.0.0.1 for local access.
        def _normalize(url: str) -> str:
            if "//0.0.0.0" not in url:
                return url
            try:
                parsed = urlparse(url)
                if parsed.hostname == "0.0.0.0":
                    netloc = f"127.0.0.1:{parsed.port}" if parsed.port else "127.0.0.1"
                    parsed = parsed._replace(netloc=netloc)
                    return urlunparse(parsed)
            except Exception:
                return url
            return url

        if self.env_endpoints:
            self.env_endpoints = [_normalize(u) for u in self.env_endpoints]
        if self.env_endpoint:
            normalized = _normalize(self.env_endpoint)
            if normalized != self.env_endpoint:
                logger.warning(
                    f"[GymEnvironmentInteraction] Rewrote env_endpoint to {normalized} "
                    "(0.0.0.0 is not a valid connect target)."
                )
            self.env_endpoint = normalized

        # Derive env_endpoints from env_endpoint + env_shards if requested.
        if (not self.env_endpoints) and self.env_endpoint and self.env_shards > 1:
            try:
                parsed = urlparse(self.env_endpoint)
                if parsed.port is None:
                    raise ValueError("env_endpoint has no port")
                base_port = parsed.port
                self.env_endpoints = []
                for i in range(self.env_shards):
                    netloc = f"{parsed.hostname}:{base_port + i}"
                    self.env_endpoints.append(
                        urlunparse(parsed._replace(netloc=netloc))
                    )
                logger.warning(
                    f"[GymEnvironmentInteraction] Using sharded env_endpoints (env_shards={self.env_shards}): "
                    f"{self.env_endpoints[:3]}{'...' if len(self.env_endpoints) > 3 else ''}"
                )
            except Exception as e:
                logger.warning(
                    f"[GymEnvironmentInteraction] Failed to derive env_endpoints from env_endpoint/env_shards: {e}. "
                    "Falling back to single env_endpoint."
                )
        self.env_factory: Optional[Callable] = config.get("env_factory")
        self.max_steps: int = config.get("max_steps", 100)
        self.observation_template: str = config.get(
            "observation_template", "{observation}"
        )
        # Job ID for statistics isolation when using shared game servers
        self.job_id: str = config.get("job_id", "default")

        # Session storage: maps instance_id to environment state
        self._instance_dict: dict[str, dict[str, Any]] = {}

        # For local environments, we can store the env objects
        self._local_envs: dict[str, Any] = {}

        # For remote environments, keep a per-instance HTTP session.
        #
        # Why: when the env server runs with multiple uvicorn workers (multi-process),
        # each worker has its own in-memory instance registry. If the client creates a
        # new TCP connection per request, /reset and /step may hit different workers,
        # causing "Instance ... not found. Call /reset first."
        #
        # A dedicated session per instance_id strongly increases connection stickiness
        # (HTTP keep-alive), so all calls for a trajectory go to the same worker.
        self._remote_sessions: dict[str, aiohttp.ClientSession] = {}
        self._remote_timeout = aiohttp.ClientTimeout(total=600000)

    def _route_endpoint(self, instance_id: str) -> str:
        """Pick a stable endpoint for this instance_id.

        When the env server is sharded across multiple ports (multiple single-worker
        processes), this ensures all /reset and /step calls for a trajectory hit
        the same shard, avoiding 'Instance not found' issues.
        """
        if self.env_endpoints:
            idx = zlib.crc32(instance_id.encode("utf-8")) % len(self.env_endpoints)
            return self.env_endpoints[idx]
        if self.env_endpoint:
            return self.env_endpoint
        raise ValueError(
            "Either env_factory or env_endpoint/env_endpoints must be configured"
        )

    def _get_remote_session(self, instance_id: str) -> aiohttp.ClientSession:
        """Get or create a dedicated aiohttp session for this instance_id.

        Session is keyed by (routed endpoint + instance_id) so a trajectory never
        accidentally reuses a keep-alive connection across different shards.
        """
        session_key = f"{self._route_endpoint(instance_id)}::{instance_id}"
        session = self._remote_sessions.get(session_key)
        if session is not None and not session.closed:
            return session

        connector = aiohttp.TCPConnector(
            limit=1,
            limit_per_host=1,
            enable_cleanup_closed=True,
            ttl_dns_cache=300,
            keepalive_timeout=300,
        )
        session = aiohttp.ClientSession(connector=connector)
        self._remote_sessions[session_key] = session
        return session

    async def _retryable_post_json(
        self, instance_id: str, url: str, payload: dict, timeout: aiohttp.ClientTimeout
    ) -> dict:
        """POST JSON with a single retry on connection reset (worker restart / stale keepalive)."""
        session = self._get_remote_session(instance_id)
        try:
            async with session.post(url, json=payload, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"{await response.text()}")
                return await response.json()
        except aiohttp.ClientOSError as e:
            # Common when server side closes a keep-alive connection (errno 104).
            logger.warning(
                f"[GymEnvironmentInteraction] POST {url} failed with {type(e).__name__}: {e}. Retrying once..."
            )
            # Recreate session and retry once.
            session_key = f"{self._route_endpoint(instance_id)}::{instance_id}"
            old = self._remote_sessions.pop(session_key, None)
            if old is not None and not old.closed:
                try:
                    await old.close()
                except Exception:
                    pass
            session = self._get_remote_session(instance_id)
            async with session.post(url, json=payload, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"{await response.text()}")
                return await response.json()

    async def start_interaction(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> str:
        """Initialize a new environment instance.

        This calls env.reset() to get the initial observation and sets up
        tracking for this trajectory.

        Args:
            instance_id: Optional instance ID, will be generated if not provided
            **kwargs: Additional arguments, may include:
                - initial_observation: Skip reset and use this observation
                - env_kwargs: Arguments to pass to env.reset()

        Returns:
            The instance ID for this trajectory
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Get initial observation (either from reset or provided)
        initial_observation = kwargs.get("initial_observation")
        env_kwargs = kwargs.get("env_kwargs", {})

        if initial_observation is None:
            reset_data = await self._call_env_reset(instance_id, **env_kwargs)
            # reset_data can be either a string (old format) or dict with observation + board_state
            if isinstance(reset_data, dict):
                initial_observation = reset_data.get("observation", "")
                initial_board_state = reset_data.get("board_state")
            else:
                initial_observation = reset_data
                initial_board_state = None
        else:
            initial_board_state = None

        self._instance_dict[instance_id] = {
            "observations": [initial_observation],
            "rewards": [],
            "done": False,
            "step_count": 0,
            "cumulative_reward": 0.0,
            "info": {},
            "initial_board_state": initial_board_state,  # Store initial board state
        }

        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        """Execute one step in the environment.

        This extracts the action from the last assistant message, calls env.step(),
        and returns the observation as the next user message content.

        Args:
            instance_id: The trajectory instance ID
            messages: Conversation history, last assistant message contains action
            **kwargs: Additional arguments

        Returns:
            Tuple of (should_terminate, observation, reward, info)
            - should_terminate: True if episode is done or max_steps reached
            - observation: String observation from environment (formatted)
            - reward: Reward from this step
            - info: Additional info from environment step
        """
        if instance_id not in self._instance_dict:
            raise ValueError(
                f"Instance {instance_id} not found. Call start_interaction first."
            )

        instance_data = self._instance_dict[instance_id]

        # Check if already done
        if instance_data["done"]:
            return True, "Episode has ended.", 0.0, {}

        # Extract action from the last assistant message
        action = self._extract_action(messages)

        # Call environment step
        observation, reward, done, info = await self._call_env_step(instance_id, action)

        # Update instance data
        instance_data["observations"].append(observation)
        instance_data["rewards"].append(reward)
        instance_data["done"] = done
        instance_data["step_count"] += 1
        instance_data["cumulative_reward"] += reward
        instance_data["info"] = info

        # Check max steps
        if instance_data["step_count"] >= self.max_steps:
            done = True
            instance_data["done"] = True

        # Format observation using template
        formatted_observation = self.observation_template.format(
            observation=observation,
            reward=reward,
            step=instance_data["step_count"],
            cumulative_reward=instance_data["cumulative_reward"],
            **info,
        )

        return done, formatted_observation, reward, info

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Calculate the final score for this trajectory.

        Returns the cumulative reward for the episode.
        """
        if instance_id not in self._instance_dict:
            return 0.0
        return self._instance_dict[instance_id]["cumulative_reward"]

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this trajectory.

        Removes the instance data and closes any local environment.
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

        if instance_id in self._local_envs:
            env = self._local_envs.pop(instance_id)
            if hasattr(env, "close"):
                env.close()

        # Notify remote game server to return instance to pool
        if self.env_endpoint is not None or self.env_endpoints is not None:
            try:
                session = self._get_remote_session(instance_id)
                async with session.post(
                    f"{self._route_endpoint(instance_id)}/finalize",
                    json={"instance_id": instance_id, "job_id": self.job_id},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status != 200:
                        logger.warning(
                            f"[finalize_interaction] Server returned {response.status}"
                        )
            except Exception as e:
                logger.warning(f"[finalize_interaction] Failed to notify server: {e}")
            finally:
                # Always close and delete the dedicated session.
                session_key = f"{self._route_endpoint(instance_id)}::{instance_id}"
                session = self._remote_sessions.pop(session_key, None)
                if session is not None and not session.closed:
                    try:
                        await session.close()
                    except Exception:
                        pass

    # #Note(Siqi): this will likely cause a bug
    # async def mark_truncated(self, instance_id: str, reason: str = "external_limit", **kwargs) -> None:
    #     """Mark a game as done due to external truncation (e.g., response_length limit).

    #     This ensures the game server records the game as completed even when
    #     GenericAgentLoop terminates early due to external limits rather than
    #     the environment returning done=True.

    #     Args:
    #         instance_id: The trajectory instance ID
    #         reason: Reason for truncation (for logging/debugging)
    #         **kwargs: Additional arguments
    #     """

    #     if instance_id not in self._instance_dict:
    #         return

    #     instance_data = self._instance_dict[instance_id]

    #     # If already done, no need to mark again
    #     if instance_data.get("done", False):
    #         return

    #     # Mark as done locally
    #     instance_data["done"] = True

    #     # Notify remote game server if using remote environment
    #     if self.env_endpoint is not None:
    #         try:
    #             async with aiohttp.ClientSession() as session:
    #                 # Call /step with a special truncation action to mark the game as done
    #                 # This lets the game server record the game as completed
    #                 async with session.post(
    #                     f"{self.env_endpoint}/step",
    #                     json={
    #                         "instance_id": instance_id,
    #                         "action": f"[TRUNCATED: {reason}]",
    #                         "truncated": True,  # Optional flag for game server
    #                     },
    #                     timeout=aiohttp.ClientTimeout(total=600000)
    #                 ) as response:
    #                     if response.status == 200:
    #                         data = await response.json()
    #                     else:
    #                         logger.warning(f"[mark_truncated] Failed to notify server: {response.status}")
    #         except Exception as e:
    #             # Don't fail the training just because we couldn't notify the game server
    #             logger.warning(f"[mark_truncated] Error notifying server: {e}")

    def _extract_action(self, messages: list[dict[str, Any]]) -> str:
        """Extract the action from the last assistant message.

        The action is the content of the most recent assistant message.
        Subclasses can override this to implement action parsing.
        """
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                return messages[i].get("content", "")
        return ""

    async def _call_env_reset(self, instance_id: str, **kwargs):
        """Call env.reset() to get initial observation.

        Supports both local and remote environments.

        Returns:
            For remote envs: Full response dict if it contains board_state, else just observation string
            For local envs: Observation string
        """
        if self.env_factory is not None:
            # Local environment
            env = self.env_factory()
            self._local_envs[instance_id] = env
            observation = env.reset(**kwargs)
            return str(observation)

        elif self.env_endpoint is not None or self.env_endpoints is not None:
            # Remote environment via HTTP
            url = f"{self._route_endpoint(instance_id)}/reset"
            payload = {"instance_id": instance_id, "job_id": self.job_id, **kwargs}
            try:
                data = await self._retryable_post_json(
                    instance_id, url, payload, timeout=self._remote_timeout
                )
            except RuntimeError as e:
                raise RuntimeError(f"Environment reset failed: {e}")
            # Return full data if board_state is present, else just observation
            if "board_state" in data:
                return data
            return data.get("observation", "")

        else:
            raise ValueError(
                "Either env_factory or env_endpoint/env_endpoints must be configured"
            )

    async def _call_env_step(
        self, instance_id: str, action: str
    ) -> tuple[str, float, bool, dict]:
        """Call env.step(action) to get next observation, reward, done, info.

        Supports both local and remote environments.
        Returns (observation, reward, done, info) in old Gym format.
        """
        if instance_id in self._local_envs:
            # Local environment
            env = self._local_envs[instance_id]
            # Assume action is a string that the environment can process
            result = env.step(action)
            observation, reward, done, info = result
            return str(observation), float(reward), bool(done), info

        elif self.env_endpoint is not None or self.env_endpoints is not None:
            # Remote environment via HTTP
            url = f"{self._route_endpoint(instance_id)}/step"
            payload = {
                "instance_id": instance_id,
                "job_id": self.job_id,
                "action": action,
            }
            try:
                data = await self._retryable_post_json(
                    instance_id, url, payload, timeout=self._remote_timeout
                )
            except RuntimeError as e:
                raise RuntimeError(f"Environment step failed: {e}")
            return (
                data.get("observation", ""),
                float(data.get("reward", 0.0)),
                bool(data.get("done", False)),
                data.get("info", {}),
            )

        else:
            raise ValueError("No environment available for this instance")


class SimpleTextEnvironmentInteraction(BaseInteraction):
    """A simple text-based environment for testing and basic scenarios.

    This interaction doesn't connect to any external environment. Instead,
    it uses a custom response function to generate observations based on
    the conversation history. This is useful for:
    - Testing the GenericAgentLoop
    - Simple rule-based environments
    - Mock environments for development

    Configuration:
        - response_fn: Function(messages) -> (should_terminate, observation, reward, info)
        - max_turns: Maximum number of turns before termination
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.response_fn: Optional[Callable] = config.get("response_fn")
        self.max_turns: int = config.get("max_turns", 10)
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def start_interaction(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {
            "turn_count": 0,
            "rewards": [],
            **kwargs,  # Store any additional initial data
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        if instance_id not in self._instance_dict:
            raise ValueError(f"Instance {instance_id} not found")

        instance_data = self._instance_dict[instance_id]
        instance_data["turn_count"] += 1

        # Check max turns
        if instance_data["turn_count"] >= self.max_turns:
            return True, "Maximum turns reached.", 0.0, {}

        if self.response_fn is not None:
            # Use custom response function
            result = self.response_fn(messages, instance_data, **kwargs)
            if len(result) == 4:
                should_terminate, observation, reward, info = result
            else:
                should_terminate, observation, reward = result
                info = {}
        else:
            # Default behavior: simple acknowledgment
            should_terminate = False
            observation = (
                f"Turn {instance_data['turn_count']}: Observation of your response."
            )
            reward = 0.0
            info = {}

        instance_data["rewards"].append(reward)
        return should_terminate, observation, reward, info

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        if instance_id not in self._instance_dict:
            return 0.0
        return sum(self._instance_dict[instance_id].get("rewards", []))

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
