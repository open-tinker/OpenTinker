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

import asyncio
import logging
import os
import re
import threading
import zlib
from urllib.parse import urlparse, urlunparse
from typing import Any, Optional, Callable
from uuid import uuid4

import aiohttp

from verl.interactions.base import BaseInteraction

# Try to import Ray for worker ID detection
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Module-level cache for worker_id and bound_endpoint (persists across all instances in the same process)
# Key: (env_shards, base_port) tuple, Value: (worker_id, bound_endpoint) tuple
_worker_endpoint_cache: dict[tuple[int, int], tuple[int, str]] = {}
_cache_lock = threading.Lock()


class GymEnvironmentInteraction(BaseInteraction):
    """Interaction class for OpenAI Gym-like environments.

    This class wraps a Gym environment and provides the BaseInteraction interface
    for use with GenericAgentLoop. It supports both local environments (passed
    directly) and remote environments (accessed via HTTP API).

    Configuration options:
        - env_endpoint: HTTP endpoint for remote environment API
        - env_shards: Number of shards (servers on consecutive ports)
        - bind_worker_to_endpoint: If True, each worker is bound to a specific endpoint
          (1-to-1 worker <-> endpoint). Requires worker count == env_shards. Worker ID is
          detected from Ray actor name (e.g., "agent_loop_worker_0" -> worker_id=0).
          Worker ID and endpoint are cached at module level, persisting across all
          instances in the same process.
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
            env_endpoint: "http://localhost:8091"
            env_shards: 8  # Will use ports 8091..8098
            max_steps: 100
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.env_endpoint: Optional[str] = config.get("env_endpoint")
        self.env_shards: int = int(config.get("env_shards", 1) or 1)
        self.env_factory: Optional[Callable] = config.get("env_factory")
        self.max_steps: int = config.get("max_steps", 100)
        self.observation_template: str = config.get(
            "observation_template", "{observation}"
        )
        # Job ID for statistics isolation when using shared game servers
        self.job_id: str = config.get("job_id", "default")

        # When True: bind this worker to a specific endpoint (1-to-1 worker <-> endpoint)
        self.bind_worker_to_endpoint: bool = bool(config.get("bind_worker_to_endpoint", False))
        
        # Worker-bound endpoint (set if bind_worker_to_endpoint is True)
        self._bound_endpoint: Optional[str] = None
        self._worker_id: Optional[int] = None

        # Generate sharded endpoints if env_shards > 1
        self.env_endpoints: Optional[list[str]] = None
        if self.env_endpoint and self.env_shards > 1:
            parsed = urlparse(self.env_endpoint)
            if parsed.port:
                base_port = parsed.port
                self.env_endpoints = [
                    urlunparse(parsed._replace(netloc=f"{parsed.hostname}:{base_port + i}"))
                    for i in range(self.env_shards)
                ]
                logger.info(
                    f"[GymEnvironmentInteraction] Sharded mode: {self.env_shards} shards "
                    f"on ports {base_port}..{base_port + self.env_shards - 1}"
                )
        
        # Bind worker to endpoint if requested
        if self.bind_worker_to_endpoint:
            if not self.env_endpoints:
                raise ValueError(
                    "[GymEnvironmentInteraction] bind_worker_to_endpoint=True requires env_shards > 1"
                )
            # Use module-level cache keyed by (env_shards, base_port) to persist across all instances
            parsed = urlparse(self.env_endpoint)
            base_port = parsed.port if parsed.port else 0
            cache_key = (self.env_shards, base_port)
            
            with _cache_lock:
                if cache_key not in _worker_endpoint_cache:
                    # First time: detect worker_id and compute bound_endpoint
                    detected_id = self._detect_worker_id()
                    
                    if detected_id is None:
                        raise RuntimeError(
                            "[GymEnvironmentInteraction] bind_worker_to_endpoint=True but could not detect worker_id. "
                            "Make sure this is running in a Ray actor with name like 'agent_loop_worker_0', "
                            "or set RAY_WORKER_ID environment variable."
                        )
                    if detected_id < 0 or detected_id >= len(self.env_endpoints):
                        raise ValueError(
                            f"[GymEnvironmentInteraction] worker_id={detected_id} is out of range "
                            f"for {len(self.env_endpoints)} endpoints. Ensure agent_num_workers == env_shards."
                        )
                    bound_endpoint = self.env_endpoints[detected_id]
                    _worker_endpoint_cache[cache_key] = (detected_id, bound_endpoint)
                    logger.info(
                        f"[GymEnvironmentInteraction] Detected worker_id={detected_id} "
                        f"bound to endpoint={bound_endpoint} "
                        f"(module-level cache, persists across all instances in this process)"
                    )
                # Use cached values for this instance
                self._worker_id, self._bound_endpoint = _worker_endpoint_cache[cache_key]

        # Session storage: maps instance_id to environment state
        self._instance_dict: dict[str, dict[str, Any]] = {}

        # For local environments, we can store the env objects
        self._local_envs: dict[str, Any] = {}
    
    def _detect_worker_id(self) -> Optional[int]:
        """Detect worker ID from Ray actor name using get_actor_name() method.
        
        This uses the correct Ray API: runtime_context.get_actor_name()
        
        Returns:
            Worker ID (0-based) if detected, None otherwise.
        """
        if not RAY_AVAILABLE:
            return None
        
        try:
            runtime_context = ray.get_runtime_context()
            
            # Use get_actor_name() method - this is the correct Ray API
            actor_name = runtime_context.get_actor_name()
            if actor_name:
                match = re.search(r"agent_loop_worker_(\d+)", str(actor_name))
                if match:
                    worker_id = int(match.group(1))
                    logger.info(f"[GymEnvironmentInteraction] Detected worker_id={worker_id} from actor_name={actor_name}")
                    return worker_id
            
            # Fallback: Try environment variable
            worker_id_env = os.environ.get("RAY_WORKER_ID")
            if worker_id_env:
                try:
                    worker_id = int(worker_id_env)
                    logger.info(f"[GymEnvironmentInteraction] Using worker_id={worker_id} from RAY_WORKER_ID env var")
                    return worker_id
                except ValueError:
                    logger.warning(f"[GymEnvironmentInteraction] Invalid RAY_WORKER_ID value: {worker_id_env}")
            
        except Exception as e:
            logger.debug(f"[GymEnvironmentInteraction] Failed to detect worker_id: {e}")
        
        return None

    def _get_endpoint(self, instance_id: str) -> str:
        """Get the endpoint for this instance_id (supports sharding).

        If bind_worker_to_endpoint is True: return the bound endpoint for this worker.
        Otherwise: hash instance_id to pick shard (may collide).
        """
        if self.bind_worker_to_endpoint and self._bound_endpoint:
            return self._bound_endpoint
        if self.env_endpoints:
            # Sharded mode: hash instance_id to pick shard
            idx = zlib.crc32(instance_id.encode("utf-8")) % len(self.env_endpoints)
            return self.env_endpoints[idx]
        return self.env_endpoint

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

        elif self.env_endpoint is not None:
            # Remote environment via HTTP
            endpoint = self._get_endpoint(instance_id)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{endpoint}/reset",
                    json={"instance_id": instance_id, "job_id": self.job_id, **kwargs},
                    timeout=aiohttp.ClientTimeout(total=600000),
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"Environment reset failed: {await response.text()}"
                        )
                    data = await response.json()
                    # Return full data if board_state is present, else just observation
                    if "board_state" in data:
                        return data
                    return data.get("observation", "")

        else:
            raise ValueError("Either env_factory or env_endpoint must be configured")

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

        elif self.env_endpoint is not None:
            # Remote environment via HTTP
            endpoint = self._get_endpoint(instance_id)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{endpoint}/step",
                    json={
                        "instance_id": instance_id,
                        "job_id": self.job_id,
                        "action": action,
                    },
                    timeout=aiohttp.ClientTimeout(total=600000),
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"Environment step failed: {await response.text()}"
                        )
                    data = await response.json()
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
