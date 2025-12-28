#!/usr/bin/env python3
"""ALFWorld Game Implementation (Text Engine Version).

This module provides the ALFWorldGame class that implements AbstractGame interface.
It wraps the ALFWorld text-based environment for LLM training.

Example:
    from alfworld_game import ALFWorldGame

    game = ALFWorldGame()
    obs = game.reset()
    result = game.step("go to desk 1")
"""

import glob
import os
import random
import re
import threading
from typing import Any, Dict, List, Optional
import logging

from opentinker.environment.base_game import AbstractGame, StepResult

# ALFWorld imports - install with: pip install alfworld
try:
    import alfworld.agents.environment as alfworld_env
    import textworld
    from textworld.envs.wrappers import Filter
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
    import yaml

    ALFWORLD_AVAILABLE = True
except ImportError:
    ALFWORLD_AVAILABLE = False
    logging.warning("alfworld not installed. Install with: pip install alfworld")


class ALFWorldGame(AbstractGame):
    """ALFWorld text-based environment game implementation.

    This implementation wraps ALFWorld's TextWorld environment for LLM RL training.
    The agent interacts through text commands to complete household tasks.

    Attributes:
        max_steps: Maximum steps per episode (default: 50)
        task_types: List of task types to sample from (default: all)
    """

    # Reward constants
    REWARD_SUCCESS = 10.0
    REWARD_FAILURE = -1.0
    REWARD_STEP = -0.01  # Small penalty per step to encourage efficiency
    REWARD_INVALID_ACTION = -0.1

    # Limits
    DEFAULT_MAX_STEPS = 20

    # Task types in ALFWorld
    ALL_TASK_TYPES = [
        "pick_and_place_simple",
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
    ]

    # Process-level cache for ALFWorld environments (safe within same process)
    # Key: (config_path, split, num_games) -> initialized environment
    # Note: Each Ray worker process gets its own cache, avoiding cross-process issues
    # IMPORTANT: Set to False to disable sharing when running parallel instances
    _shared_envs: dict = {}
    _use_shared_env: bool = False  # Disabled by default to prevent state conflicts

    # ==========================================
    # OPTIMIZATION: Shared game path cache
    # Only the first instance scans the disk, others reuse the cached paths
    # ==========================================
    _cached_game_paths: Dict[
        str, List[str]
    ] = {}  # key: cache_key -> list of game paths
    _cache_lock = threading.Lock()

    # NOTE: TextWorld's tatsu parser is NOT thread-safe.
    # Parallelism is achieved via sharding (--shards N launches N independent server processes).

    def __init__(
        self,
        config_path: Optional[str] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        task_types: Optional[List[str]] = None,
        split: str = "train",
        num_games: int = -1,
        use_shared_env: bool = False,  # Default to per-instance env for safety
    ):
        """Initialize ALFWorld game.

        Args:
            config_path: Path to ALFWorld config file (base_config.yaml)
            max_steps: Maximum steps per episode
            task_types: Task types to sample from (None = all types)
            split: Dataset split ("train", "eval_in_distribution", "eval_out_of_distribution")
            num_games: Number of games to load (-1 = all games, e.g. 64 for faster loading)
            use_shared_env: If True, share env across instances (fast but causes state conflicts with parallel runs).
                           If False, each instance gets its own env (slower init but correct).
        """
        if not ALFWORLD_AVAILABLE:
            raise ImportError(
                "alfworld package not installed. " "Install with: pip install alfworld"
            )

        self.config_path = config_path
        self.max_steps = max_steps
        self.task_types = task_types or self.ALL_TASK_TYPES
        self.split = split
        self.num_games = num_games
        self._use_shared_env = use_shared_env

        # Load ALFWorld config
        self._load_alfworld_config()

        # Game state (instance-specific, NOT shared)
        self._env = None  # Will be created in reset()
        self._own_env = None  # Per-instance env when not using shared env
        self._tw_env = None  # TextWorld environment instance (for optimized mode)
        self._current_obs = None
        self._current_info = None
        self._step_count = 0
        self._task_desc = ""
        self._admissible_commands = []
        self._done = False
        self._initialized = False  # Track if engine is ready

    def _load_alfworld_config(self):
        """Load ALFWorld configuration."""

        if self.config_path:
            with open(self.config_path, "r") as f:
                self._config = yaml.safe_load(f)
        else:
            # ALFWorld data directory (need to run 'alfworld-download' first)
            alfworld_data = os.path.expandvars(
                os.environ.get("ALFWORLD_DATA", "$HOME/.cache/alfworld")
            )

            # Use default config with all required fields
            self._config = {
                "general": {
                    "training_method": "dqn",  # or 'dagger'
                },
                "env": {
                    "type": "AlfredTWEnv",
                    "regen_game_files": False,
                    "domain_randomization": False,
                    "goal_desc_human_anns_prob": 0,  # Required by AlfredTWEnv
                    "task_types": [1, 2, 3, 4, 5, 6],  # All 6 task types
                    "expert_type": "handcoded",  # or 'planner'
                    "registration": {
                        "batch_size": 1,
                    },
                    "split": self.split,
                },
                "dataset": {
                    "data_path": f"{alfworld_data}/json_2.1.1/train",
                    "eval_id_data_path": f"{alfworld_data}/json_2.1.1/valid_seen",
                    "eval_ood_data_path": f"{alfworld_data}/json_2.1.1/valid_unseen",
                    "num_train_games": self.num_games,  # -1 means use all games
                    "num_eval_games": self.num_games,
                },
                "logic": {
                    "domain": f"{alfworld_data}/logic/alfred.pddl",
                    "grammar": f"{alfworld_data}/logic/alfred.twl2",
                },
                "rl": {
                    "training": {
                        "max_nb_steps_per_episode": self.max_steps,
                    },
                },
            }

    def _get_cached_game_paths(self) -> List[str]:
        """Get cached game paths. Only the first instance scans the disk.

        OPTIMIZATION: This avoids 32 instances each scanning 8810 game directories.
        Instead, only the first instance scans, and others reuse the cached result.

        ALFWorld directory structure:
            json_2.1.1/train/pick_and_place_simple-xxx/trial_xxx/game.tw-pddl

        Returns:
            List of trial directory paths (containing game.tw-pddl files)
        """
        # Create a unique cache key based on split and task types
        cache_key = f"{self.split}:{','.join(sorted(self.task_types))}:{self.num_games}"

        with ALFWorldGame._cache_lock:
            if cache_key not in ALFWorldGame._cached_game_paths:
                # First instance: scan the disk
                alfworld_data = os.path.expandvars(
                    os.environ.get("ALFWORLD_DATA", "$HOME/.cache/alfworld")
                )

                # Determine base path based on split
                if self.split == "train":
                    base_path = f"{alfworld_data}/json_2.1.1/train"
                elif self.split == "eval_in_distribution":
                    base_path = f"{alfworld_data}/json_2.1.1/valid_seen"
                elif self.split == "eval_out_of_distribution":
                    base_path = f"{alfworld_data}/json_2.1.1/valid_unseen"
                else:
                    base_path = f"{alfworld_data}/json_2.1.1/{self.split}"

                print(
                    f"[ALFWorldGame] First-time scanning: {base_path} (PID: {os.getpid()})..."
                )

                # Scan for all matching task types
                # ALFWorld structure: task_type-xxx/trial_xxx/game.tw-pddl
                # NOTE: Some trial directories are incomplete (missing game files), so we filter them
                all_paths = []
                for task_type in self.task_types:
                    # Pattern to find trial directories
                    pattern = os.path.join(base_path, f"{task_type}-*", "trial_*")
                    matching = glob.glob(pattern)

                    # Only keep directories that actually contain a game file
                    for trial_path in matching:
                        game_file = os.path.join(trial_path, "game.tw-pddl")
                        if os.path.exists(game_file):
                            all_paths.append(trial_path)
                        else:
                            # Check for alternative game file formats
                            alt_files = glob.glob(os.path.join(trial_path, "game.*"))
                            if alt_files:
                                all_paths.append(trial_path)

                # Sort for consistency
                all_paths.sort()
                print(
                    f"[ALFWorldGame] Found {len(all_paths)} valid games (with game files)."
                )

                # Apply num_games limit if specified
                if self.num_games > 0 and len(all_paths) > self.num_games:
                    # Use a fixed seed so all instances get the same subset
                    rng = random.Random(42)
                    all_paths = rng.sample(all_paths, self.num_games)
                    all_paths.sort()

                ALFWorldGame._cached_game_paths[cache_key] = all_paths
                print(
                    f"[ALFWorldGame] Scan complete. Found {len(all_paths)} games. Cached for reuse."
                )
            else:
                # Subsequent instances: reuse cached paths (no disk scan!)
                pass

        return ALFWorldGame._cached_game_paths[cache_key]

    def _init_env(self):
        """Initialize ALFWorld environment.

        OPTIMIZED: Uses direct TextWorld loading instead of ALFWorld's heavy init_env.
        This avoids redundant disk scanning across multiple instances.
        """
        if self._initialized:
            return

        # Mark as initialized (engine ready, will load specific game on reset)
        self._initialized = True
        print(f"[ALFWorldGame] Engine ready (PID: {os.getpid()}, id: {id(self)})")

    def reset(
        self, task_type: Optional[str] = None, seed: Optional[int] = None, **kwargs
    ) -> str:
        """Reset the game to a new episode.

        OPTIMIZED: Uses cached game paths and direct TextWorld loading.
        This avoids redundant disk scanning - only the first instance scans.

        Args:
            task_type: Specific task type to use (None = random from task_types)
            seed: Random seed for reproducibility
            **kwargs: Additional arguments (ignored)

        Returns:
            Initial observation string
        """
        # Initialize engine if needed
        self._init_env()

        # Set seed if provided
        if seed is not None:
            random.seed(seed)

        # Get cached game paths (only first instance scans disk)
        # Paths are now trial directories: .../task_type-xxx/trial_xxx/
        game_paths = self._get_cached_game_paths()

        if not game_paths:
            raise RuntimeError(
                f"No games found for split='{self.split}', task_types={self.task_types}"
            )

        # Select a random game from the cached paths (trial directory)
        selected_path = random.choice(game_paths)

        # Game file is directly in the trial directory
        game_file = os.path.join(selected_path, "game.tw-pddl")

        # Fallback to other possible game file names
        if not os.path.exists(game_file):
            game_file = os.path.join(selected_path, "game.z8")
        if not os.path.exists(game_file):
            # Try to find any game file
            possible_files = glob.glob(os.path.join(selected_path, "game.*"))
            if possible_files:
                game_file = possible_files[0]
            else:
                raise RuntimeError(f"No game file found in {selected_path}")

        # Close previous TextWorld environment if exists
        if self._tw_env is not None:
            try:
                self._tw_env.close()
            except Exception:
                pass

        # Create TextWorld environment with direct file loading (no disk scan!)
        infos = textworld.EnvInfos(
            feedback=True,
            inventory=True,
            description=True,
            admissible_commands=True,
            score=True,
            max_score=True,
            won=True,
            lost=True,
            extras=["walkthrough", "expert_plan"],
        )

        # Direct loading: bypass ALFWorld's init_env which scans all 8810 games
        # Thread safety: parallelism comes from sharding (multiple server processes)
        self._tw_env = textworld.start(
            game_file, infos, wrappers=[Filter, AlfredDemangler()]
        )
        self._env = self._tw_env  # For compatibility with step()

        # Reset the environment
        game_state, info = self._tw_env.reset()

        # Extract observation
        if hasattr(game_state, "feedback"):
            self._current_obs = game_state.feedback
        elif isinstance(game_state, str):
            self._current_obs = game_state
        else:
            self._current_obs = str(game_state)

        self._current_info = info

        # Parse task description from observation
        self._task_desc = self._extract_task_description(self._current_obs)

        # Get admissible commands
        if hasattr(game_state, "admissible_commands"):
            self._admissible_commands = game_state.admissible_commands or []
        else:
            self._admissible_commands = info.get("admissible_commands", [])

        # Reset state
        self._step_count = 0
        self._done = False

        return self._format_observation(self._current_obs)

    def _extract_task_description(self, obs: str) -> str:
        """Extract task description from observation."""
        # ALFWorld observations typically start with the task
        lines = obs.strip().split("\n")
        for line in lines:
            if line.startswith("Your task is to"):
                return line
        return "Complete the household task."

    def _format_observation(self, obs: str) -> str:
        """Format observation for LLM."""
        formatted = f"=== Current State ===\n{obs}\n"

        if self._admissible_commands:
            formatted += "\n=== Available Actions ===\n"
            formatted += "\n".join(f"- {cmd}" for cmd in self._admissible_commands)

        return formatted

    def step(self, action: str) -> StepResult:
        """Execute an action in the environment.

        Args:
            action: Text command to execute (e.g., "go to desk 1", "take book 1")

        Returns:
            StepResult with observation, reward, done flag, and info
        """
        if self._done:
            return StepResult(
                observation="Episode already finished.",
                reward=0.0,
                done=True,
                info={"error": "episode_finished"},
            )

        self._step_count += 1

        # Parse action from LLM output
        parsed_action = self._parse_action(action)

        # Execute action in TextWorld environment
        # TextWorld returns: game_state, reward, done, info
        # Thread safety: parallelism comes from sharding (multiple server processes)
        game_state, reward, done, info = self._tw_env.step(parsed_action)

        # Extract observation from game_state
        if hasattr(game_state, "feedback"):
            obs = game_state.feedback
        elif isinstance(game_state, str):
            obs = game_state
        else:
            obs = str(game_state)

        # Update state
        self._current_obs = obs

        # Get admissible commands
        # NOTE: Depending on wrappers / TextWorld versions, admissible commands may be
        # stored either on game_state or inside info dict. If we drop this, the agent
        # becomes "blind" and often collapses to repetitive actions (e.g., always 'look').
        if hasattr(game_state, "admissible_commands"):
            self._admissible_commands = game_state.admissible_commands or []
        else:
            self._admissible_commands = info.get("admissible_commands", [])

        # Check for timeout
        if self._step_count >= self.max_steps and not done:
            done = True
            reward = self.REWARD_FAILURE
            obs = f"TIMEOUT: Maximum steps ({self.max_steps}) reached.\n\n{obs}"

        # Adjust rewards
        if done and reward > 0:
            # Task completed successfully
            final_reward = self.REWARD_SUCCESS
            obs = f"SUCCESS! Task completed!\n\n{obs}"
        elif done:
            # Task failed
            final_reward = self.REWARD_FAILURE
        else:
            # Step penalty
            final_reward = self.REWARD_STEP
            if "Nothing happens" in obs or "invalid" in obs.lower():
                final_reward = self.REWARD_INVALID_ACTION

        self._done = done

        return StepResult(
            observation=self._format_observation(obs),
            reward=final_reward,
            done=done,
            info={
                # Note: Don't include "step" here as gym_environment_interaction.py
                # already passes it explicitly to observation_template.format()
                "raw_reward": float(reward),
                "action_taken": parsed_action,
                "task": self._task_desc,
            },
        )

    def _parse_action(self, raw_action: str) -> str:
        """Parse action from LLM output.

        Supports formats:
        - <action>go to desk 1</action>
        - Direct command: go to desk 1
        """
        # Try to extract from <action> tags
        match = re.search(
            r"<action>\s*(.*?)\s*</action>", raw_action, re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()

        # Otherwise, use the last line as action
        lines = raw_action.strip().split("\n")
        return lines[-1].strip()

    def get_system_prompt(self) -> str:
        """Return the system prompt for ALFWorld."""
        return (
            "You are an AI assistant playing ALFWorld, a text-based household environment.\n"
            "Your goal is to complete household tasks by interacting with objects.\n\n"
            "IMPORTANT: You MUST respond in the following format:\n"
            "1. First, think about your plan in <thinking></thinking> tags\n"
            "2. Then, output your action in <action></action> tags\n\n"
            "Common actions:\n"
            "- go to [receptacle]: Move to a location (e.g., 'go to desk 1')\n"
            "- take [object] from [receptacle]: Pick up an object\n"
            "- put [object] in/on [receptacle]: Place an object\n"
            "- open [receptacle]: Open a container\n"
            "- close [receptacle]: Close a container\n"
            "- use [object]: Use a device (lamp, microwave, etc.)\n"
            "- examine [object]: Look at an object closely\n"
            "- inventory: Check what you're holding\n"
            "- look: Look around the room\n\n"
            "Example response:\n"
            "<thinking>I need to find the book first. Let me check the desk.</thinking>\n"
            "<action>go to desk 1</action>"
        )

    def get_initial_user_message(self) -> str:
        """Return the initial user message for ALFWorld."""
        return (
            f"Task: {self._task_desc}\n\n"
            "Explore the environment and complete the task. "
            "What would you like to do first?"
        )

    def get_state(self) -> Dict[str, Any]:
        """Return current game state."""
        return {
            "observation": self._current_obs,
            "task": self._task_desc,
            "step_count": self._step_count,
            "max_steps": self.max_steps,
            "done": self._done,
            "admissible_commands": self._admissible_commands[:10],
        }

    # =========================================================================
    # Data Generation Methods (for training)
    # =========================================================================

    def generate_initial_state(self) -> Dict[str, Any]:
        """Generate random initial state for training data.

        Returns a dict with task configuration that reset() will use.
        """
        # Randomly select a task type
        task_type = random.choice(self.task_types)

        return {
            "task_type": task_type,
            "seed": random.randint(0, 1000000),
        }

    def get_user_message_with_state(
        self, task_type: Optional[str] = None, **kwargs
    ) -> str:
        """Generate user message with rendered initial state for prompt."""
        # Reset to get actual observatio
        self.reset(task_type=task_type, **kwargs)

        return (
            f"Task: {self._task_desc}\n\n"
            f"{self._format_observation(self._current_obs)}\n\n"
            "What would you like to do?"
        )

    def get_interaction_name(self) -> str:
        """Return interaction name for ALFWorld."""
        return "alfworld"
