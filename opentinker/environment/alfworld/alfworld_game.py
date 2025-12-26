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

import os
import random
import re
from typing import Any, Dict, List, Optional
import logging

from opentinker.environment.base_game import AbstractGame, StepResult

# ALFWorld imports - install with: pip install alfworld
try:
    import alfworld.agents.environment as alfworld_env
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
    DEFAULT_MAX_STEPS = 50
    
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
    _shared_envs: dict = {}
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        task_types: Optional[List[str]] = None,
        split: str = "train",
        num_games: int = -1,
    ):
        """Initialize ALFWorld game.
        
        Args:
            config_path: Path to ALFWorld config file (base_config.yaml)
            max_steps: Maximum steps per episode
            task_types: Task types to sample from (None = all types)
            split: Dataset split ("train", "eval_in_distribution", "eval_out_of_distribution")
            num_games: Number of games to load (-1 = all games, e.g. 64 for faster loading)
        """
        if not ALFWORLD_AVAILABLE:
            raise ImportError(
                "alfworld package not installed. "
                "Install with: pip install alfworld"
            )
        
        self.config_path = config_path
        self.max_steps = max_steps
        self.task_types = task_types or self.ALL_TASK_TYPES
        self.split = split
        self.num_games = num_games
        
        # Load ALFWorld config
        self._load_alfworld_config()
        
        # Game state (instance-specific, NOT shared)
        self._env = None  # Will be set from shared cache in reset()
        self._current_obs = None
        self._current_info = None
        self._step_count = 0
        self._task_desc = ""
        self._admissible_commands = []
        self._done = False
    
    def _load_alfworld_config(self):
        """Load ALFWorld configuration."""
        import alfworld
        
        if self.config_path:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            # ALFWorld data directory (need to run 'alfworld-download' first)
            alfworld_data = os.path.expandvars(os.environ.get('ALFWORLD_DATA', '$HOME/.cache/alfworld'))
            
            # Use default config with all required fields
            self._config = {
                'general': {
                    'training_method': 'dqn',  # or 'dagger'
                },
                'env': {
                    'type': 'AlfredTWEnv',
                    'regen_game_files': False,
                    'domain_randomization': False,
                    'goal_desc_human_anns_prob': 0,  # Required by AlfredTWEnv
                    'task_types': [1, 2, 3, 4, 5, 6],  # All 6 task types
                    'expert_type': 'handcoded',  # or 'planner'
                    'registration': {
                        'batch_size': 1,
                    },
                    'split': self.split,
                },
                'dataset': {
                    'data_path': f'{alfworld_data}/json_2.1.1/train',
                    'eval_id_data_path': f'{alfworld_data}/json_2.1.1/valid_seen',
                    'eval_ood_data_path': f'{alfworld_data}/json_2.1.1/valid_unseen',
                    'num_train_games': self.num_games,  # -1 means use all games
                    'num_eval_games': self.num_games,
                },
                'logic': {
                    'domain': f'{alfworld_data}/logic/alfred.pddl',
                    'grammar': f'{alfworld_data}/logic/alfred.twl2',
                },
                'rl': {
                    'training': {
                        'max_nb_steps_per_episode': self.max_steps,
                    },
                },
            }
    
    def _init_env(self):
        """Initialize ALFWorld environment using process-level cache.
        
        The environment is shared within the same process to avoid repeated
        loading of game files (which takes ~4 seconds for 8810 games).
        Each Ray worker process gets its own cached environment.
        """
        cache_key = (self.config_path, self.split, self.num_games)
        
        if cache_key not in ALFWorldGame._shared_envs:
            print(f"[ALFWorldGame] Initializing environment for split='{self.split}', num_games={self.num_games}...")
            env_type = self._config['env']['type']
            env_class = alfworld_env.get_environment(env_type)
            env = env_class(self._config)
            ALFWorldGame._shared_envs[cache_key] = env.init_env(batch_size=1)
            print(f"[ALFWorldGame] Environment cached (PID: {os.getpid()})")
        
        self._env = ALFWorldGame._shared_envs[cache_key]
    
    def reset(self, task_type: Optional[str] = None, seed: Optional[int] = None, **kwargs) -> str:
        """Reset the game to a new episode.
        
        Args:
            task_type: Specific task type to use (None = random from task_types)
            seed: Random seed for reproducibility
            **kwargs: Additional arguments (ignored)
        
        Returns:
            Initial observation string
        """
        # Initialize environment if needed
        self._init_env()
        
        # Set seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Reset environment
        obs, info = self._env.reset()
        
        # Extract observation and task
        # Handle obs being list, tuple, or string
        if isinstance(obs, (list, tuple)):
            self._current_obs = obs[0]
        else:
            self._current_obs = obs
        self._current_info = info
        
        # Parse task description from observation
        self._task_desc = self._extract_task_description(self._current_obs)
        
        # Get admissible commands
        self._admissible_commands = info.get('admissible_commands', [[]])[0]
        
        # Reset state
        self._step_count = 0
        self._done = False
        
        return self._format_observation(self._current_obs)
    
    def _extract_task_description(self, obs: str) -> str:
        """Extract task description from observation."""
        # ALFWorld observations typically start with the task
        lines = obs.strip().split('\n')
        for line in lines:
            if line.startswith("Your task is to"):
                return line
        return "Complete the household task."
    
    def _format_observation(self, obs: str) -> str:
        """Format observation for LLM."""
        formatted = f"=== Current State ===\n{obs}\n"
        
        if self._admissible_commands:
            formatted += "\n=== Available Actions ===\n"
            formatted += "\n".join(f"- {cmd}" for cmd in self._admissible_commands[:20])
            if len(self._admissible_commands) > 20:
                formatted += f"\n... and {len(self._admissible_commands) - 20} more"
        
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
                info={"error": "episode_finished"}
            )
        
        self._step_count += 1
        
        # Parse action from LLM output
        parsed_action = self._parse_action(action)
        
        # Execute action in environment
        obs, reward, done, info = self._env.step([parsed_action])
        
        # Extract from batch (handle list or tuple)
        obs = obs[0] if isinstance(obs, (list, tuple)) else obs
        reward = reward[0] if isinstance(reward, (list, tuple)) else reward
        done = done[0] if isinstance(done, (list, tuple)) else done
        
        # Update state
        self._current_obs = obs
        self._current_info = info
        self._admissible_commands = info.get('admissible_commands', [[]])[0]
        
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
            }
        )
    
    def _parse_action(self, raw_action: str) -> str:
        """Parse action from LLM output.
        
        Supports formats:
        - <action>go to desk 1</action>
        - Direct command: go to desk 1
        """
        # Try to extract from <action> tags
        match = re.search(r'<action>\s*(.*?)\s*</action>', raw_action, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Otherwise, use the last line as action
        lines = raw_action.strip().split('\n')
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
    
    def get_user_message_with_state(self, task_type: Optional[str] = None, **kwargs) -> str:
        """Generate user message with rendered initial state for prompt."""
        # Reset to get actual observation
        self.reset(task_type=task_type, **kwargs)
        
        return (
            f"Task: {self._task_desc}\n\n"
            f"{self._format_observation(self._current_obs)}\n\n"
            "What would you like to do?"
        )
    
    def get_interaction_name(self) -> str:
        """Return interaction name for ALFWorld."""
        return "alfworld"

