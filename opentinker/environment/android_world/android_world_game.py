import os
import sys
import random
import re
import threading
import json
import ast
from typing import Any, Dict, List, Optional
import logging

from opentinker.environment.base_game import AbstractGame, StepResult
from opentinker.environment.android_world import prompts

logger = logging.getLogger(__name__)

# Ensure android_world is in path if it's in the current directory
if os.path.exists("android_world"):
    sys.path.append(os.path.abspath("android_world"))

# Android emulator connection defaults (overridable via env vars)
os.environ.setdefault("ADB_PATH", "adb")
os.environ.setdefault("ANDROID_CONSOLE_PORT", "5556")
os.environ.setdefault("ANDROID_GRPC_PORT", "8554")

# AndroidWorld imports
try:
    import android_world
    from android_world import registry
    from android_world.env import env_launcher
    from android_world.env import interface
    from android_world.env import json_action
    from android_world.env import representation_utils
    from android_world.agents import m3a_utils
    ANDROID_WORLD_AVAILABLE = True
except ImportError:
    ANDROID_WORLD_AVAILABLE = False
    logger.warning("android_world not installed or not found.")


class AndroidWorldGame(AbstractGame):
    """AndroidWorld environment game implementation.

    This implementation wraps the AndroidWorld environment for LLM RL training.
    It adopts the T3A agent's prompting and observation style.
    
    NOTE: All game instances share ONE emulator. A global lock serializes all
    reset() and step() operations to prevent concurrent access conflicts.
    """

    # Reward constants
    REWARD_SUCCESS = 10.0
    REWARD_FAILURE = -1.0
    REWARD_STEP = -0.01
    REWARD_INVALID_ACTION = -0.1

    # Limits
    DEFAULT_MAX_STEPS = 20

    # Task types in AndroidWorld
    ALL_TASK_TYPES = [
        "ContactsAddContact",
    ]
    
    _shared_envs: dict = {}
    _use_shared_env: bool = False

    _cached_game_paths: Dict[str, List[str]] = {}
    _cache_lock = threading.Lock()
    
    # NOTE: No emulator lock needed - with PPO (rollout_n=1), operations are
    # naturally serialized. The env server's asyncio lock handles any remaining
    # concurrency at the HTTP level.

    def __init__(
        self,
        config_path: Optional[str] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        task_types: Optional[List[str]] = None,
        split: str = "train",
        num_games: int = -1,
        use_shared_env: bool = False,
        emulator_console_port: Optional[int] = None,
        emulator_grpc_port: Optional[int] = None,
    ):
        """Initialize AndroidWorld game.
        
        Args:
            emulator_console_port: Emulator console port (default: from ANDROID_CONSOLE_PORT env or 5556)
            emulator_grpc_port: Emulator gRPC port (default: from ANDROID_GRPC_PORT env or 8554)
        """
        self.config_path = config_path
        self.max_steps = max_steps
        self.task_types = task_types or self.ALL_TASK_TYPES
        self.split = split
        self.num_games = num_games
        self._use_shared_env = use_shared_env
        
        # Emulator ports (can be set per-shard for multi-emulator support)
        self._emulator_console_port = emulator_console_port
        self._emulator_grpc_port = emulator_grpc_port

        # Game state
        self._env: Optional[interface.AsyncEnv] = None
        self._task = None
        self._current_obs = None
        self._step_count = 0
        self._task_desc = ""
        self._done = False
        self._initialized = False
        self._history = []  # List of step summaries for T3A prompt

    def _init_env(self):
        """Initialize AndroidWorld environment."""
        if self._initialized:
            return

        if ANDROID_WORLD_AVAILABLE:
            adb_path = os.environ.get("ADB_PATH", None)
            # Use instance port if set, otherwise fall back to env var
            console_port = self._emulator_console_port or int(os.environ.get("ANDROID_CONSOLE_PORT", 5556))
            grpc_port = self._emulator_grpc_port or int(os.environ.get("ANDROID_GRPC_PORT", 8554))

            if not adb_path:
                try:
                    from android_world.env import android_world_controller
                    adb_path = android_world_controller.DEFAULT_ADB_PATH
                except ImportError:
                     adb_path = "adb"

            logger.info(f"Initializing AndroidEnv (console_port={console_port}, grpc_port={grpc_port})")
            try:
                self._env = env_launcher.load_and_setup_env(
                    console_port=console_port,
                    grpc_port=grpc_port,
                    adb_path=adb_path,
                    emulator_setup=False,
                    freeze_datetime=True
                )
            except Exception as e:
                logger.error(f"Failed to initialize AndroidWorld env: {e}")
                logger.warning("Falling back to MOCK mode.")
                self._env = None
        else:
             logger.info("Running in MOCK mode (android_world not installed).")
        
        self._initialized = True

    def reset(
        self,
        task_type: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> str:
        """Reset the game to a new episode."""
        self._init_env()

        if seed is not None:
            random.seed(seed)
            
        self._step_count = 0
        self._done = False
        self._history = [] # Reset history

        if ANDROID_WORLD_AVAILABLE and self._env:
            task_name = task_type
            if not task_name:
                task_name = random.choice(self.task_types)
            
            try:
                task_registry = registry.TaskRegistry()
                all_tasks = task_registry.get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY)
                
                if task_name not in all_tasks:
                    logger.warning(f"Task {task_name} not found in registry. Using default.")
                    task_class = all_tasks.get("ContactsAddContact")
                else:
                    task_class = all_tasks[task_name]

                params = task_class.generate_random_params()
                if seed is not None:
                    params['seed'] = seed
                
                self._task = task_class(params)
                logger.info(f"Resetting task: {self._task.name}")
                
                self._task.initialize_task(self._env)
                self._env.hide_automation_ui()
                state = self._env.reset(go_home=True)
                
                self._task_desc = self._task.goal
                self._process_state(state)
                
            except Exception as e:
                logger.error(f"Failed to reset AndroidWorld task {task_name}: {e}", exc_info=True)
                self._set_mock_state(task_name)
        else:
            task_name = task_type or random.choice(self.task_types)
            self._set_mock_state(task_name)

        return self._format_observation(include_instructions=True)

    def _set_mock_state(self, task_name: str):
        """Set mock state for testing without the library."""
        self._task_desc = f"Mock Task: {task_name}. Interact with the device."
        self._current_obs = "UI element 0: Button 'Settings' at [100, 200]\nUI element 1: Icon 'Chrome' at [300, 200]"
        self._done = False

    def _validate_ui_element(self, ui_element: Any, screen_size: tuple[int, int]) -> bool:
        """Filter out invalid UI elements (invisible or invalid bbox)."""
        screen_width, screen_height = screen_size
        if not ui_element.is_visible:
            return False
        if ui_element.bbox_pixels:
            x_min = ui_element.bbox_pixels.x_min
            x_max = ui_element.bbox_pixels.x_max
            y_min = ui_element.bbox_pixels.y_min
            y_max = ui_element.bbox_pixels.y_max
            if (x_min >= x_max or x_min >= screen_width or x_max <= 0 or
                y_min >= y_max or y_min >= screen_height or y_max <= 0):
                return False
        return True

    def _process_state(self, state: Any):
        """Process AndroidWorld State into text representation."""
        if hasattr(state, "ui_elements") and self._env:
            logical_screen_size = self._env.logical_screen_size
            elements_text = ""
            valid_count = 0
            for index, ui_element in enumerate(state.ui_elements):
                if self._validate_ui_element(ui_element, logical_screen_size):
                    # Use str(ui_element) which typically provides a good summary
                    elements_text += f"UI element {index}: {str(ui_element)}\n"
                    valid_count += 1
            
            logger.debug(f"_process_state: {len(state.ui_elements)} total elements, {valid_count} valid")
            self._current_obs = elements_text if elements_text else "No visible interactable elements."
        else:
            self._current_obs = "Screen updated (No UI elements info)"

    def _format_observation(self, obs: Optional[str] = None, include_instructions: bool = False) -> str:
        """Format observation using prompt template for agent loop.
        
        Args:
            obs: Optional observation string. If None, uses self._current_obs.
            include_instructions: If True, include PROMPT_PREFIX and GUIDANCE (for initial prompt).
                                  If False, use simplified template (for subsequent turns).
        """
        if obs is None:
            obs = self._current_obs
        
        if include_instructions or self._step_count == 0:
            # Full template with PROMPT_PREFIX and GUIDANCE for initial prompt
            history_str = '\n'.join(self._history) if self._history else 'You just started, no action has been performed yet.'
        
            # We put everything in the user message/observation for the agent
            return prompts.INITIAL_ACTION_SELECTION_PROMPT_TEMPLATE.format(
                goal=self._task_desc,
                history=history_str,
                ui_elements_description=obs,
                additional_guidelines="" 
            )
        else:
            # Simplified template: only goal and UI elements description for subsequent turns
            return prompts.ACTION_SELECTION_PROMPT_TEMPLATE.format(
                goal=self._task_desc,
                ui_elements_description=obs,
            )

    def step(self, action: str) -> StepResult:
        """Execute an action in the environment."""
        if self._done:
            return StepResult(observation="Episode finished.", reward=0.0, done=True, info={})

        self._step_count += 1
        
        # Parse Reason/Action
        reason, json_str = self._parse_reason_action(action)
        
        reward = self.REWARD_STEP
        step_summary = "Action failed."
        
        if ANDROID_WORLD_AVAILABLE and self._env and self._task:
            if json_str:
                try:
                    env_action = self._parse_json_to_env_action(json_str)
                    
                    if env_action:
                        # Handle special 'status' action for completion
                        if env_action.action_type == 'status':
                            if env_action.goal_status == 'complete':
                                reward = self.REWARD_SUCCESS
                                self._done = True
                                if self._task.is_successful(self._env) > 0.0:
                                    step_summary = "Agent finished. Task Successful."
                                else:
                                    reward = self.REWARD_FAILURE
                                    step_summary = "Agent finished. Task Failed (Condition not met)."
                            else:
                                self._done = True
                                reward = self.REWARD_FAILURE
                                step_summary = "Agent declared task infeasible."
                        
                        elif env_action.action_type == 'answer':
                            step_summary = f"Agent answered: {env_action.text}"
                        
                        else:
                            # Execute physical action
                            self._env.execute_action(env_action)
                            state = self._env.get_state(wait_to_stabilize=True)
                            self._process_state(state)
                            reason_str = reason[:50] if reason else ""
                            step_summary = f"Executed {env_action.action_type}. {reason_str}..."
                    else:
                        reward = self.REWARD_INVALID_ACTION
                        step_summary = "Invalid JSON action format."
                except Exception as e:
                    logger.error(f"Error executing action: {e}", exc_info=True)
                    reward = self.REWARD_INVALID_ACTION
                    step_summary = f"Execution error: {str(e)}"
            else:
                reward = self.REWARD_INVALID_ACTION
                step_summary = "Could not parse Action JSON."
        else:
            # Mock mode
            step_summary = f"Mock execute: {action}"
            self._current_obs = "Screen updated."

        # Update History
        self._history.append(f"Step {self._step_count}: {step_summary}")

        if self._step_count >= self.max_steps and not self._done:
            self._done = True
            reward = self.REWARD_FAILURE
            step_summary += " (Timeout)"

        return StepResult(
            observation=self._format_observation(),
            reward=reward,
            done=self._done,
            info={
                "raw_reward": float(reward),
                "action_taken": action,
                "task": self._task_desc,
            },
        )

    def _parse_reason_action(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """Parse 'Reason: ... Action: {...}' output."""
        reason_match = re.search(r'Reason:(.*)Action:', text, flags=re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else None
        
        action_match = re.search(r'Action:(.*)', text, flags=re.DOTALL)
        action_json = action_match.group(1).strip() if action_match else None
        
        # If strict format fails, try to just find the last JSON blob
        if not action_json:
             json_candidates = re.findall(r'\{.*?\}', text, re.DOTALL)
             if json_candidates:
                 action_json = json_candidates[-1]
        
        return reason, action_json

    def _parse_json_to_env_action(self, json_str: str) -> Any:
        """Convert JSON string to AndroidWorld JSONAction."""
        # Clean up json_str - extract just the first dict/JSON object
        # The string may contain extra text like "Your Answer:" after the JSON
        
        # First, try to find a complete JSON object with double quotes
        json_match = re.search(r'\{[^{}]*"action_type"[^{}]*\}', json_str)
        if json_match:
            try:
                action_dict = json.loads(json_match.group())
                if isinstance(action_dict, dict):
                    return json_action.JSONAction(**action_dict)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Try to find a complete dict with single quotes (Python style)
        dict_match = re.search(r"\{[^{}]*'action_type'[^{}]*\}", json_str)
        if dict_match:
            try:
                action_dict = ast.literal_eval(dict_match.group())
                if isinstance(action_dict, dict):
                    return json_action.JSONAction(**action_dict)
            except (ValueError, SyntaxError, TypeError):
                pass
        
        # Fallback: try parsing the whole string
        try:
            action_dict = json.loads(json_str)
            if isinstance(action_dict, dict):
                return json_action.JSONAction(**action_dict)
        except json.JSONDecodeError:
            try:
                action_dict = ast.literal_eval(json_str)
                if isinstance(action_dict, dict):
                    return json_action.JSONAction(**action_dict)
            except:
                pass
        
        return None

    def get_system_prompt(self) -> str:
        """Return the system prompt."""
        # The T3A prompt template in get_user_message_with_state contains the instructions.
        # We can return a generic system prompt here.
        return "You are a helpful AI assistant capable of operating an Android device."

    def get_initial_user_message(self) -> str:
        """Return the initial user message."""
        # Used if environment is already reset
        return self._format_observation(include_instructions=True)

    def get_state(self) -> Dict[str, Any]:
        """Return the current game state."""
        return {
            "observation": self._current_obs,
            "task": self._task_desc,
            "step_count": self._step_count,
            "max_steps": self.max_steps,
            "done": self._done,
        }

    def generate_initial_state(self) -> Dict[str, Any]:
        """Generate a random initial state for training data."""
        task_type = random.choice(self.task_types)
        return {
            "task_type": task_type,
            "seed": random.randint(0, 1000000),
        }

    def get_user_message_with_state(
        self, task_type: Optional[str] = None, **kwargs
    ) -> str:
        """Generate a user message with the rendered initial state for the prompt.
        
        NOTE: This is called by Client (DataLoader) to generate the initial prompt.
        We do NOT call reset() here because:
        1. Server will call reset() again via start_interaction() -> /reset
        2. Both would operate the same Android emulator, causing conflicts
        
        Instead, we return a properly formatted prompt with placeholder UI elements.
        The real reset() and UI elements come from Server's /reset call.
        """
        task_name = task_type or "ContactsAddContact"
        seed = kwargs.get("seed")
        
        # Generate task goal based on task type (same logic as in reset)
        # This ensures consistency between client prompt and server reset
        if ANDROID_WORLD_AVAILABLE:
            try:
                task_registry = registry.TaskRegistry()
                all_tasks = task_registry.get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY)
                if task_name in all_tasks:
                    task_class = all_tasks[task_name]
                    params = task_class.generate_random_params()
                    if seed is not None:
                        params['seed'] = seed
                    temp_task = task_class(params)
                    goal = temp_task.goal
                else:
                    goal = f"Complete the {task_name} task."
            except Exception:
                goal = f"Complete the {task_name} task."
        else:
            goal = f"Complete the {task_name} task."
        
        # Return formatted prompt with real Home Screen UI elements
        # All AndroidWorld tasks start from Home Screen (go_home=True in reset)
        # This UI is consistent across all tasks, so we can hardcode it
        home_screen_ui = (
            "UI element 0: UIElement(text=None, content_description=None, class_name='android.widget.ScrollView', is_clickable=False, is_scrollable=True, is_visible=True, package_name='com.google.android.apps.nexuslauncher')\n"
            "UI element 1: UIElement(text=None, content_description='Home', class_name='android.view.View', is_clickable=False, is_visible=True, package_name='com.google.android.apps.nexuslauncher')\n"
            "UI element 2: UIElement(text='Phone', content_description='Phone', class_name='android.widget.TextView', is_clickable=True, is_visible=True, package_name='com.google.android.apps.nexuslauncher')\n"
            "UI element 3: UIElement(text='Messages', content_description='Messages', class_name='android.widget.TextView', is_clickable=True, is_visible=True, package_name='com.google.android.apps.nexuslauncher')\n"
            "UI element 4: UIElement(text='Chrome', content_description='Chrome', class_name='android.widget.TextView', is_clickable=True, is_visible=True, package_name='com.google.android.apps.nexuslauncher')\n"
            "UI element 5: UIElement(text='Gmail', content_description='Gmail', class_name='android.widget.TextView', is_clickable=True, is_visible=True, package_name='com.google.android.apps.nexuslauncher')\n"
            "UI element 6: UIElement(text=None, content_description='Search', class_name='android.widget.FrameLayout', is_clickable=True, is_visible=True, package_name='com.google.android.apps.nexuslauncher')\n"
            "UI element 7: UIElement(text='Photos', content_description='Photos', class_name='android.widget.TextView', is_clickable=True, is_visible=True, package_name='com.google.android.apps.nexuslauncher')\n"
            "UI element 8: UIElement(text='YouTube', content_description='YouTube', class_name='android.widget.TextView', is_clickable=True, is_visible=True, package_name='com.google.android.apps.nexuslauncher')\n"
            "UI element 9: UIElement(text=None, content_description='Google app', class_name='android.widget.ImageView', is_clickable=True, is_visible=True, package_name='com.google.android.apps.nexuslauncher')\n"
        )
        history = "You just started, no action has been performed yet."
        
        return prompts.INITIAL_ACTION_SELECTION_PROMPT_TEMPLATE.format(
            goal=goal,
            history=history,
            ui_elements_description=home_screen_ui,
            additional_guidelines=""
        )

    def get_interaction_name(self) -> str:
        """Return the interaction name."""
        return "android_world"
