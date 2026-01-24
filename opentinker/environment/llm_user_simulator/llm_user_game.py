#!/usr/bin/env python3
"""LLM User Simulator Game Implementation.

This module provides an environment where an LLM acts as a user simulator,
enabling training of conversational agents through self-play or cross-play.

Example:
    from llm_user_game import LLMUserGame
    
    game = LLMUserGame(
        simulator_model="gpt-4o-mini",
        task_prompt="You are a customer trying to book a flight.",
    )
    obs = game.reset()
    result = game.step("Hello! How can I help you today?")
"""

import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from opentinker.environment.base_game import AbstractGame, StepResult

# Try to import LLM clients
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "agent" or "user"
    content: str


class LLMUserGame(AbstractGame):
    """LLM-based user simulator environment.
    
    The agent (being trained) plays the role of an assistant/agent,
    while an LLM simulates the user providing requests and feedback.
    
    Attributes:
        simulator_model: Model name for the user simulator (e.g., "gpt-4o-mini")
        task_prompt: System prompt defining the user's persona and task
        max_turns: Maximum conversation turns before episode ends
        success_keywords: Keywords that indicate task success
    """
    
    # Reward constants
    REWARD_SUCCESS = 10.0
    REWARD_FAILURE = -1.0
    REWARD_STEP = -0.01
    REWARD_USER_SATISFIED = 5.0
    
    # Default max turns
    DEFAULT_MAX_TURNS = 10
    
    # LLM Judge prompt
    JUDGE_PROMPT = """You are an expert evaluator assessing conversational AI quality.

Evaluate the following conversation between an AI assistant and a user.

## Evaluation Criteria (score 1-10 each):

1. **Helpfulness**: Did the assistant understand and address the user's needs?
2. **Clarity**: Were the responses clear and easy to understand?
3. **Problem Resolution**: Was the user's issue/request successfully resolved?
4. **Professionalism**: Was the tone appropriate and professional?
5. **Efficiency**: Did the assistant resolve the issue without unnecessary back-and-forth?

## Conversation:
{conversation}

## Your Evaluation:
Provide scores and a brief explanation in this exact JSON format:
```json
{
    "helpfulness": <1-10>,
    "clarity": <1-10>,
    "problem_resolution": <1-10>,
    "professionalism": <1-10>,
    "efficiency": <1-10>,
    "overall_score": <1-10>,
    "success": <true/false>,
    "explanation": "<brief explanation>"
}
```
"""
    
    # Default task prompts for various scenarios
    TASK_PROMPTS = {
        "customer_service": """You are a customer contacting customer service.
You have a specific problem that needs to be resolved.
Be realistic - ask clarifying questions, express frustration if not helped properly.
When your issue is resolved satisfactorily, say "Thank you, that resolves my issue."
If the agent is unhelpful after several attempts, say "This is not helpful, goodbye."
""",
        "booking_assistant": """You are a user trying to book a reservation.
You have specific preferences (date, time, number of people, etc.).
Ask questions about availability and options.
When you successfully complete a booking, say "Great, the booking is confirmed."
If unable to book, say "I'll try elsewhere, thanks."
""",
        "tech_support": """You are a user with a technical problem.
Describe your issue and provide details when asked.
If the solution works, say "That fixed it, thank you!"
If multiple attempts fail, say "This still doesn't work."
""",
        "information_seeking": """You are a user looking for specific information.
Ask questions to get the information you need.
When you get satisfactory answers, say "That's exactly what I needed, thanks!"
If answers are unclear or wrong, say "That's not what I was looking for."
""",
    }
    
    def __init__(
        self,
        simulator_model: str = "gpt-4o-mini",
        simulator_api_key: Optional[str] = None,
        simulator_base_url: Optional[str] = None,
        task_prompt: Optional[str] = None,
        task_type: str = "customer_service",
        max_turns: int = DEFAULT_MAX_TURNS,
        success_keywords: Optional[List[str]] = None,
        failure_keywords: Optional[List[str]] = None,
        temperature: float = 0.7,
        seed: Optional[int] = None,
        use_llm_judge: bool = True,
        judge_model: Optional[str] = None,
    ):
        """Initialize LLM User Simulator.
        
        Args:
            simulator_model: Model name for user simulation
            simulator_api_key: API key (defaults to env var)
            simulator_base_url: Custom API base URL (for local models)
            task_prompt: Custom system prompt for user persona
            task_type: Predefined task type if task_prompt not provided
            max_turns: Maximum conversation turns
            success_keywords: Phrases indicating success (fallback if LLM judge disabled)
            failure_keywords: Phrases indicating failure (fallback if LLM judge disabled)
            temperature: Sampling temperature for user LLM
            seed: Random seed for reproducibility
            use_llm_judge: Use LLM-as-a-Judge for evaluation (recommended)
            judge_model: Model for judging (defaults to simulator_model)
        """
        self.simulator_model = simulator_model
        self.max_turns = max_turns
        self.temperature = temperature
        
        # Set API key
        self.api_key = simulator_api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = simulator_base_url
        
        # Initialize LLM client
        self._init_llm_client()
        
        # Set task prompt
        if task_prompt:
            self.task_prompt = task_prompt
        else:
            self.task_prompt = self.TASK_PROMPTS.get(
                task_type, self.TASK_PROMPTS["customer_service"]
            )
        
        # LLM-as-a-Judge settings
        self.use_llm_judge = use_llm_judge
        self.judge_model = judge_model or simulator_model
        
        # Success/failure detection (fallback when LLM judge is disabled)
        self.success_keywords = success_keywords or [
            "thank you", "that resolves", "that fixed it", "exactly what I needed",
            "booking is confirmed", "issue is resolved", "problem solved"
        ]
        self.failure_keywords = failure_keywords or [
            "not helpful", "goodbye", "doesn't work", "not what I was looking for",
            "try elsewhere", "give up", "frustrated"
        ]
        
        # Judge evaluation result (populated at end of episode)
        self._judge_result: Optional[Dict[str, Any]] = None
        
        # Game state
        self._conversation: List[ConversationTurn] = []
        self._turn_count = 0
        self._done = False
        self._success = False
        self._current_task = ""
        
        if seed is not None:
            random.seed(seed)
    
    def _init_llm_client(self):
        """Initialize the LLM client for user simulation."""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )
        
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        self._client = OpenAI(**client_kwargs)
    
    def _generate_user_response(self, agent_message: str) -> str:
        """Generate user response using the simulator LLM."""
        # Build conversation history for the user LLM
        messages = [
            {"role": "system", "content": self.task_prompt + "\n\n" + self._current_task}
        ]
        
        # Add conversation history
        for turn in self._conversation:
            if turn.role == "agent":
                # Agent messages appear as "assistant" to the user simulator
                messages.append({"role": "user", "content": turn.content})
            else:
                # User's own previous messages
                messages.append({"role": "assistant", "content": turn.content})
        
        # Add the latest agent message
        messages.append({"role": "user", "content": agent_message})
        
        # Generate user response
        response = self._client.chat.completions.create(
            model=self.simulator_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=500,
        )
        
        return response.choices[0].message.content
    
    def _generate_initial_user_message(self) -> str:
        """Generate the initial user message to start conversation."""
        messages = [
            {"role": "system", "content": self.task_prompt + "\n\n" + self._current_task},
            {"role": "user", "content": "Start the conversation by stating your request or problem."}
        ]
        
        response = self._client.chat.completions.create(
            model=self.simulator_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=300,
        )
        
        return response.choices[0].message.content
    
    def _check_success(self, text: str) -> bool:
        """Check if conversation indicates success."""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in self.success_keywords)
    
    def _check_failure(self, text: str) -> bool:
        """Check if conversation indicates failure."""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in self.failure_keywords)
    
    def _evaluate_with_llm_judge(self) -> Dict[str, Any]:
        """Evaluate the conversation using LLM-as-a-Judge.
        
        Returns:
            Dictionary with scores and evaluation details.
        """
        import json
        
        # Format conversation for judge
        conv_text = ""
        for turn in self._conversation:
            role_label = "Assistant" if turn.role == "agent" else "User"
            conv_text += f"{role_label}: {turn.content}\n\n"
        
        prompt = self.JUDGE_PROMPT.format(conversation=conv_text)
        
        try:
            response = self._client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=500,
            )
            
            result_text = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                # Try to parse the whole response as JSON
                result = json.loads(result_text)
            
            # Validate and normalize scores
            for key in ["helpfulness", "clarity", "problem_resolution", "professionalism", "efficiency", "overall_score"]:
                if key in result:
                    result[key] = max(1, min(10, int(result[key])))
            
            return result
            
        except Exception as e:
            # Fallback to keyword-based evaluation
            print(f"[LLM Judge] Evaluation failed: {e}, using fallback")
            last_user_msg = self._conversation[-1].content if self._conversation else ""
            success = self._check_success(last_user_msg)
            return {
                "helpfulness": 7 if success else 3,
                "clarity": 5,
                "problem_resolution": 8 if success else 2,
                "professionalism": 5,
                "efficiency": 5,
                "overall_score": 7 if success else 3,
                "success": success,
                "explanation": "Fallback evaluation (LLM judge failed)",
            }
    
    def _calculate_reward_from_judge(self, judge_result: Dict[str, Any]) -> float:
        """Calculate reward from LLM judge evaluation.
        
        Maps the overall_score (1-10) to reward range.
        """
        overall = judge_result.get("overall_score", 5)
        success = judge_result.get("success", False)
        
        if success:
            # Success: reward based on quality (5-10 range)
            # Map overall_score 1-10 to reward 5-10
            return 5.0 + (overall / 10.0) * 5.0
        else:
            # Failure: negative reward based on how bad
            # Map overall_score 1-10 to reward -5 to 0
            return (overall / 10.0) * 5.0 - 5.0
    
    def reset(
        self,
        task_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> str:
        """Reset the game to start a new conversation.
        
        Args:
            task_prompt: Override task prompt for this episode
            seed: Random seed
            **kwargs: Additional arguments
            
        Returns:
            Initial observation (user's first message)
        """
        if seed is not None:
            random.seed(seed)
        
        # Update task prompt if provided
        if task_prompt:
            self._current_task = task_prompt
        else:
            # Generate a random specific task
            self._current_task = self._generate_random_task()
        
        # Reset state
        self._conversation = []
        self._turn_count = 0
        self._done = False
        self._success = False
        self._judge_result = None
        
        # Generate initial user message
        initial_message = self._generate_initial_user_message()
        self._conversation.append(ConversationTurn(role="user", content=initial_message))
        
        return self._format_observation(initial_message)
    
    def _generate_random_task(self) -> str:
        """Generate a random specific task for variety."""
        tasks = [
            "Your flight was cancelled and you need to rebook.",
            "You received a defective product and want a refund.",
            "You need to change your hotel reservation dates.",
            "Your internet connection is not working.",
            "You want to upgrade your subscription plan.",
            "You're looking for recommendations for a restaurant.",
            "You need help resetting your password.",
            "You want to cancel your order.",
        ]
        return random.choice(tasks)
    
    def _format_observation(self, message: str) -> str:
        """Format observation for the agent."""
        obs = f"=== User Message ===\n{message}\n"
        obs += f"\n=== Conversation Turn: {self._turn_count + 1}/{self.max_turns} ==="
        return obs
    
    def step(self, action: str) -> StepResult:
        """Execute agent action and get user response.
        
        Args:
            action: Agent's response to the user
            
        Returns:
            StepResult with user's response, reward, done flag, and info
        """
        if self._done:
            return StepResult(
                observation="Conversation has ended.",
                reward=0.0,
                done=True,
                info={"error": "conversation_ended"},
            )
        
        self._turn_count += 1
        
        # Parse agent's action
        parsed_action = self._parse_action(action)
        
        # Add agent message to conversation
        self._conversation.append(ConversationTurn(role="agent", content=parsed_action))
        
        # Generate user response
        user_response = self._generate_user_response(parsed_action)
        self._conversation.append(ConversationTurn(role="user", content=user_response))
        
        # Check for episode end conditions
        episode_ended = False
        end_reason = ""
        
        if self._check_success(user_response):
            episode_ended = True
            end_reason = "success_keyword"
        elif self._check_failure(user_response):
            episode_ended = True
            end_reason = "failure_keyword"
        elif self._turn_count >= self.max_turns:
            episode_ended = True
            end_reason = "timeout"
        
        # Calculate reward
        if episode_ended:
            self._done = True
            
            if self.use_llm_judge:
                # Use LLM-as-a-Judge for evaluation
                self._judge_result = self._evaluate_with_llm_judge()
                reward = self._calculate_reward_from_judge(self._judge_result)
                self._success = self._judge_result.get("success", False)
                
                # Add evaluation summary to response
                judge_summary = (
                    f"\n\n=== LLM Judge Evaluation ===\n"
                    f"Overall Score: {self._judge_result.get('overall_score', 'N/A')}/10\n"
                    f"Success: {self._success}\n"
                    f"Explanation: {self._judge_result.get('explanation', 'N/A')}"
                )
                user_response = f"{user_response}{judge_summary}"
            else:
                # Fallback to keyword-based evaluation
                if end_reason == "success_keyword":
                    self._success = True
                    reward = self.REWARD_SUCCESS
                else:
                    self._success = False
                    reward = self.REWARD_FAILURE
            
            # Add end reason prefix
            if end_reason == "timeout":
                user_response = f"TIMEOUT: Maximum turns reached.\n\n{user_response}"
            elif self._success:
                user_response = f"SUCCESS: {user_response}"
            else:
                user_response = f"FAILURE: {user_response}"
        else:
            reward = self.REWARD_STEP
        
        # Build info dict
        info = {
            "turn": self._turn_count,
            "success": self._success,
            "agent_message": parsed_action,
            "user_message": user_response,
        }
        
        # Add judge evaluation if available
        if self._judge_result:
            info["judge_evaluation"] = self._judge_result
        
        return StepResult(
            observation=self._format_observation(user_response),
            reward=reward,
            done=self._done,
            info=info,
        )
    
    def _parse_action(self, raw_action: str) -> str:
        """Parse action from LLM output."""
        # Try to extract from <response> tags
        match = re.search(
            r"<response>\s*(.*?)\s*</response>", raw_action, re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()
        
        # Otherwise use the whole output
        return raw_action.strip()
    
    def get_system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        return (
            "You are a helpful assistant engaging in a conversation with a user.\n"
            "Your goal is to understand the user's needs and help them effectively.\n\n"
            "IMPORTANT: Respond naturally and helpfully. Be concise but thorough.\n"
            "If you need more information, ask clarifying questions.\n"
            "If you can help, provide clear solutions or information.\n\n"
            "Wrap your response in <response></response> tags.\n\n"
            "Example:\n"
            "<response>I understand you're having trouble with your order. "
            "Could you please provide your order number so I can look into this?</response>"
        )
    
    def get_initial_user_message(self) -> str:
        """Return context for the agent."""
        return "You are helping a user. Respond to their message."
    
    def get_state(self) -> Dict[str, Any]:
        """Return current game state."""
        state = {
            "turn_count": self._turn_count,
            "max_turns": self.max_turns,
            "done": self._done,
            "success": self._success,
            "conversation_length": len(self._conversation),
            "use_llm_judge": self.use_llm_judge,
        }
        if self._judge_result:
            state["judge_evaluation"] = self._judge_result
        return state
    
    def generate_initial_state(self) -> Dict[str, Any]:
        """Generate random initial state for training."""
        return {
            "seed": random.randint(0, 1000000),
        }
    
    def get_user_message_with_state(self, **kwargs) -> str:
        """Generate user message with state for prompt."""
        self.reset(**kwargs)
        initial_obs = self._format_observation(self._conversation[0].content)
        return f"{initial_obs}\n\nRespond to the user."
    
    def get_interaction_name(self) -> str:
        """Return interaction name."""
        return "llm_user_simulator"

