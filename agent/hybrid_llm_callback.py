"""
Hybrid DRL+LLM Callback with Tool Calling

This callback integrates LLM-based strategic planning with DRL training:
1. DRL agent learns low-level policies (movement, actions)
2. LLM evaluates progress every N episodes
3. LLM sets objectives based on dialogues and game state
4. LLM adjusts policy parameters using tool calling
"""

import logging
import json
import ollama
from typing import List, Dict, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback

from agent.objectives_manager import ObjectivesManager, Objective
from agent.llm_tools import LLMTools, get_tool_definitions

logger = logging.getLogger(__name__)


class HybridLLMCallback(BaseCallback):
    """
    Callback that uses LLM with tool calling to guide DRL training.
    
    The LLM acts as a "strategic planner" that:
    - Analyzes game progress and dialogues
    - Sets objectives dynamically
    - Adjusts reward weights and policy parameters
    - Evaluates if current strategies are working
    """
    
    def __init__(
        self,
        check_frequency: int = 5000,  # Check every N steps (after ~few episodes)
        llm_model: str = "qwen3:8b",
        objectives_file: str = "agent/current_objectives.json",
        max_llm_iterations: int = 10,  # Max tool calling iterations
        log_dir: str = "logs/llm_responses",  # 🆕 Directory for LLM logs
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.check_frequency = check_frequency
        self.llm_model = llm_model
        self.max_llm_iterations = max_llm_iterations
        self.last_check_step = 0
        self.log_dir = log_dir
        
        # 🆕 Create logs directory
        from pathlib import Path
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize objectives manager
        self.objectives_manager = ObjectivesManager(objectives_file)
        
        # Initialize LLM tools
        self.llm_tools = LLMTools(self.objectives_manager)
        
        # Get tool definitions for Ollama
        self.tool_definitions = get_tool_definitions()
        
        # Episode tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episodes_since_last_check = 0
        
        # 🆕 LLM conversation tracking
        self.llm_call_count = 0
        self._last_episode_count = 0  # Track episodes for reward comparison
        
        logger.info(f"🤖 Hybrid LLM Callback initialized with model: {llm_model}")
        logger.info(f"📋 Available tools: {[t['function']['name'] for t in self.tool_definitions]}")
        logger.info(f"📁 LLM responses will be logged to: {self.log_dir}")
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        
        # Check if it's time to consult the LLM
        if self.num_timesteps - self.last_check_step < self.check_frequency:
            return True
        
        self.last_check_step = self.num_timesteps
        
        logger.info("="*60)
        logger.info(f"🤖 LLM STRATEGIC ANALYSIS at step {self.num_timesteps}")
        logger.info("="*60)
        
        # Run LLM analysis with tool calling
        self._run_llm_analysis()
        
        return True
    
    def _run_llm_analysis(self):
        """
        Run LLM analysis with tool calling.
        The LLM can use tools to:
        - Read current objectives
        - Read dialogues
        - Create new objectives
        - Adjust policy parameters
        """
        
        # 🆕 Save objectives state before LLM analysis
        self._objectives_before = self.objectives_manager.to_json_str()
        
        # Prepare the system prompt
        system_prompt = self._create_system_prompt()
        
        # Get current game state summary
        game_state_summary = self._get_game_state_summary()
        
        # Create user message
        user_message = f"""
Analyze the current training progress and take appropriate actions:

{game_state_summary}

🎯 YOUR TASKS (follow this order):

1️⃣ READ OBJECTIVES:
   - Call read_objectives() to see active objectives and their dialogue_history

2️⃣ EVALUATE EACH ACTIVE OBJECTIVE:
   - For each objective, look at its dialogue_history field (NOT global dialogues)
   - Call evaluate_objective_completion(objective_id, "your reasoning") to analyze
   - If completion criteria met: call complete_objective(objective_id)
   - If not met: leave it active to gather more dialogues

3️⃣ CREATE NEW OBJECTIVE (if needed):
   - If NO active objectives OR last objective just completed:
     a. Call read_dialogues() to see recent story context
     b. Create ONE new objective with write_objective() based on what should happen next
     c. The new objective will start with EMPTY dialogue_history and track future dialogues

4️⃣ ADJUST POLICY (optional):
   - If agent is stuck, call update_policy() to adjust learning

🔑 KEY POINTS:
- Each objective has ITS OWN dialogue_history (isolated context)
- New objectives start with NO previous dialogues
- Only evaluate completion based on THAT objective's dialogue_history
- Create objectives ONE AT A TIME for focused learning

Remember: The DRL agent learns low-level actions. You set HIGH-LEVEL objectives based on story progression.
"""
        
        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Run tool calling loop
        for iteration in range(self.max_llm_iterations):
            logger.info(f"🔄 LLM iteration {iteration + 1}/{self.max_llm_iterations}")
            
            try:
                # Call LLM with tools
                response = ollama.chat(
                    model=self.llm_model,
                    messages=messages,
                    tools=self.tool_definitions,
                    think=False
                )
                
                message = response['message']
                messages.append(message)
                
                # Check if LLM provided final answer
                if message.get('content'):
                    logger.info(f"✅ LLM Analysis Complete:")
                    logger.info(message['content'])
                    break
                
                # Check if LLM wants to use tools
                tool_calls = message.get('tool_calls')
                if tool_calls:
                    # Execute all requested tools
                    for tool_call in tool_calls:
                        tool_result = self._execute_tool(tool_call)
                        
                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "content": json.dumps(tool_result)
                        })
                else:
                    logger.warning("⚠️ LLM did not provide content or tool calls. Ending loop.")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Error in LLM analysis: {e}")
                break
        
        # 🆕 Save LLM conversation to log file
        self._save_llm_log(messages, game_state_summary)
        
        # 🆕 Track episode count for next comparison
        self._last_episode_count = len(self.episode_rewards)
        
        # Save updated objectives
        self.objectives_manager.save()
    
    def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call from the LLM."""
        function_name = tool_call['function']['name']
        
        # Parse arguments
        args_raw = tool_call['function'].get('arguments', {})
        if isinstance(args_raw, str):
            try:
                arguments = json.loads(args_raw)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse arguments: {args_raw}")
                return {'error': 'Invalid arguments'}
        else:
            arguments = args_raw
        
        logger.info(f"🛠️ Executing tool: {function_name}({arguments})")
        
        # Execute the appropriate tool
        try:
            if function_name == 'read_objectives':
                result = self.llm_tools.read_objectives()
            elif function_name == 'evaluate_objective_completion':
                result = self.llm_tools.evaluate_objective_completion(**arguments)
            elif function_name == 'write_objective':
                result = self.llm_tools.write_objective(**arguments)
            elif function_name == 'complete_objective':
                result = self.llm_tools.complete_objective(**arguments)
            elif function_name == 'read_dialogues':
                result = self.llm_tools.read_dialogues(**arguments)
            elif function_name == 'update_policy':
                result = self.llm_tools.update_policy(**arguments)
                # Apply policy changes to environments
                self._apply_policy_changes()
            else:
                result = {'error': f'Unknown tool: {function_name}'}
            
            logger.info(f"✅ Tool result: {json.dumps(result, indent=2)[:200]}...")
            return result
            
        except Exception as e:
            logger.error(f"❌ Tool execution failed: {e}")
            return {'error': str(e)}
    
    def _apply_policy_changes(self):
        """Apply policy parameter changes to the training environments."""
        # Get environments
        num_envs = self.training_env.num_envs if hasattr(self.training_env, 'num_envs') else 1
        
        policy_params = self.llm_tools.policy_params
        
        logger.info(f"🎯 Applying policy changes to {num_envs} environments:")
        logger.info(f"   {policy_params}")
        
        # For now, we'll store these in the callback and use them in reward calculation
        # In a more advanced version, you could modify the environment's reward function
        # TODO: Implement environment-level policy parameter updates
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the LLM."""
        return """You are an AI strategic planner for a Pokemon Emerald speedrun training system.

Your role:
- The DRL (Deep Reinforcement Learning) agent handles LOW-LEVEL actions (button presses, movement)
- YOU handle HIGH-LEVEL strategy (objectives, goals, reward tuning)

🆕 OBJECTIVE SYSTEM WITH PER-OBJECTIVE DIALOGUE TRACKING:
- Each objective has its OWN dialogue_history that tracks dialogues while it's active
- When you create a new objective, it starts with an EMPTY dialogue history
- Only dialogues encountered AFTER objective creation are tracked in that objective
- This gives you FOCUSED CONTEXT for each objective without past dialogues interfering

You have access to these tools:
1. read_objectives() - See active objectives with THEIR dialogue_history (not global history)
2. evaluate_objective_completion(objective_id, reasoning) - Analyze if objective is complete based on ITS dialogue_history
3. complete_objective(objective_id) - Mark objective as complete (use AFTER evaluating)
4. write_objective() - Create new objective (starts with empty dialogue_history)
5. read_dialogues(count) - Read GLOBAL dialogue history (for general context)
6. update_policy() - Adjust reward weights to guide learning

🎯 RECOMMENDED WORKFLOW:
1. Call read_objectives() to see active objectives and their dialogue_history
2. For each active objective:
   a. Analyze its dialogue_history field (not global dialogues)
   b. Call evaluate_objective_completion(objective_id, "reasoning") to check if complete
   c. If complete: call complete_objective(objective_id)
   d. If not complete: leave it active to gather more dialogues
3. If no active objectives OR last objective completed:
   a. Call read_dialogues() to see recent game context
   b. Create a NEW objective with write_objective() based on story progression
   c. The new objective will start tracking dialogues from this point forward

📋 OBJECTIVE DESIGN TIPS:
- Make objectives DIALOGUE-BASED: target should check for specific keywords/NPCs
- Example: {"keywords": ["PROF. BIRCH", "WELCOME"], "any_match": true}
- Each objective should have CLEAR completion criteria visible in dialogue
- Create ONE objective at a time for focused learning
- New objectives inherit NO dialogue history from previous ones

Example objective types:
- dialogue: {"keywords": ["PROF_BIRCH", "WELCOME"], "any_match": true}
- location: {"keywords": ["ROUTE_101", "OLDALE"], "any_match": true}
- custom: {"condition": "any descriptive condition"}

Be strategic and incremental. Create objectives one at a time based on story flow.
"""
    
    def _get_game_state_summary(self) -> str:
        """Get a summary of current game state for the LLM."""
        
        # Get training stats
        avg_reward = sum(self.episode_rewards[-10:]) / len(self.episode_rewards[-10:]) if self.episode_rewards else 0
        avg_length = sum(self.episode_lengths[-10:]) / len(self.episode_lengths[-10:]) if self.episode_lengths else 0
        
        summary = f"""
TRAINING STATISTICS:
- Total steps: {self.num_timesteps}
- Episodes completed: {len(self.episode_rewards)}
- Average reward (last 10 episodes): {avg_reward:.2f}
- Average episode length: {avg_length:.0f}

CURRENT OBJECTIVES:
{self.objectives_manager.to_json_str()}
"""
        return summary
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (episode)."""
        # Track episode statistics
        # Note: This might need adjustment based on how you track episodes
        pass
    
    def _get_reward_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive reward statistics for logging.
        
        Returns:
            Dictionary with reward statistics including recent trends
        """
        if not self.episode_rewards:
            return {
                "total_episodes": 0,
                "avg_reward_last_10": 0.0,
                "avg_reward_last_50": 0.0,
                "avg_reward_all_time": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "recent_rewards": [],
                "reward_trend": "N/A"
            }
        
        recent_10 = self.episode_rewards[-10:]
        recent_50 = self.episode_rewards[-50:]
        all_rewards = self.episode_rewards
        
        avg_10 = sum(recent_10) / len(recent_10) if recent_10 else 0.0
        avg_50 = sum(recent_50) / len(recent_50) if recent_50 else 0.0
        avg_all = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        
        # Calculate reward trend (comparing last 10 vs previous 10)
        if len(self.episode_rewards) >= 20:
            prev_10 = self.episode_rewards[-20:-10]
            avg_prev_10 = sum(prev_10) / len(prev_10)
            if avg_10 > avg_prev_10 * 1.1:
                trend = "📈 IMPROVING (+{:.1f}%)".format((avg_10 / avg_prev_10 - 1) * 100)
            elif avg_10 < avg_prev_10 * 0.9:
                trend = "📉 DECLINING ({:.1f}%)".format((avg_10 / avg_prev_10 - 1) * 100)
            else:
                trend = "➡️ STABLE"
        else:
            trend = "⏳ COLLECTING DATA"
        
        return {
            "total_episodes": len(self.episode_rewards),
            "avg_reward_last_10": round(avg_10, 3),
            "avg_reward_last_50": round(avg_50, 3),
            "avg_reward_all_time": round(avg_all, 3),
            "min_reward": round(min(all_rewards), 3) if all_rewards else 0.0,
            "max_reward": round(max(all_rewards), 3) if all_rewards else 0.0,
            "recent_rewards": [round(r, 3) for r in recent_10],
            "reward_trend": trend,
            "episodes_since_last_check": len(self.episode_rewards) - getattr(self, '_last_episode_count', 0)
        }
    
    def _save_llm_log(self, messages: List[Dict[str, Any]], game_state_summary: str):
        """
        Save LLM conversation to a log file.
        
        Args:
            messages: The conversation messages
            game_state_summary: Summary of game state at time of call
        """
        from datetime import datetime
        
        self.llm_call_count += 1
        
        # Helper function to convert tool_calls to serializable format
        def serialize_tool_call(tool_call):
            """Convert ToolCall object to dict"""
            if isinstance(tool_call, dict):
                return tool_call
            # Handle ToolCall object
            return {
                'function': {
                    'name': tool_call.function.name if hasattr(tool_call.function, 'name') else str(tool_call.function.get('name', '')),
                    'arguments': tool_call.function.arguments if hasattr(tool_call.function, 'arguments') else tool_call.function.get('arguments', {})
                }
            }
        
        # Extract tool calls summary for quick reference
        tool_calls_summary = []
        for msg in messages:
            msg_dict = msg if isinstance(msg, dict) else {'role': getattr(msg, 'role', None), 'content': getattr(msg, 'content', None)}
            if msg_dict.get('role') == 'assistant':
                tool_calls = msg.get('tool_calls') if isinstance(msg, dict) else getattr(msg, 'tool_calls', None)
                if tool_calls:
                    for tool_call in tool_calls:
                        tc_dict = serialize_tool_call(tool_call)
                        tool_calls_summary.append({
                            'tool': tc_dict['function']['name'],
                            'arguments': tc_dict['function'].get('arguments', {})
                        })
        
        # Get final response (last assistant message with content)
        final_response = None
        for msg in reversed(messages):
            if isinstance(msg, dict):
                if msg.get('role') == 'assistant' and msg.get('content'):
                    final_response = msg.get('content')
                    break
            else:
                if getattr(msg, 'role', None) == 'assistant' and getattr(msg, 'content', None):
                    final_response = getattr(msg, 'content')
                    break
        
        # Convert messages to JSON-serializable format
        # Ollama Message objects and ToolCall objects need to be converted to dicts
        serializable_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                # Already a dict, but check if tool_calls need serialization
                msg_copy = msg.copy()
                if msg_copy.get('tool_calls'):
                    msg_copy['tool_calls'] = [serialize_tool_call(tc) for tc in msg_copy['tool_calls']]
                serializable_messages.append(msg_copy)
            else:
                # Convert Ollama Message object to dict
                msg_dict = {
                    'role': getattr(msg, 'role', None),
                    'content': getattr(msg, 'content', None)
                }
                # Add tool_calls if present (and serialize them)
                tool_calls = getattr(msg, 'tool_calls', None)
                if tool_calls:
                    msg_dict['tool_calls'] = [serialize_tool_call(tc) for tc in tool_calls]
                
                serializable_messages.append(msg_dict)
        
        # 🆕 Calculate reward statistics
        reward_stats = self._get_reward_statistics()
        
        # Create log entry
        log_entry = {
            "call_number": self.llm_call_count,
            "timestamp": datetime.now().isoformat(),
            "training_step": self.num_timesteps,
            "model": self.llm_model,
            "game_state_summary": game_state_summary,
            "reward_statistics": reward_stats,  # 🆕 Reward tracking
            "tool_calls_summary": tool_calls_summary,  # 🆕 Quick reference
            "final_response": final_response,  # 🆕 LLM's final analysis
            "full_conversation": serializable_messages,  # Complete conversation for debugging (JSON serializable)
            "objectives_before": json.loads(self.objectives_manager.to_json_str()) if hasattr(self, '_objectives_before') else None,
            "objectives_after": json.loads(self.objectives_manager.to_json_str())
        }
        
        # Generate filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_call_{self.llm_call_count:03d}_step_{self.num_timesteps}_{timestamp_str}.json"
        filepath = f"{self.log_dir}/{filename}"
        
        # Save to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 LLM conversation saved to: {filepath}")
            logger.info(f"   Tools used: {[t['tool'] for t in tool_calls_summary]}")
            
            # 🆕 Log reward statistics
            reward_stats = log_entry.get('reward_statistics', {})
            if reward_stats.get('total_episodes', 0) > 0:
                logger.info(f"   📊 Rewards - Avg (last 10): {reward_stats['avg_reward_last_10']:.3f}, Trend: {reward_stats['reward_trend']}")
                logger.info(f"   📈 Episodes: {reward_stats['total_episodes']} total, {reward_stats['episodes_since_last_check']} since last check")
            
            if final_response:
                logger.info(f"   💬 Response: {final_response[:100]}...")
        except Exception as e:
            logger.error(f"❌ Failed to save LLM log: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")


# Standalone function to test the LLM with tools
def test_llm_tools():
    """Test the LLM tool calling system independently."""
    
    print("="*60)
    print("🧪 Testing LLM Tool Calling System")
    print("="*60)
    
    # Initialize managers
    objectives_manager = ObjectivesManager("test_objectives.json")
    llm_tools = LLMTools(objectives_manager)
    tool_definitions = get_tool_definitions()
    
    # Add some test dialogues
    objectives_manager.add_dialogue(
        "Welcome to the world of Pokemon! My name is Birch.",
        npc="PROF_BIRCH",
        location="LITTLEROOT_TOWN"
    )
    objectives_manager.add_dialogue(
        "Go ahead and explore Route 101!",
        npc="PROF_BIRCH",
        location="LITTLEROOT_TOWN"
    )
    
    # Create system prompt
    system_prompt = """You are testing a tool calling system for Pokemon Emerald.
Use the available tools to:
1. Read current objectives
2. Read dialogues
3. Create a new objective based on the dialogue (to explore Route 101)
"""
    
    user_message = "Analyze the game state and create appropriate objectives."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    # Tool calling loop
    for i in range(5):
        print(f"\n🔄 Iteration {i+1}")
        
        response = ollama.chat(
            model="qwen3:8b",
            messages=messages,
            tools=tool_definitions,
            think=False
        )
        
        message = response['message']
        messages.append(message)
        
        if message.get('content'):
            print(f"✅ Final response: {message['content']}")
            break
        
        tool_calls = message.get('tool_calls')
        if tool_calls:
            for tool_call in tool_calls:
                func_name = tool_call['function']['name']
                args = tool_call['function'].get('arguments', {})
                if isinstance(args, str):
                    args = json.loads(args)
                
                print(f"🛠️ Tool: {func_name}({args})")
                
                # Execute tool
                if func_name == 'read_objectives':
                    result = llm_tools.read_objectives()
                elif func_name == 'read_dialogues':
                    result = llm_tools.read_dialogues(**args)
                elif func_name == 'write_objective':
                    result = llm_tools.write_objective(**args)
                else:
                    result = {'error': 'Unknown tool'}
                
                print(f"   Result: {json.dumps(result, indent=2)[:200]}...")
                
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result)
                })
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    # Run test
    logging.basicConfig(level=logging.INFO)
    test_llm_tools()
