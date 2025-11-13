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

Your tasks:
1. Read the current objectives to see what the agent is working on
2. Read recent dialogues to understand story progression
3. Evaluate if current objectives are appropriate
4. If dialogues suggest new objectives, create them
5. If the agent is struggling, adjust policy parameters
6. If objectives are completed, mark them as complete

Remember: The DRL agent learns the low-level actions. Your job is to set the RIGHT objectives and tune the rewards.
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

You have access to these tools:
1. read_objectives() - See what objectives the agent is working on
2. write_objective() - Create new objectives based on game progress
3. complete_objective() - Mark objectives as done
4. read_dialogues() - Read in-game dialogues to understand story
5. update_policy() - Adjust reward weights to guide learning

Your workflow:
1. ALWAYS start by reading objectives and dialogues
2. Analyze if current objectives are appropriate
3. Based on dialogues, infer what should happen next in the game
4. Create new objectives that guide the agent toward game progression
5. If agent is stuck, adjust policy parameters (e.g., increase exploration)

Example objective types:
- location: {"map": "ROUTE_101", "x": 10, "y": 5}
- dialogue: {"npc": "PROF_BIRCH", "trigger": "talk"}
- custom: {"condition": "any descriptive condition"}

Be strategic and incremental. Don't create too many objectives at once.
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
    
    def _save_llm_log(self, messages: List[Dict[str, Any]], game_state_summary: str):
        """
        Save LLM conversation to a log file.
        
        Args:
            messages: The conversation messages
            game_state_summary: Summary of game state at time of call
        """
        from datetime import datetime
        
        self.llm_call_count += 1
        
        # Extract tool calls summary for quick reference
        tool_calls_summary = []
        for msg in messages:
            if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                for tool_call in msg['tool_calls']:
                    tool_calls_summary.append({
                        'tool': tool_call['function']['name'],
                        'arguments': tool_call['function'].get('arguments', {})
                    })
        
        # Get final response (last assistant message with content)
        final_response = None
        for msg in reversed(messages):
            if msg.get('role') == 'assistant' and msg.get('content'):
                final_response = msg.get('content')
                break
        
        # Create log entry
        log_entry = {
            "call_number": self.llm_call_count,
            "timestamp": datetime.now().isoformat(),
            "training_step": self.num_timesteps,
            "model": self.llm_model,
            "game_state_summary": game_state_summary,
            "tool_calls_summary": tool_calls_summary,  # 🆕 Quick reference
            "final_response": final_response,  # 🆕 LLM's final analysis
            "full_conversation": messages,  # Complete conversation for debugging
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
            if final_response:
                logger.info(f"   Response: {final_response[:100]}...")
        except Exception as e:
            logger.error(f"❌ Failed to save LLM log: {e}")


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
