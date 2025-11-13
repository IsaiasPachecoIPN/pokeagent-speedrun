# Hybrid DRL+LLM System Guide

## 🚀 Overview

This system combines **Deep Reinforcement Learning (DRL)** with **Large Language Model (LLM)** strategic planning to train an agent to play Pokemon Emerald.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID TRAINING LOOP                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐        ┌──────────────────┐           │
│  │   DRL Agent     │◄───────┤  Game Emulator   │           │
│  │  (PPO Policy)   │        │  (mGBA)          │           │
│  └────────┬────────┘        └──────────────────┘           │
│           │                                                   │
│           │ Low-level actions (buttons, movement)            │
│           │                                                   │
│  ┌────────▼────────┐                                         │
│  │  Environment    │                                         │
│  │  (drl_env.py)   │                                         │
│  └────────┬────────┘                                         │
│           │                                                   │
│           │ Captures dialogues, tracks progress              │
│           │                                                   │
│  ┌────────▼──────────────────────────────────────┐          │
│  │         Objectives Manager                     │          │
│  │  - Current objectives (JSON file)              │          │
│  │  - Dialogue history                            │          │
│  │  - Progress tracking                           │          │
│  └────────┬──────────────────────────────────────┘          │
│           │                                                   │
│           │ Every N episodes (~5000 steps)                   │
│           │                                                   │
│  ┌────────▼────────────────────────────────────┐            │
│  │       LLM Strategist (Ollama)               │            │
│  │  - Reads objectives                         │            │
│  │  - Reads dialogues                          │            │
│  │  - Analyzes progress                        │            │
│  │  - Sets NEW objectives via tool calling     │            │
│  │  - Adjusts policy parameters                │            │
│  └─────────────────────────────────────────────┘            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Key Concepts

### 1. **DRL Agent** (Low-Level Controller)
- **What it does**: Learns which buttons to press (A, B, arrows, etc.)
- **Input**: Game state (map tiles, position, party info)
- **Output**: Actions (button presses)
- **Training**: PPO (Proximal Policy Optimization)

### 2. **LLM Strategist** (High-Level Planner)
- **What it does**: Sets objectives and guides learning strategy
- **Input**: Training progress, dialogues, objectives
- **Output**: New objectives, policy adjustments
- **Model**: Ollama (qwen3:8b or similar)

### 3. **Objectives System**
Objectives are stored in `agent/current_objectives.json`:

```json
{
  "objectives": {
    "exit_starting_room": {
      "id": "exit_starting_room",
      "name": "Exit Starting Room",
      "description": "Leave the starting room and explore Littleroot Town",
      "type": "location",
      "target": {
        "map": "LITTLEROOT_TOWN",
        "condition": "different_from_start"
      },
      "reward_weight": 2.0,
      "completed": false,
      "progress": 0.5
    }
  }
}
```

## 🛠️ LLM Tools

The LLM has access to these tools via function calling:

### 1. `read_objectives()`
Reads current objectives and their status.

**Example usage by LLM:**
```
LLM: I need to understand what the agent is working on.
→ Calls read_objectives()
← Returns: {"active_objectives": [...], "completed_objectives": [...]}
```

### 2. `read_dialogues(count=10)`
Reads recent in-game dialogues.

**Example usage by LLM:**
```
LLM: Let me see what NPCs have said recently.
→ Calls read_dialogues(count=10)
← Returns: {"dialogues": [{"text": "Welcome to Pokemon!", "npc": "PROF_BIRCH", ...}]}
```

### 3. `write_objective(name, description, type, target, reward_weight)`
Creates a new objective for the agent.

**Example usage by LLM:**
```
LLM: The dialogue mentions Route 101. I should create an objective to go there.
→ Calls write_objective(
    name="Explore Route 101",
    description="Travel north to Route 101 as suggested by Prof. Birch",
    type="location",
    target={"map": "ROUTE_101"},
    reward_weight=1.5
  )
← Returns: {"success": true, "objective_id": "explore_route_101"}
```

### 4. `complete_objective(objective_id)`
Marks an objective as completed.

**Example usage by LLM:**
```
LLM: The agent has reached Littleroot Town. Let me mark that objective as complete.
→ Calls complete_objective(objective_id="exit_starting_room")
← Returns: {"success": true}
```

### 5. `update_policy(reward_weight_exploration=1.5, ...)`
Adjusts policy parameters to change agent behavior.

**Example usage by LLM:**
```
LLM: The agent is stuck. Let me increase exploration rewards.
→ Calls update_policy(
    reward_weight_exploration=2.0,
    penalty_stationary=-0.2
  )
← Returns: {"success": true, "policy_params": {...}}
```

## 📋 How It Works Step-by-Step

### Phase 1: Initial Training (Steps 0-5000)
1. **DRL agent** starts with default objective: "Exit starting room"
2. Agent learns to move around, press buttons
3. Environment captures any dialogues automatically
4. Agent gets rewards for:
   - Moving to new positions
   - Exploring new map tiles
   - Making progress toward objectives

### Phase 2: LLM Evaluation (Step 5000)
1. **Hybrid callback** triggers LLM analysis
2. LLM receives:
   - Training statistics (avg reward, episode length)
   - Current objectives
   - Dialogue history
3. LLM uses tools to:
   ```
   Step 1: read_objectives()
   Step 2: read_dialogues()
   Step 3: Analyze context
   Step 4: write_objective() - Creates "Talk to Prof. Birch"
   Step 5: update_policy() - Increase dialogue rewards
   ```
4. New objectives are saved to JSON file

### Phase 3: Continued Training (Steps 5000-10000)
1. Agent continues training with updated objectives
2. Reward function now emphasizes:
   - Talking to NPCs (higher weight)
   - Reaching specific locations
3. Environment continues capturing dialogues

### Phase 4: Next LLM Evaluation (Step 10000)
1. LLM analyzes new dialogues
2. Sees: "Go help me with the Pokemon on Route 101!"
3. Creates new objective: "Help Professor on Route 101"
4. Adjusts policy to prioritize exploration northward

## 🎮 Usage

### Basic Training with Hybrid Mode

```bash
python train_ppo.py --mode train --hybrid --n-envs 1 --timesteps 100000
```

### Parameters

- `--hybrid`: Enable hybrid DRL+LLM mode
- `--n-envs`: Number of parallel environments (recommend 1 for hybrid mode)
- `--timesteps`: Total training steps
- `--visualize`: Show game window (only with n-envs=1)

### Example Commands

**Hybrid mode with visualization:**
```bash
python train_ppo.py --hybrid --visualize --timesteps 50000
```

**Hybrid mode, multiple environments (faster):**
```bash
python train_ppo.py --hybrid --n-envs 4 --timesteps 200000
```

**Pure DRL (no LLM):**
```bash
python train_ppo.py --pure-drl --n-envs 4 --timesteps 1000000
```

## 🧪 Testing the System

### Test LLM Tool Calling

```bash
python test_hybrid_llm.py
```

This will:
1. Create a test objectives file
2. Add sample dialogues
3. Call the LLM with tool calling enabled
4. Show the LLM creating objectives based on dialogues

### Expected Output

```
🧪 Testing LLM Tool Calling System
==================================================
🔄 Iteration 1
🛠️ Tool: read_objectives({})
   Result: {"active_objectives": [...]}

🔄 Iteration 2
🛠️ Tool: read_dialogues({"count": 10})
   Result: {"dialogues": [...]}

🔄 Iteration 3
🛠️ Tool: write_objective({
    "name": "Explore Route 101",
    "description": "Travel to Route 101 as suggested by Prof. Birch",
    "type": "location",
    "target": {"map": "ROUTE_101"},
    "reward_weight": 1.5
})
   Result: {"success": true, "objective_id": "explore_route_101"}

✅ Final response: I've analyzed the dialogues and created an objective
to explore Route 101. The agent should head north from Littleroot Town.
```

## 📁 File Structure

```
agent/
  ├── objectives_manager.py       # Manages objectives JSON file
  ├── llm_tools.py                # Tool definitions for LLM
  ├── hybrid_llm_callback.py      # Callback that runs LLM analysis
  ├── drl_env.py                  # Environment with dialogue capture
  └── current_objectives.json     # Active objectives (auto-generated)

train_ppo.py                      # Training script with --hybrid flag
test_hybrid_llm.py                # Test LLM tool calling
HYBRID_DRL_LLM_GUIDE.md           # This guide
```

## 🔧 Configuration

### LLM Model Selection

Edit `train_ppo.py`:
```python
hybrid_callback = HybridLLMCallback(
    llm_model="qwen3:8b",  # Change to your preferred model
    ...
)
```

Supported models (via Ollama):
- `qwen3:8b` (recommended)
- `llama3:8b`
- `mistral:7b`
- Any Ollama model with function calling support

### Check Frequency

How often should the LLM evaluate progress?

```python
hybrid_callback = HybridLLMCallback(
    check_frequency=5000,  # Every 5000 steps (~5-10 episodes)
    ...
)
```

- Lower = More frequent LLM analysis (slower, more guidance)
- Higher = Less frequent (faster, more autonomous DRL learning)

### Objective Types

When creating objectives, use these types:

#### Location Objective
```json
{
  "type": "location",
  "target": {
    "map": "ROUTE_101",
    "x": 10,
    "y": 5
  }
}
```

#### Dialogue Objective
```json
{
  "type": "dialogue",
  "target": {
    "npc": "PROF_BIRCH",
    "trigger": "talk"
  }
}
```

#### Item Objective
```json
{
  "type": "item",
  "target": {
    "item_id": "POKE_BALL",
    "count": 5
  }
}
```

#### Custom Objective
```json
{
  "type": "custom",
  "target": {
    "condition": "Reach the Pokemon Center"
  }
}
```

## 🎓 Curriculum Learning

The hybrid system implements **curriculum learning**:

1. **Stage 1** (Early game): Simple objectives like "Exit room", "Explore town"
2. **Stage 2** (Mid game): NPC interactions, item collection
3. **Stage 3** (Late game): Battle strategies, gym challenges

The LLM automatically progresses through stages by:
- Reading dialogues to understand story progression
- Creating appropriate objectives for current game state
- Adjusting reward weights based on success/failure

## 🐛 Troubleshooting

### LLM Not Creating Objectives

**Problem**: LLM calls tools but doesn't create objectives.

**Solution**:
- Check Ollama is running: `ollama list`
- Verify model supports function calling: `ollama show qwen3:8b`
- Check logs for LLM responses

### Dialogues Not Being Captured

**Problem**: `dialogue_history` is empty in objectives file.

**Solution**:
- Ensure `enable_dialogue_capture = True` in environment
- Check if game has reached dialogue sections
- Verify OCR/memory reading is working

### Agent Ignoring Objectives

**Problem**: Agent doesn't seem to follow LLM-set objectives.

**Solution**:
- Objectives affect reward shaping, not direct control
- Agent needs time to learn (objectives guide, don't force)
- Increase `reward_weight` for important objectives
- Check if objective is achievable from current game state

## 📊 Monitoring Training

### View Current Objectives

```bash
cat agent/current_objectives.json | jq
```

### View Tensorboard

```bash
tensorboard --logdir=./tensorboard_logs
```

### Check Dialogue History

```python
from agent.objectives_manager import ObjectivesManager
manager = ObjectivesManager()
dialogues = manager.get_recent_dialogues(20)
for d in dialogues:
    print(f"[{d['location']}] {d['text']}")
```

## 🚦 Best Practices

### 1. Start Simple
Begin with basic objectives (movement, exploration) before complex goals.

### 2. Iterate Gradually
Let DRL learn for several episodes before next LLM evaluation.

### 3. Balance Autonomy
Don't create too many objectives - let DRL explore.

### 4. Use Reward Weights Wisely
- Default: 1.0
- Important: 1.5-2.0
- Critical: 2.0-3.0

### 5. Monitor Progress
Check objectives file regularly to see what LLM is creating.

## 📚 Advanced Topics

### Custom Tools

Add new tools in `llm_tools.py`:

```python
def analyze_battle_strategy(self, party_levels: list) -> dict:
    """Analyze if party is ready for next gym."""
    avg_level = sum(party_levels) / len(party_levels)
    return {
        'ready_for_gym': avg_level >= 15,
        'recommendation': 'Train more' if avg_level < 15 else 'Challenge gym'
    }
```

Register in `get_tool_definitions()`.

### Reward Function Integration

Modify `drl_env.py` to use objectives in reward calculation:

```python
def _calculate_reward_from_lightweight(self, prev, current):
    base_reward = ...
    
    # Check active objectives
    for obj in self.get_current_objectives():
        if obj.type == 'location':
            if self._is_near_target(obj.target):
                base_reward *= obj.reward_weight
    
    return base_reward
```

## 🎯 Example Workflow

Here's a complete training session:

```bash
# 1. Test LLM system
python test_hybrid_llm.py

# 2. Start training with hybrid mode
python train_ppo.py --hybrid --visualize --timesteps 50000

# 3. Monitor objectives being created
watch -n 5 'cat agent/current_objectives.json | jq .objectives'

# 4. Check tensorboard
tensorboard --logdir=./tensorboard_logs

# 5. Continue training from checkpoint
python train_ppo.py --hybrid --timesteps 50000 --model-path ./logs/checkpoints/ppo_pokemon_50000_steps.zip
```

## 🤝 Contributing

To extend this system:
1. Add new objective types in `objectives_manager.py`
2. Create new tools in `llm_tools.py`
3. Update reward function in `drl_env.py`
4. Test with `test_hybrid_llm.py`

## 📖 References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Ollama Function Calling](https://ollama.ai/blog/function-calling)
- [Curriculum Learning in RL](https://lilianweng.github.io/posts/2020-01-29-curriculum-rl/)

---

**Version**: 1.0  
**Last Updated**: November 2025  
**Maintained By**: Your Team
