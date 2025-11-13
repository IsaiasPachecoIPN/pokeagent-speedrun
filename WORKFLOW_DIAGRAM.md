# Hybrid DRL+LLM Workflow Diagram

## Complete System Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HYBRID TRAINING SYSTEM                           │
└─────────────────────────────────────────────────────────────────────────┘

Step 0: Initialization
═══════════════════════════════════════════════════════════════════════════

  ┌──────────────────┐
  │  train_ppo.py    │
  │  --hybrid flag   │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────────────┐
  │ ObjectivesManager()      │
  │ Loads/Creates:           │
  │ current_objectives.json  │
  └────────┬─────────────────┘
           │
           │ Contains:
           │ - "Exit Starting Room" (default objective)
           │ - Empty dialogue history
           │
           ▼
  ┌──────────────────────────┐
  │ PokemonEmeraldEnv        │
  │ enable_hybrid_mode()     │
  │ - objectives_manager ✓   │
  │ - dialogue_capture ON    │
  └────────┬─────────────────┘
           │
           ▼
  ┌──────────────────────────┐
  │ Training begins...       │
  └──────────────────────────┘


Steps 1-5000: DRL Learning Phase
═══════════════════════════════════════════════════════════════════════════

  ┌──────────────────┐
  │ PPO Agent        │
  │ (Neural Network) │
  └────────┬─────────┘
           │
           │ Observes game state
           │ (map tiles, position, party info)
           ▼
  ┌──────────────────────────┐
  │ CNN Policy               │
  │ - Processes map          │
  │ - Processes vector data  │
  └────────┬─────────────────┘
           │
           │ Outputs action
           │ (A, B, arrows, etc.)
           ▼
  ┌──────────────────────────┐
  │ PokemonEmeraldEnv        │
  │ - Execute button press   │
  │ - Advance 36 frames      │
  │ - Capture dialogue       │◄──── Every 5 steps
  └────────┬─────────────────┘
           │
           │ Calculates reward
           │ based on:
           │ - Movement progress
           │ - Objective proximity
           │ - New areas explored
           ▼
  ┌──────────────────────────┐
  │ Reward → PPO Agent       │
  │ Agent learns policy      │
  └────────┬─────────────────┘
           │
           │ Dialogues captured to:
           │ current_objectives.json
           │ {"dialogue_history": [
           │   {"text": "Welcome!", ...}
           │ ]}
           │
           └──── Repeat 5000 times ────┐
                                        │
                                        ▼

Step 5000: LLM Strategic Analysis
═══════════════════════════════════════════════════════════════════════════

  ┌──────────────────────────┐
  │ HybridLLMCallback        │
  │ Triggered at step 5000   │
  └────────┬─────────────────┘
           │
           │ Prepares context:
           │ - Training stats (avg reward, episode length)
           │ - Current objectives
           │ - Dialogue history
           │
           ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ LLM (Ollama - qwen3:8b)                                     │
  │                                                                │
  │ System Prompt:                                                │
  │ "You are an AI strategic planner. DRL handles actions,       │
  │  YOU handle objectives. Use tools to guide training."         │
  │                                                                │
  │ User Message:                                                 │
  │ "Analyze progress:                                            │
  │  - Steps: 5000                                                │
  │  - Avg Reward: 12.5                                           │
  │  - Objectives: [Exit Starting Room]                           │
  │  - Dialogues: [Welcome! Go to Route 101!]"                   │
  └────────┬─────────────────────────────────────────────────────┘
           │
           │ LLM Tool Calling Loop
           │
           ▼
  ┌────────────────────────────────────────┐
  │ Iteration 1: read_objectives()          │
  │                                         │
  │ LLM → Calls read_objectives()          │
  │       ↓                                 │
  │ LLMTools → Returns:                     │
  │ {                                       │
  │   "active_objectives": [                │
  │     {"name": "Exit Starting Room", ...}│
  │   ],                                    │
  │   "completed_objectives": []            │
  │ }                                       │
  └────────┬────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────┐
  │ Iteration 2: read_dialogues(count=10)  │
  │                                         │
  │ LLM → Calls read_dialogues()           │
  │       ↓                                 │
  │ LLMTools → Returns:                     │
  │ {                                       │
  │   "dialogues": [                        │
  │     {                                   │
  │       "text": "Go to Route 101!",      │
  │       "npc": "PROF_BIRCH",             │
  │       "location": "LITTLEROOT_TOWN"    │
  │     }                                   │
  │   ]                                     │
  │ }                                       │
  └────────┬────────────────────────────────┘
           │
           │ LLM analyzes:
           │ "The dialogue mentions Route 101.
           │  I should create an objective to explore there."
           │
           ▼
  ┌────────────────────────────────────────┐
  │ Iteration 3: write_objective()         │
  │                                         │
  │ LLM → Calls write_objective(           │
  │         name="Explore Route 101",      │
  │         description="Travel north...", │
  │         type="location",               │
  │         target={"map": "ROUTE_101"},   │
  │         reward_weight=1.5              │
  │       )                                 │
  │       ↓                                 │
  │ LLMTools → ObjectivesManager           │
  │           → Saves to JSON              │
  │           → Returns {"success": true}  │
  └────────┬────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────┐
  │ Iteration 4: update_policy()           │
  │                                         │
  │ LLM → "Agent should explore more"      │
  │     → Calls update_policy(             │
  │         reward_weight_exploration=1.8  │
  │       )                                 │
  │       ↓                                 │
  │ LLMTools → Updates policy params       │
  │           → Returns {"success": true}  │
  └────────┬────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────┐
  │ Iteration 5: Final Response            │
  │                                         │
  │ LLM → "I've analyzed the dialogues     │
  │        and created an objective to     │
  │        explore Route 101. Increased    │
  │        exploration rewards."           │
  └────────┬────────────────────────────────┘
           │
           │ Updates saved:
           │ current_objectives.json now has:
           │ - "Exit Starting Room" (existing)
           │ - "Explore Route 101" (NEW)
           │
           ▼
  ┌──────────────────────────┐
  │ Training continues...    │
  └──────────────────────────┘


Steps 5001-10000: DRL with Updated Objectives
═══════════════════════════════════════════════════════════════════════════

  ┌──────────────────────────┐
  │ PPO Agent continues       │
  │ BUT NOW:                  │
  │                           │
  │ Reward function considers:│
  │ 1. Exit Starting Room     │
  │ 2. Explore Route 101 ✨   │
  │                           │
  │ Exploration weight: 1.8✨ │
  └────────┬─────────────────┘
           │
           │ Agent behavior changes:
           │ - More likely to explore north
           │ - Higher rewards for new areas
           │ - Guided toward Route 101
           │
           └──── Repeat 5000 times ────┐
                                        │
                                        ▼

Step 10000: Second LLM Analysis
═══════════════════════════════════════════════════════════════════════════

  ┌──────────────────────────┐
  │ HybridLLMCallback        │
  │ Triggered again          │
  └────────┬─────────────────┘
           │
           │ New context:
           │ - Steps: 10000
           │ - Objectives: 2 active
           │ - New dialogues: [Help me with wild Pokemon!]
           │
           ▼
  ┌────────────────────────────────────────┐
  │ LLM Analyzes Again                     │
  │                                         │
  │ - Reads objectives (2 now)             │
  │ - Reads NEW dialogues                  │
  │ - Sees: "Help with wild Pokemon"       │
  │ - Creates: "Catch First Pokemon"       │
  │ - Increases: dialogue interaction      │
  └────────┬────────────────────────────────┘
           │
           │ New objective added:
           │ - "Catch First Pokemon"
           │
           ▼
  ┌──────────────────────────┐
  │ Training continues...    │
  │ Now with 3 objectives    │
  └──────────────────────────┘


Continuous Cycle
═══════════════════════════════════════════════════════════════════════════

  Every 5000 steps:
  
  DRL Phase → LLM Analysis → Updated Objectives → DRL Phase → ...
  
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │ Agent       │────▶│ LLM         │────▶│ New         │
  │ Learns      │     │ Evaluates   │     │ Objectives  │
  │ Actions     │     │ & Creates   │     │ & Policies  │
  └─────────────┘     └─────────────┘     └─────────────┘
         ▲                                        │
         │                                        │
         └────────────────────────────────────────┘
                    Feedback Loop


File Interactions
═══════════════════════════════════════════════════════════════════════════

  current_objectives.json
  ═══════════════════════
  
  Read by:                    Written by:
  - LLM (read_objectives)     - LLM (write_objective)
  - DRL Env (reward calc)     - Env (dialogue capture)
  
  Contains:
  {
    "objectives": {
      "exit_starting_room": {...},
      "explore_route_101": {...},
      "catch_first_pokemon": {...}
    },
    "dialogue_history": [
      {"text": "...", "npc": "...", "location": "..."},
      ...
    ]
  }


Data Flow Summary
═══════════════════════════════════════════════════════════════════════════

  Game State
  (Map, Position, Party)
        │
        ▼
  ┌─────────────┐
  │ DRL Agent   │ ──────────────┐
  │ (PPO)       │               │
  └─────┬───────┘               │
        │                        │
        │ Actions                │ Dialogues
        │ (Buttons)              │
        ▼                        ▼
  ┌──────────────────────────────────┐
  │ Environment                       │
  │ - Execute actions                 │
  │ - Calculate rewards               │
  │ - Capture dialogues               │
  │ - Check objective progress        │
  └──────┬───────────────────────────┘
         │
         │ Every 5000 steps
         │
         ▼
  ┌──────────────────────┐
  │ Objectives File      │
  │ (JSON)               │
  └──────┬───────────────┘
         │
         │ Read by LLM
         │
         ▼
  ┌──────────────────────┐
  │ LLM Strategist       │
  │ - Analyze progress   │
  │ - Read dialogues     │
  │ - Create objectives  │
  │ - Adjust policies    │
  └──────┬───────────────┘
         │
         │ Write new objectives
         │
         └───────► Back to Objectives File
                   Back to Environment
                   Back to DRL Agent


Key Insight
═══════════════════════════════════════════════════════════════════════════

  DRL Agent:              LLM Strategist:
  "How do I move?"        "Where should we go?"
  "Which button?"         "What's the goal?"
  "What happens next?"    "What's the strategy?"
  
  Low-level control   ←───┼───►   High-level planning
  Fast reactions      ←───┼───►   Strategic decisions
  Learn patterns      ←───┼───►   Understand context
  
  Together: Intelligent, goal-directed gameplay
```

## Tool Calling Detail

```
LLM Tool Calling Mechanism
══════════════════════════════════════════════════════════════

Initial Call:
┌─────────────────────────────────────────────────────┐
│ ollama.chat(                                         │
│   model="qwen3:8b",                               │
│   messages=[system_prompt, user_message],           │
│   tools=[read_objectives, write_objective, ...]     │
│ )                                                    │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ LLM Response:                                        │
│ {                                                    │
│   "message": {                                       │
│     "tool_calls": [                                  │
│       {                                              │
│         "function": {                                │
│           "name": "read_objectives",                 │
│           "arguments": {}                            │
│         }                                            │
│       }                                              │
│     ]                                                │
│   }                                                  │
│ }                                                    │
└─────────────────────────────────────────────────────┘
         │
         ▼
Execute Tool:
┌─────────────────────────────────────────────────────┐
│ llm_tools.read_objectives()                          │
│ → Returns objectives dict                            │
└─────────────────────────────────────────────────────┘
         │
         ▼
Add to Conversation:
┌─────────────────────────────────────────────────────┐
│ messages.append({                                    │
│   "role": "tool",                                    │
│   "content": json.dumps(tool_result)                 │
│ })                                                   │
└─────────────────────────────────────────────────────┘
         │
         ▼
Next LLM Call:
┌─────────────────────────────────────────────────────┐
│ ollama.chat(                                         │
│   model="qwen3:8b",                               │
│   messages=[...previous messages, tool_result],     │
│   tools=[...]                                        │
│ )                                                    │
└─────────────────────────────────────────────────────┘
         │
         ▼
Repeat until LLM provides final content
(not a tool call)
```

## Summary Flow

```
╔═══════════════════════════════════════════════════════════╗
║  START: python train_ppo.py --hybrid                      ║
╚═══════════════════════════════════════════════════════════╝
                         │
                         ▼
           ┌─────────────────────────┐
           │ Initialize System       │
           │ - Objectives Manager    │
           │ - DRL Environment       │
           │ - PPO Agent             │
           └─────────────────────────┘
                         │
                         ▼
           ┌─────────────────────────┐
           │ Phase 1: DRL Training   │
           │ 5000 steps              │
           │ Capture dialogues       │
           └─────────────────────────┘
                         │
                         ▼
           ┌─────────────────────────┐
           │ Phase 2: LLM Analysis   │
           │ - Read objectives       │
           │ - Read dialogues        │
           │ - Create new objectives │
           │ - Adjust policies       │
           └─────────────────────────┘
                         │
                         ▼
           ┌─────────────────────────┐
           │ Phase 3: DRL Training   │
           │ 5000 steps              │
           │ With updated objectives │
           └─────────────────────────┘
                         │
                         ▼
           ┌─────────────────────────┐
           │ Phase 4: LLM Analysis   │
           │ Repeat cycle...         │
           └─────────────────────────┘
                         │
                         ▼
                      [Loop]
```

---

This diagram illustrates the complete workflow of the Hybrid DRL+LLM system, from initialization through the continuous learning cycle.
