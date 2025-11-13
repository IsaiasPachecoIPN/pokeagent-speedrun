# Reward System Architecture

## 📊 Complete Reward Flow

```
┌─────────────────────────────────────────────────────────────────┐
│              REWARD CALCULATION PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

Step 1: Base Rewards (Hard-coded)
═══════════════════════════════════════════════════════════════════

Location: agent/drl_env.py → _calculate_reward_from_lightweight()

  ┌─────────────────────────────────────┐
  │ Badge obtained:        +1000.0      │  ← Primary objective
  │ Level up (party):      +50.0        │  ← Secondary objective
  │ Movement:              +0.5         │  ← Exploration
  │ Stationary penalty:    -0.05/step   │  ← Anti-stuck
  │ Low HP (<20%):         -5.0         │  ← Survival
  │ Low HP (<50%):         -1.0         │  ← Health awareness
  └─────────────────────────────────────┘
              │
              ▼
         base_reward


Step 2: Objective-Based Rewards (LLM-Created) 🆕
═══════════════════════════════════════════════════════════════════

Location: agent/drl_env.py → _calculate_objective_rewards()

The LLM creates objectives, and rewards are given for progress:

  ┌──────────────────────────────────────────────────────────┐
  │ LOCATION OBJECTIVES                                       │
  │                                                           │
  │ Objective: "Explore Route 101"                           │
  │ Target: {"map": "ROUTE_101", "x": 10, "y": 5}           │
  │ Reward weight: 1.5                                       │
  │                                                           │
  │ Rewards:                                                  │
  │ • On target map:          +5.0 × weight                  │
  │ • Near target (dist<20):  +(20-dist)×0.5 × weight       │
  │ • At exact location:      +20.0 × weight                │
  └──────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────┐
  │ DIALOGUE OBJECTIVES                                       │
  │                                                           │
  │ Objective: "Talk to Professor Birch"                     │
  │ Target: {"npc": "PROF_BIRCH"}                           │
  │ Reward weight: 2.0                                       │
  │                                                           │
  │ Rewards:                                                  │
  │ • New dialogue detected:  +10.0 × weight                │
  │ • Target NPC dialogue:    +30.0 × weight                │
  └──────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────┐
  │ BATTLE OBJECTIVES                                         │
  │                                                           │
  │ Objective: "Battle wild Pokemon"                         │
  │ Reward weight: 1.2                                       │
  │                                                           │
  │ Rewards:                                                  │
  │ • In battle:             +5.0 × weight                   │
  └──────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────┐
  │ CUSTOM OBJECTIVES                                         │
  │                                                           │
  │ Objective: "Reach Pokemon Center"                       │
  │ Reward weight: 1.0                                       │
  │                                                           │
  │ Rewards:                                                  │
  │ • General progress:      +1.0 × weight                   │
  └──────────────────────────────────────────────────────────┘

              │
              ▼
    objective_reward = sum of all objective rewards


Step 3: Combine Base + Objective Rewards
═══════════════════════════════════════════════════════════════════

  reward = base_reward + objective_reward


Step 4: Apply Multipliers
═══════════════════════════════════════════════════════════════════

Location: agent/drl_env.py → _calculate_reward_from_lightweight()

  ┌─────────────────────────────────────────────────────┐
  │ LLM Multiplier (from old callback)                  │
  │ • Milestone progress:    1.8×                       │
  │ • Agent stuck:           0.3×                       │
  │ • Normal:                1.0×                       │
  └─────────────────────────────────────────────────────┘
              │
              ▼
  ┌─────────────────────────────────────────────────────┐
  │ Directional Multiplier (proximity-based)            │
  │ • Moving toward goals:   1.5×                       │
  │ • Moving away:           0.8×                       │
  │ • Normal:                1.0×                       │
  └─────────────────────────────────────────────────────┘
              │
              ▼
  combined_multiplier = llm_multiplier × directional_multiplier
  
  final_reward = reward × combined_multiplier


Final Result
═══════════════════════════════════════════════════════════════════

  TOTAL REWARD = (base_reward + objective_reward) × multipliers
```

## 🎯 Examples

### Example 1: Moving Toward LLM-Set Objective

**Scenario:**
- LLM created objective: "Explore Route 101" (weight: 1.5)
- Agent moves north toward Route 101
- Agent is at position (8, 12), target is (10, 5)

**Reward calculation:**
```python
# Step 1: Base rewards
base_reward = +0.5  # Movement

# Step 2: Objective rewards
objective_reward = 0
# - On Route 101 map: +5.0 × 1.5 = +7.5
# - Distance to target: |8-10| + |12-5| = 9
# - Proximity reward: (20-9)×0.5 × 1.5 = +8.25
objective_reward = 7.5 + 8.25 = 15.75

# Step 3: Combine
reward = 0.5 + 15.75 = 16.25

# Step 4: Apply multipliers
combined_multiplier = 1.0 × 1.0 = 1.0  # No special multipliers

final_reward = 16.25 × 1.0 = +16.25
```

**Result:** Agent gets **+16.25** reward for moving toward LLM objective!

---

### Example 2: Talking to NPC (LLM Objective)

**Scenario:**
- LLM created objective: "Talk to Professor Birch" (weight: 2.0)
- Agent presses A near Professor Birch
- Dialogue appears: "Hello! I'm Professor Birch!"

**Reward calculation:**
```python
# Step 1: Base rewards
base_reward = +0.5  # Movement (to reach NPC)

# Step 2: Objective rewards
objective_reward = 0
# - New dialogue detected: +10.0 × 2.0 = +20.0
# - Target NPC match ("PROF_BIRCH" in text): +30.0 × 2.0 = +60.0
objective_reward = 20.0 + 60.0 = 80.0

# Step 3: Combine
reward = 0.5 + 80.0 = 80.5

# Step 4: Apply multipliers
combined_multiplier = 1.0 × 1.0 = 1.0

final_reward = 80.5 × 1.0 = +80.5
```

**Result:** Agent gets **+80.5** reward for completing dialogue objective!

---

### Example 3: Agent Stuck (Negative Multiplier)

**Scenario:**
- No objectives active
- Agent pressing same button (stuck in corner)
- 30 steps without movement

**Reward calculation:**
```python
# Step 1: Base rewards
base_reward = -0.05 × min(30, 20) = -1.0  # Stationary penalty

# Step 2: Objective rewards
objective_reward = 0  # No objectives

# Step 3: Combine
reward = -1.0 + 0 = -1.0

# Step 4: Apply multipliers
llm_multiplier = 0.3  # LLM detected agent is stuck
combined_multiplier = 0.3 × 1.0 = 0.3

final_reward = -1.0 × 0.3 = -0.3
```

**Result:** Penalty is reduced (less harsh) to encourage trying new actions

---

## 📍 Code Locations

### Where Rewards Are Calculated

1. **Base Rewards:**
   ```
   File: agent/drl_env.py
   Function: _calculate_reward_from_lightweight()
   Lines: 508-560
   ```

2. **Objective Rewards:** 🆕
   ```
   File: agent/drl_env.py
   Function: _calculate_objective_rewards()
   Lines: 593-700 (newly added)
   ```

3. **Multiplier Application:**
   ```
   File: agent/drl_env.py
   Function: _calculate_reward_from_lightweight()
   Lines: 562-575
   ```

### Where Objectives Are Created

1. **By LLM via Tool Calling:**
   ```
   File: agent/llm_tools.py
   Function: write_objective()
   Lines: 67-104
   
   Called by: agent/hybrid_llm_callback.py
   During: LLM strategic analysis (every 5000 steps)
   ```

2. **Default Starter Objective:**
   ```
   File: agent/objectives_manager.py
   Function: _initialize_default()
   Lines: 115-130
   
   Creates: "Exit Starting Room" objective
   ```

### Where Policy Parameters Are Set

1. **By LLM via Tool Calling:**
   ```
   File: agent/llm_tools.py
   Function: update_policy()
   Lines: 149-171
   ```

2. **Stored in:**
   ```
   File: agent/llm_tools.py
   Variable: self.policy_params
   
   Note: Currently stored but NOT yet applied to environment!
   TODO: Need to pass these to environment's reward function
   ```

---

## 🔧 How LLM Influences Rewards

### Direct Influence (NEW - Objectives)

```python
# LLM creates objective at step 5000:
write_objective(
    name="Explore Route 101",
    type="location",
    target={"map": "ROUTE_101", "x": 10, "y": 5},
    reward_weight=1.5
)

# Saved to: agent/current_objectives.json

# Environment reads objectives at each step:
objectives = self.objectives_manager.get_active_objectives()

# Calculates objective rewards:
for objective in objectives:
    if agent_near_target:
        reward += base_value × objective.reward_weight
```

**Flow:**
```
LLM → write_objective() → JSON file → Environment reads → Reward calculation
```

### Indirect Influence (OLD - Multipliers)

```python
# LLM adjusts policy at step 5000:
update_policy(
    reward_weight_exploration=1.8
)

# Stored in: llm_tools.policy_params

# Applied as multiplier:
final_reward = base_reward × llm_multiplier
```

**Note:** Multipliers affect ALL rewards globally, objectives affect specific behaviors.

---

## 🎨 Reward Weight Guidelines

### For Objectives

- **Low priority (0.5-1.0):** Optional exploration, nice-to-have
- **Normal priority (1.0-1.5):** Standard objectives
- **High priority (1.5-2.5):** Critical path objectives
- **Very high priority (2.5-5.0):** Urgent or important milestones

### Examples

```python
# Low priority - explore optional area
Objective(
    name="Visit Pokemon Center",
    reward_weight=0.8
)

# Normal priority - standard progression
Objective(
    name="Talk to Mom",
    reward_weight=1.0
)

# High priority - main quest
Objective(
    name="Help Professor with Pokemon",
    reward_weight=2.0
)

# Very high priority - critical milestone
Objective(
    name="Get First Gym Badge",
    reward_weight=3.0
)
```

---

## 🚀 Future Enhancements

### TODO: Apply Policy Parameters

Currently, `update_policy()` stores parameters but doesn't apply them. Need to add:

```python
# In _calculate_objective_rewards():

# Get policy params from LLM tools
policy_params = self.llm_tools.policy_params if hasattr(self, 'llm_tools') else {}

# Apply to specific reward types
if obj_type == 'location':
    exploration_weight = policy_params.get('reward_weight_exploration', 1.0)
    objective_reward *= exploration_weight

if obj_type == 'dialogue':
    dialogue_weight = policy_params.get('reward_weight_dialogue', 1.0)
    objective_reward *= dialogue_weight
```

### TODO: Automatic Objective Completion

Detect when objectives are achieved and auto-complete:

```python
# In _calculate_objective_rewards():

if distance <= 2:  # Reached location
    objective_reward += 20.0 * weight
    self.objectives_manager.complete_objective(objective.id)
    logger.info(f"✅ Objective completed: {objective.name}")
```

### TODO: Objective Progress Tracking

Update progress percentage for visualization:

```python
if distance <= 10:
    progress = 1.0 - (distance / 10.0)
    self.objectives_manager.update_progress(objective.id, progress)
```

---

## 📊 Summary

**Question:** Where are rewards set?

**Answer:**

1. **Base rewards:** Hard-coded in `_calculate_reward_from_lightweight()` (badges, movement, HP)

2. **Objective rewards:** 🆕 Calculated in `_calculate_objective_rewards()` based on LLM-created objectives

3. **Multipliers:** Applied from LLM/directional callbacks (scale ALL rewards)

4. **Final formula:**
   ```
   TOTAL = (base_reward + objective_reward) × multipliers
   ```

The LLM creates **objectives** (via tool calling) which generate **additional rewards** when the agent makes progress toward them. This guides the DRL agent's learning toward strategic goals!
