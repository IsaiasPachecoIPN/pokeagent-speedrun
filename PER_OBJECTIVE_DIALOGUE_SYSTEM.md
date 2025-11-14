# Per-Objective Dialogue History System

## 🎯 Overview

The enhanced hybrid DRL+LLM system now tracks dialogue history **per objective**, allowing the LLM to make focused decisions based only on dialogues relevant to each specific objective.

## 🔑 Key Concept

**Each objective has its own `dialogue_history` field** that:
- Starts EMPTY when the objective is created
- Accumulates ONLY dialogues encountered while that objective is active
- Gets FROZEN when the objective is completed
- Provides FOCUSED CONTEXT to the LLM (no pollution from past objectives)

## 📋 How It Works

### 1. Objective Creation
```python
# When LLM creates a new objective
write_objective(
    name="Meet Professor Birch",
    description="Find and talk to Professor Birch",
    type="dialogue",
    target={"keywords": ["PROF. BIRCH", "WELCOME"], "any_match": True}
)
# This objective starts with dialogue_history = []
```

### 2. Dialogue Tracking
```python
# When agent encounters dialogue in game
add_dialogue("PROF. BIRCH: Welcome to Pokemon!")

# This dialogue is added to:
# - Global dialogue_history (for general context)
# - ALL ACTIVE objectives' dialogue_history (for per-objective context)
```

### 3. LLM Evaluation
```python
# LLM reads objectives
read_objectives()
# Returns:
# {
#   "active_objectives": [
#     {
#       "id": "meet_prof_birch",
#       "name": "Meet Professor Birch",
#       "dialogue_history": [
#         {"text": "PROF. BIRCH: Welcome to Pokemon!", "timestamp": "..."}
#       ]
#     }
#   ]
# }

# LLM evaluates if objective is complete
evaluate_objective_completion(
    objective_id="meet_prof_birch",
    reasoning="Dialogue contains 'PROF. BIRCH' and 'WELCOME', matching target keywords"
)

# If complete, LLM marks it done
complete_objective("meet_prof_birch")
```

### 4. Next Objective
```python
# LLM creates next objective based on story progression
write_objective(
    name="Choose Starter Pokemon",
    description="Select your first Pokemon",
    type="dialogue",
    target={"keywords": ["CHOOSE", "POKEMON", "STARTER"], "any_match": True}
)
# This NEW objective starts with dialogue_history = []
# It will NOT have dialogues from "meet_prof_birch"
```

## 🆕 New LLM Tools

### `evaluate_objective_completion(objective_id, reasoning)`
Analyzes an objective's dialogue_history to determine if it should be completed.

**Parameters:**
- `objective_id`: ID of objective to evaluate
- `reasoning`: LLM's analysis of why objective is/isn't complete

**Returns:**
```json
{
  "success": true,
  "objective_id": "meet_prof_birch",
  "objective_name": "Meet Professor Birch",
  "dialogue_history": [...],
  "dialogue_count": 2,
  "message": "Objective has 2 dialogues. Use complete_objective() if criteria are met."
}
```

### Enhanced `read_objectives()`
Now returns each objective WITH its dialogue_history field.

**Returns:**
```json
{
  "active_objectives": [
    {
      "id": "meet_prof_birch",
      "name": "Meet Professor Birch",
      "dialogue_history": [
        {"text": "...", "npc": "...", "location": "...", "timestamp": "..."}
      ],
      "target": {"keywords": ["PROF. BIRCH"], "any_match": true},
      ...
    }
  ],
  "completed_objectives": [...]
}
```

## 🎯 Recommended LLM Workflow

The LLM should follow this sequence:

### 1. Read Current State
```python
objectives = read_objectives()
# See all active objectives with their dialogue_history
```

### 2. Evaluate Each Active Objective
```python
for obj in objectives["active_objectives"]:
    # Analyze obj["dialogue_history"]
    evaluation = evaluate_objective_completion(
        objective_id=obj["id"],
        reasoning="Explain why complete/incomplete based on dialogue_history"
    )
    
    # If complete (based on target keywords/conditions)
    if completion_criteria_met:
        complete_objective(obj["id"])
```

### 3. Create New Objective (if needed)
```python
# If no active objectives OR last one just completed
if len(active_objectives) == 0:
    # Get general story context
    recent_context = read_dialogues(count=10)
    
    # Create next objective based on story progression
    write_objective(
        name="Next Story Objective",
        description="...",
        type="dialogue",
        target={"keywords": ["..."], "any_match": true}
    )
    # New objective starts with empty dialogue_history
```

## 📊 Benefits

### 1. **Focused Context**
- LLM only sees dialogues relevant to current objective
- No confusion from past objectives' dialogues

### 2. **Clear Completion Criteria**
- Each objective evaluated based on ITS dialogue_history
- No false positives from unrelated dialogues

### 3. **Story Progression**
- Objectives created sequentially based on story flow
- Each objective represents one story beat

### 4. **Isolation**
- Completed objectives' dialogues don't interfere with new ones
- Clean slate for each new objective

## 🔧 Technical Implementation

### ObjectivesManager Changes
```python
class Objective:
    def __init__(self, ..., dialogue_history=None):
        self.dialogue_history = dialogue_history or []  # Per-objective tracking

def add_dialogue(self, text, npc, location):
    # Add to global history
    self.dialogue_history.append(dialogue_entry)
    
    # Add to ALL active objectives
    for obj in self.get_active_objectives():
        obj.dialogue_history.append(dialogue_entry)
```

### LLMTools Changes
```python
class LLMTools:
    def evaluate_objective_completion(self, objective_id, reasoning):
        obj = self.objectives_manager.objectives[objective_id]
        return {
            "dialogue_history": obj.dialogue_history,  # Only THIS objective's dialogues
            "dialogue_count": len(obj.dialogue_history)
        }
```

## 🚀 Usage Example

```python
# Training with hybrid mode
python train_ppo.py --hybrid --timesteps 1000000

# The LLM will:
# 1. Start with default objective (explore_house)
# 2. Track dialogues: ["TV movie", "PC booted", "MOM: See you"]
# 3. Evaluate based on THESE 3 dialogues only
# 4. Complete objective when criteria met
# 5. Create new objective (meet_prof_birch) with EMPTY dialogue_history
# 6. Track new dialogues: ["PROF. BIRCH'S LAB", "Welcome to Pokemon"]
# 7. Evaluate based on THESE 2 new dialogues only
# 8. Continue this pattern throughout the game
```

## 🧪 Testing

Run the test script to see the system in action:
```bash
python test_per_objective_dialogues.py
```

This demonstrates:
- Creating objectives with empty dialogue_history
- Adding dialogues to active objectives only
- Isolation between objectives
- What the LLM sees when evaluating

## 📝 System Prompt Guidance

The LLM receives clear instructions:
- Each objective has ITS OWN dialogue_history
- New objectives start with NO previous dialogues
- Evaluate completion based on THAT objective's dialogue_history only
- Create objectives ONE AT A TIME for focused learning

This ensures the LLM makes intelligent, context-aware decisions about game progression!
