# Quick Reference: LLM Objective System with Per-Objective Dialogues

## 🎯 What Changed?

### Before:
- All dialogues in one global history
- LLM saw ALL past dialogues when evaluating ANY objective
- Context pollution from previous objectives

### After:
- Each objective has its OWN dialogue_history
- LLM sees ONLY dialogues relevant to that objective
- Clean, focused context for decision-making

## 🔧 New Tool: `evaluate_objective_completion`

Use this to check if an objective's completion criteria are met:

```python
# LLM workflow
objectives = read_objectives()  # Get objectives with their dialogue_history

for obj in objectives['active_objectives']:
    # Analyze obj['dialogue_history'] - only dialogues for THIS objective
    
    # Evaluate completion
    result = evaluate_objective_completion(
        objective_id=obj['id'],
        reasoning="Found 'PROF. BIRCH' and 'WELCOME' in dialogues, matching target keywords"
    )
    
    # If criteria met, complete it
    if criteria_met:
        complete_objective(obj['id'])
```

## 📋 Objective Structure (Enhanced)

```json
{
  "id": "meet_prof_birch",
  "name": "Meet Professor Birch",
  "description": "Find and talk to Prof. Birch",
  "type": "dialogue",
  "target": {
    "keywords": ["PROF. BIRCH", "WELCOME"],
    "any_match": true
  },
  "dialogue_history": [
    {
      "text": "PROF. BIRCH'S POKéMON LAB",
      "location": "Outside Lab",
      "timestamp": "2025-11-13T14:30:00"
    },
    {
      "text": "PROF. BIRCH: Welcome to the world of Pokemon!",
      "npc": "PROF. BIRCH",
      "location": "Lab",
      "timestamp": "2025-11-13T14:30:15"
    }
  ],
  "created_at": "2025-11-13T14:29:00",
  "completed": false
}
```

## 🤖 LLM Workflow (Step-by-Step)

### 1. Read Current Objectives
```python
objectives = read_objectives()
# Returns active and completed objectives WITH their dialogue_history
```

### 2. Evaluate Each Active Objective
```python
for obj in objectives['active_objectives']:
    # Look at obj['dialogue_history'] - only dialogues for THIS objective
    
    # Check if target keywords/conditions are met
    if matches_target_criteria(obj):
        evaluate_objective_completion(
            objective_id=obj['id'],
            reasoning="Explain why complete based on dialogue_history"
        )
        
        complete_objective(obj['id'])
```

### 3. Create New Objective (if needed)
```python
# When no active objectives or last one completed
if no_active_objectives:
    # Get general story context
    context = read_dialogues(count=10)
    
    # Create next objective
    write_objective(
        name="Next Story Beat",
        description="What should happen next in the game",
        type="dialogue",
        target={"keywords": ["KEY", "WORDS"], "any_match": true}
    )
    # This starts with dialogue_history = []
```

## 📊 Example Flow

```
Step 1: Objective "explore_house" created
        dialogue_history: []

Step 2: Dialogues encountered:
        - "There is a movie on TV."
        - "JOHNNY booted up the PC."
        - "MOM: See you, honey!"
        dialogue_history: [TV, PC, MOM] (3 dialogues)

Step 3: LLM evaluates:
        evaluate_objective_completion("explore_house")
        Found 3 interactions → COMPLETE!

Step 4: complete_objective("explore_house")
        dialogue_history: [TV, PC, MOM] (frozen)

Step 5: New objective "meet_prof_birch" created
        dialogue_history: [] (fresh start!)

Step 6: New dialogues encountered:
        - "PROF. BIRCH'S POKéMON LAB"
        - "PROF. BIRCH: Welcome!"
        dialogue_history: [LAB, BIRCH] (2 NEW dialogues)

Step 7: LLM evaluates:
        Only sees [LAB, BIRCH] - not [TV, PC, MOM]
        Found "PROF. BIRCH" → COMPLETE!
```

## 🎯 Key Benefits

1. **Focused Context**: Each objective evaluated on its own merits
2. **No Pollution**: Past dialogues don't interfere with current decisions
3. **Sequential Story**: Each objective = one story beat
4. **Clear Criteria**: Dialogue history shows exact progress toward goal
5. **LLM Clarity**: Model sees only relevant context for each decision

## 🚀 Training Command

```bash
# Start training with hybrid mode
python train_ppo.py --hybrid --timesteps 1000000 --n-envs 1
```

The LLM will:
- Create objectives one at a time
- Track dialogues per objective
- Evaluate completion based on focused context
- Generate new objectives based on story progression

## 📁 Files Modified

- `agent/objectives_manager.py` - Added `dialogue_history` to Objective class
- `agent/llm_tools.py` - Added `evaluate_objective_completion` tool
- `agent/hybrid_llm_callback.py` - Updated system prompt and tool execution
- `PER_OBJECTIVE_DIALOGUE_SYSTEM.md` - Full documentation
- `test_dialogue_system_simple.py` - Demo script

## 🧪 Test It

```bash
python test_dialogue_system_simple.py
```

This shows exactly how dialogue isolation works!
