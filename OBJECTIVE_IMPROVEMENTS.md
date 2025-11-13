# Objective System Improvements

## Changes Made (November 13, 2025)

### Problem Identified
After analyzing 5000 steps of captured text (68 total, 21 unique), we found:
- The default objective "Exit Starting Room" was looking for map name "LITTLEROOT_TOWN"
- But the game actually shows SIGN TEXT like "JOHNNY''s HOUSE", "PROF. BIRCH''S HOUSE"
- The objective was too vague and hard to complete

### Solution: Smarter Starting Objectives

#### New Default Objectives:

**1. Explore Starting House** (reward_weight: 1.5)
- More achievable first objective
- Checks for keywords: PC, TV, HOUSE, upstairs, MOM
- Requires 3 interactions to complete
- Encourages agent to explore before leaving

**2. Meet Professor Birch** (reward_weight: 2.0)
- Based on Mom''s dialogue hint
- Keywords: PROF. BIRCH, BIRCH''S HOUSE, BIRCH''S LAB, POKéMON LAB
- Completes when ANY keyword is found
- Aligns with game story progression

### Auto-Completion System

Added `_check_dialogue_objectives()` method that:
-  Automatically checks objectives when new dialogue is captured
-  Supports keyword-based objectives
-  Tracks progress (e.g., 2/3 interactions complete)
-  Auto-completes when conditions are met

### Improved Text Capture

Enhanced `important_patterns` in `drl_env.py`:
- Added: "house", "birch", "rival", "mom", "dad", "dangerous", "wild", "grass"
- Now captures location signs correctly
- Better recognition of story-important NPCs and warnings

### Expected Results

With these changes:
1. **First objective completes faster** - Agent explores house, sees PC/TV texts
2. **Second objective is story-aligned** - Agent goes to Prof Birch as Mom suggested
3. **LLM gets better context** - More relevant dialogues captured
4. **Progress is visible** - Auto-tracking shows 1/3, 2/3, 3/3 progress

### Testing

To test the improvements:
```bash
# Delete old objectives file to get new defaults
rm agent/current_objectives.json

# Run training with hybrid mode
python train_ppo.py --hybrid --timesteps 10000 --visualize

# Watch for:
# - " Objective completed by dialogue" messages
# - Progress updates (1/3, 2/3, etc.)
# - Auto-completion when keywords are found
```

### Files Modified
- `agent/objectives_manager.py` - New default objectives + auto-check system
- `agent/drl_env.py` - Improved important_patterns for text capture

