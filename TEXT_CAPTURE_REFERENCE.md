# 📝 Text Capture Reference

## What Gets Captured Now

The system now captures **ALL important game text** including location names, NPC dialogues, and story content.

### ✅ Captured Text Types

| Category | Examples | Why Important |
|----------|----------|---------------|
| **Location Names** | "Littleroot Town", "Route 101", "Oldale City" | LLM knows where agent is |
| **Places** | "Pokemon Lab", "Pokemon Center", "Poké Mart" | Identifies key locations |
| **NPCs** | "Professor Birch", "Gym Leader", "Prof. Oak" | Story characters |
| **Story Keywords** | "Welcome", "Help!", "Save", "Pokemon" | Game progression |

### ❌ Filtered Text (Not Captured)

| Category | Examples | Why Filtered |
|----------|----------|--------------|
| **TV Content** | "There is a movie on TV", "Two men are dancing" | Not relevant to objectives |
| **Furniture** | "It's a bookshelf", "It's a clock" | Environmental flavor |
| **Ambient** | "Took a closer look", "Examined", "Nothing here" | No strategic value |

---

## 📍 Location Detection Example

### Scenario: Leaving Starting Room

```
1. Agent is inside house
   → Game memory: No special text

2. Agent exits door
   → Screen shows: "LITTLEROOT TOWN"
   → System detects: "town" keyword
   → 📍 Captured as location text!

3. Saved to dialogue_history:
   {
     "text": "LITTLEROOT TOWN",
     "npc": null,
     "location": "LITTLEROOT_TOWN",
     "timestamp": "2025-11-13T11:58:00"
   }

4. LLM reads at next check (step 1000)
   → "Agent reached Littleroot Town"
   → Can complete "exit_starting_room" objective
   → Can create "explore_littleroot_town" objective
```

---

## 🔍 Detection Keywords

### Important Patterns (Always Captured)

**Locations:**
- `town`, `route`, `city`

**Buildings:**
- `lab`, `center`, `mart`

**NPCs:**
- `professor`, `prof.`, `gym`, `leader`

**Story:**
- `welcome`, `help`, `save`, `pokemon`, `poké`

### Example Matches:

| Text | Contains | Captured? |
|------|----------|-----------|
| "LITTLEROOT TOWN" | "town" | ✅ YES |
| "Welcome to Littleroot Town!" | "welcome", "town" | ✅ YES |
| "Route 101" | "route" | ✅ YES |
| "Prof. Birch's Pokemon Lab" | "prof.", "pokemon", "lab" | ✅ YES |
| "It's a bookshelf. There are books." | none | ❌ NO |
| "There is a movie on TV." | none | ❌ NO |

---

## 💾 Dialogue History Persistence

### OLD Behavior (Before Fix):
```
Training Run 1:
  - Start: 0 dialogues
  - Agent plays: captures 5 dialogues
  - End: 5 dialogues saved

Training Run 2:
  - Start: 0 dialogues ❌ (cleared!)
  - Agent plays: captures 3 new dialogues
  - End: 3 dialogues saved
```

### NEW Behavior (After Fix):
```
Training Run 1:
  - Start: 0 dialogues
  - Agent plays: captures 5 dialogues
  - End: 5 dialogues saved

Training Run 2:
  - Start: 5 dialogues ✅ (kept!)
  - Agent plays: captures 3 new dialogues
  - End: 8 dialogues saved (accumulated!)

Training Run 3:
  - Start: 8 dialogues ✅
  - Agent plays: captures 2 new dialogues
  - End: 10 dialogues saved
```

**Benefit:** LLM builds up story context over multiple runs!

---

## 🎯 How LLM Uses This Information

### Example 1: Exit Starting Room

**Captured Text:**
```json
[
  {
    "text": "LITTLEROOT TOWN",
    "location": "LITTLEROOT_TOWN",
    "timestamp": "2025-11-13T10:30:00"
  }
]
```

**LLM Analysis:**
```
Agent saw "LITTLEROOT TOWN" text
→ This appears when exiting the starting house
→ Objective "exit_starting_room" should be completed!
→ Create new objective: "explore_littleroot_town"
```

### Example 2: Find Pokemon Lab

**Captured Text:**
```json
[
  {
    "text": "LITTLEROOT TOWN",
    "location": "LITTLEROOT_TOWN"
  },
  {
    "text": "PROF. BIRCH'S POKéMON LAB",
    "location": "LITTLEROOT_TOWN"
  }
]
```

**LLM Analysis:**
```
Agent found the Pokemon Lab!
→ Create objective: "enter_pokemon_lab"
→ Increase reward weight for lab entrance
```

### Example 3: Route Discovery

**Captured Text:**
```json
[
  {
    "text": "ROUTE 101",
    "location": "ROUTE_101"
  },
  {
    "text": "Help! A wild POKéMON is chasing me!",
    "location": "ROUTE_101"
  }
]
```

**LLM Analysis:**
```
Agent reached Route 101 AND encountered story event!
→ Professor needs help on Route 101
→ Create high-priority objective: "help_professor_route_101" (weight: 3.0)
```

---

## 🔧 Configuration

### Change What Gets Captured

**File:** `agent/drl_env.py`  
**Method:** `_cache_dialog_if_present()`  
**Lines:** ~800-820

**To add more keywords:**
```python
important_patterns = [
    "town", "route", "city", "lab", "center", "mart",
    "professor", "prof.", "gym", "leader",
    "welcome", "help", "save", "pokemon", "poké",
    "your_new_keyword_here"  # Add here!
]
```

**To filter more text:**
```python
ambient_text_patterns = [
    "there is a movie on tv", "two men are dancing",
    # ... existing patterns ...
    "your_unwanted_pattern_here"  # Add here!
]
```

### Change Dialogue Persistence

**File:** `train_ppo.py`  
**Line:** ~298

**Keep dialogues (current):**
```python
objectives_manager.reset(keep_dialogues=True)
```

**Clear dialogues each run:**
```python
objectives_manager.reset(keep_dialogues=False)
```

---

## 📊 Monitoring Captures

### During Training

**Watch logs for:**
```
📍 [Step 125] NEW TEXT: 'LITTLEROOT TOWN'
💬 [Step 340] NEW TEXT: 'Welcome to Littleroot Town!'
📍 [Step 567] NEW TEXT: 'ROUTE 101'
💬 [Step 892] NEW TEXT: 'Help! A wild POKéMON...'
```

- `📍` = Location/place text
- `💬` = Dialogue/story text

### Check Captured Data

**View current dialogues:**
```bash
cat agent/current_objectives.json | jq .dialogue_history
```

**Count dialogues:**
```bash
cat agent/current_objectives.json | jq '.dialogue_history | length'
```

**Latest dialogue:**
```bash
cat agent/current_objectives.json | jq '.dialogue_history[-1]'
```

---

## 🧪 Testing

### Test Location Capture

1. Start training:
   ```bash
   python train_ppo.py --hybrid --timesteps 1000 --visualize
   ```

2. Watch agent exit house

3. Look for log:
   ```
   📍 [Step XXX] NEW TEXT: 'LITTLEROOT TOWN'
   ```

4. Check file:
   ```bash
   cat agent/current_objectives.json | jq '.dialogue_history[] | select(.text | contains("TOWN"))'
   ```

### Test Dialogue Persistence

1. Run training once:
   ```bash
   python train_ppo.py --hybrid --timesteps 500
   ```
   Note dialogue count.

2. Run again:
   ```bash
   python train_ppo.py --hybrid --timesteps 500
   ```
   Dialogue count should be same or higher (not reset to 0).

---

## 📝 Summary

**Key Changes:**
1. ✅ Location names like "Littleroot Town" now captured
2. ✅ Dialogue history persists across training runs
3. ✅ Important text prioritized over ambient descriptions
4. ✅ LLM has full story context to create objectives

**Result:**
- LLM knows when agent leaves house (sees "Littleroot Town")
- LLM knows when agent finds Pokemon Lab (sees "Lab" text)
- LLM knows when agent reaches routes (sees "Route 101")
- Objectives can be created based on actual game progress!

🎉 **Your hybrid system now has full awareness of agent location and story progression!**
