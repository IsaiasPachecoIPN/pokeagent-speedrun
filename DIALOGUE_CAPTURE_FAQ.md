# Dialogue Capture FAQ

## ❓ Why is `dialogue_history` Empty?

### Quick Answer

**The dialogue history is empty because dialogues are only captured DURING training when the agent actively plays the game.**

If you see:
```json
{
  "dialogue_history": [],
  "last_updated": "2025-11-13T11:15:08.893091"
}
```

This means:
- ✅ The file structure is correct
- ✅ The system is ready to capture dialogues
- ⏳ **But training hasn't encountered any dialogues yet**

---

## 🔄 When Dialogue Capture Happens

### The Flow:

```
1. Start training with --hybrid flag
   ↓
2. Environment initialized
   ↓
3. enable_hybrid_mode() called
   ↓
4. Agent starts playing the game
   ↓
5. Agent presses buttons, moves around
   ↓
6. Agent encounters NPC or dialogue
   ↓
7. 📝 Dialogue detected in memory
   ↓
8. 💾 Saved to current_objectives.json
   ↓
9. LLM reads it at next check (every 5000 steps)
```

### Code Location:

**File:** `agent/drl_env.py`  
**Function:** `step()` (line ~268)

```python
# At each step:
if self.enable_dialogue_capture and self.objectives_manager:
    dialog = self._get_current_dialog()
    if dialog and dialog != self.last_dialog:
        # NEW DIALOGUE! Save it
        self.objectives_manager.add_dialogue(dialog, location=location)
```

---

## 🎯 When Will Dialogues Appear?

### Scenario 1: During Training

```bash
python train_ppo.py --hybrid --timesteps 10000
```

**Timeline:**
- **Steps 0-100:** Agent learning to move, no dialogues yet
- **Steps 100-500:** Agent explores, may encounter signs or objects
- **Steps 500-2000:** Agent finds NPCs, dialogues appear! ✅
- **Step 5000:** LLM reads dialogues, creates objectives

**Expected in `current_objectives.json`:**
```json
{
  "dialogue_history": [
    {
      "text": "Welcome! This is Littleroot Town!",
      "npc": null,
      "location": "LITTLEROOT_TOWN",
      "timestamp": "2025-11-13T11:20:15.123456"
    },
    {
      "text": "Hello! I'm Professor Birch!",
      "npc": null,
      "location": "LITTLEROOT_TOWN",
      "timestamp": "2025-11-13T11:21:30.789012"
    }
  ]
}
```

### Scenario 2: Test Script

```bash
python test_dialogue_capture.py
```

This runs the agent for 100 steps and checks for dialogues.

---

## 🔍 How to Verify It's Working

### Method 1: Run Training and Watch

```bash
python train_ppo.py --hybrid --visualize --timesteps 5000
```

**Watch for:**
```
📝 Captured dialogue at LITTLEROOT_TOWN: 'Welcome to Littleroot Town! This quiet town...'
```

### Method 2: Check File During Training

**Terminal 1:**
```bash
python train_ppo.py --hybrid --timesteps 10000
```

**Terminal 2 (while training):**
```powershell
# Watch file update in real-time
while ($true) {
    Clear-Host
    Get-Content agent/current_objectives.json | ConvertFrom-Json | 
        Select-Object -ExpandProperty dialogue_history | 
        Format-Table -AutoSize
    Start-Sleep 5
}
```

### Method 3: Run Diagnostic Test

```bash
python test_dialogue_capture.py
```

Expected output:
```
✅ SUCCESS - Dialogues are being captured!

Recent dialogues:
  1. [LITTLEROOT_TOWN] Welcome to Littleroot Town!...
  2. [ROUTE_101] Wild POOCHYENA appeared!...
```

---

## 🐛 Troubleshooting

### Issue 1: Still Empty After 1000 Steps

**Possible Cause:** Agent hasn't found NPCs/dialogues yet

**Solution:**
- Let training run longer (5000+ steps)
- Agent needs time to explore and find NPCs
- Starting position may be far from dialogues

### Issue 2: `enable_dialogue_capture = False`

**Check:**
```python
# In train_ppo.py, verify hybrid mode is enabled:
if hybrid_mode:
    env.env_method('enable_hybrid_mode', objectives_manager, indices=[i])
```

**Fix:** Make sure you used `--hybrid` flag

### Issue 3: No Dialogue Detection

**Check game state file:**
```bash
# Your save state might not have active dialogues
# Try different save states:
python train_ppo.py --hybrid --state Emerald-GBAdvance/start.state
```

### Issue 4: Memory Reading Not Working

**Check logs for:**
```
Error caching dialog: ...
Could not get dialog: ...
```

**Solution:** Check emulator memory reader is working

---

## 📊 Expected Behavior Over Time

### Training Progress:

```
Steps 0-1000:
dialogue_history: []
→ Agent exploring, learning basic movement

Steps 1000-5000:
dialogue_history: [1-3 entries]
→ Agent found some NPCs or signs

Steps 5000-10000:
dialogue_history: [5-10 entries]
→ More exploration, more dialogues

Step 5000 (First LLM Check):
→ LLM reads 5 dialogues
→ Creates objectives based on context
→ "Talk to Professor Birch", "Explore Route 101", etc.
```

---

## 🎓 Understanding the System

### Why Dialogues Matter

The LLM uses dialogues to understand story context:

1. **Agent plays** → Encounters dialogue
2. **Dialogue captured** → Saved to JSON
3. **LLM reads** → Understands what NPC said
4. **LLM creates objectives** → Based on story context

**Example:**

```
Dialogue: "Help! A wild Pokémon is chasing me on Route 101!"
         ↓
LLM Analysis: "The professor needs help on Route 101"
         ↓
New Objective: "Go to Route 101 and help Professor"
         ↓
Reward Weight: 2.5 (high priority!)
```

---

## ✅ Verification Checklist

To ensure dialogue capture will work:

- [ ] **Training started with `--hybrid` flag**
  ```bash
  python train_ppo.py --hybrid
  ```

- [ ] **Hybrid mode enabled in code**
  ```python
  env.enable_hybrid_mode(objectives_manager)
  ```

- [ ] **Training is running** (not just file creation)

- [ ] **Agent has had time to explore** (1000+ steps minimum)

- [ ] **Game state has reachable NPCs/dialogues**

- [ ] **Memory reading is working** (no errors in logs)

---

## 🚀 Quick Test

Run this to verify everything works:

```bash
# 1. Run diagnostic
python test_dialogue_capture.py

# 2. Start short training
python train_ppo.py --hybrid --timesteps 5000 --visualize

# 3. Watch for dialogue capture logs
# Look for: "📝 Captured dialogue at..."

# 4. Check file after training
cat agent/current_objectives.json | jq .dialogue_history
```

---

## 💡 Key Insight

**The empty dialogue history is NORMAL and EXPECTED if:**

1. ✅ You just created the file (via setup/test scripts)
2. ✅ Training hasn't started yet
3. ✅ Training just started (< 500 steps)
4. ✅ Agent hasn't explored enough yet

**It will populate automatically during training as the agent encounters dialogues in the game!**

---

## 📝 Example Timeline

### Real Training Session:

```
11:00:00 - Training starts
11:00:01 - dialogue_history: [] (empty)

11:05:23 - Agent finds sign
11:05:23 - dialogue_history: [1] ✅ "Welcome to Littleroot Town!"

11:08:45 - Agent talks to Mom
11:08:45 - dialogue_history: [2] ✅ "Take care, dear!"

11:12:10 - Agent reaches Route 101
11:12:10 - dialogue_history: [3] ✅ "Wild POOCHYENA appeared!"

11:15:00 - Step 5000 reached
11:15:01 - LLM Analysis triggered
11:15:02 - LLM reads 3 dialogues
11:15:05 - LLM creates new objectives:
           - "Catch first Pokémon"
           - "Return to town"

11:20:00 - Training continues with new objectives
```

---

## 🎯 Summary

**Q:** Why is dialogue_history empty?

**A:** Because dialogues are captured **during training** when the agent plays. If training hasn't started or the agent hasn't encountered dialogues yet, the history will be empty. **This is normal!**

**To populate it:**
1. Start training with `--hybrid`
2. Wait for agent to explore (1000+ steps)
3. Dialogues will appear automatically
4. LLM will read them at step 5000

**To verify it works:**
```bash
python test_dialogue_capture.py
```
