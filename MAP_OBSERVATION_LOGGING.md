# Map Observation Logging

## Overview

The LLM callback now captures and logs the **7x7 CNN map observation** that the agent sees at the time of each LLM call. This allows you to verify exactly what spatial context the LLM had when making decisions.

## What Gets Logged

Each LLM log file now includes a map_observation field with:

### 1. Raw Map Data
`json
{
  "map_array": {
    "shape": [7, 7, 3],
    "channels": {
      "metatile_id": [[...], ...],   // Channel 0: Visual appearance
      "collision": [[...], ...],      // Channel 1: Collision data (walls, water)
      "behavior": [[...], ...]        // Channel 2: Behavior (doors, NPCs, exits)
    },
    "center_position": [3, 3],
    "description": "7x7 grid centered on player - [y][x] indexed"
  }
}
`

### 2. Spatial Analysis
`json
{
  "spatial_analysis": {
    "nearby_doors": [["north", 2.0]],
    "nearby_npcs": [["east", 1.0]],
    "nearby_exits": [],
    "closest_door": ["north", 2.0],
    "obstacles": {
      "north": false,
      "south": false,
      "east": false,
      "west": false
    },
    "passable_directions": ["north", "south", "east", "west"],
    "in_grass": false,
    "near_water": false,
    "num_obstacles": 0,
    "num_passable": 4
  }
}
`

### 3. ASCII Visualization
Easy-to-read text representation:
`
7x7 Map Observation (Player at center 'P'):
===========================================
| #  #  #  #  #  #  # |
| .  .  .  D  .  .  . |
| .  .  .  .  .  .  . |
| .  .  .  P  N  .  . |
| .  .  .  .  .  .  . |
| .  .  .  .  .  .  . |
| #  #  #  #  #  #  # |
===========================================
Legend: P=Player D=Door N=NPC E=Exit G=Grass #=Wall ~=Water .=Passable

Spatial Analysis:
   Doors: [('north', 2.0)]
   NPCs: [('east', 1.0)]
   Can move: north, south, east, west
`

## Viewing Map Observations

### Option 1: Use view_llm_logs.py (Enhanced)

`ash
python view_llm_logs.py
`

The script now displays:
- ASCII map visualization
- Spatial analysis summary
- Nearby features (doors, NPCs, exits)
- Passable directions

### Option 2: Read JSON Directly

`python
import json

with open('logs/llm_responses/llm_call_001_step_5000_20251113_143022.json', 'r') as f:
    log = json.load(f)

# Access map data
map_obs = log['map_observation']
print(map_obs['ascii_visualization'])

# Check spatial context
spatial = map_obs['spatial_analysis']
print(f"Doors nearby: {spatial['nearby_doors']}")
print(f"NPCs nearby: {spatial['nearby_npcs']}")
`

## Use Cases

### 1. Verify LLM Spatial Awareness
Check if the LLM is seeing the doors/NPCs it should be guiding the agent toward:
`ash
python view_llm_logs.py
# Look for: " Map Observation" section
# Verify: Door positions match objective targets
`

### 2. Debug Objective Completion
If an objective isn't completing, check if the agent is actually at the target location:
`python
# Check if agent is standing at a door
spatial = log['map_observation']['spatial_analysis']
if spatial['closest_door'] and spatial['closest_door'][1] < 1.0:
    print("Agent is at the door!")
`

### 3. Analyze Training Progress
Compare map observations across multiple LLM calls to see how agent explores:
`ash
# List all logs
python view_llm_logs.py --list

# View specific log
python view_llm_logs.py --call 5
`

### 4. Verify Proximity Rewards
Check if spatial bonuses are being applied correctly:
`python
spatial = log['map_observation']['spatial_analysis']
doors = spatial['nearby_doors']

if doors:
    direction, distance = doors[0]
    if distance < 2.0:
        print(f"Proximity bonus should be active! Door is {distance:.1f} tiles away")
`

## Log Size

- **ASCII visualization:** ~500 bytes
- **Spatial analysis:** ~300 bytes  
- **Full map arrays:** ~3-4 KB (JSON serialized)
- **Total per log:** +4-5 KB overhead

With default settings (LLM call every 5000 steps), this adds minimal storage overhead.

## Example Log Entry

`json
{
  "call_number": 3,
  "timestamp": "2025-11-13T14:30:22.123456",
  "training_step": 15000,
  "map_observation": {
    "map_array": {
      "shape": [7, 7, 3],
      "channels": {...},
      "center_position": [3, 3]
    },
    "spatial_analysis": {
      "nearby_doors": [["north", 2.0]],
      "nearby_npcs": [["east", 1.0]],
      "passable_directions": ["north", "south", "east", "west"],
      "num_passable": 4
    },
    "ascii_visualization": "7x7 Map Observation...",
    "timestamp": "2025-11-13T14:30:22.123456"
  },
  "game_state_summary": "...",
  "spatial_context": " Location: LITTLEROOT_TOWN...",
  "tool_calls_summary": [...],
  "final_response": "..."
}
`

## Training Console Output

When LLM is called, you'll now see:
`
 LLM conversation saved to: logs/llm_responses/llm_call_003_step_15000_20251113_143022.json
   Tools used: ['read_objectives', 'write_objective']
    Map captured - Doors: 1, NPCs: 1, Passable: 4/4
    Rewards - Avg (last 10): 125.450, Trend:  IMPROVING (+12.3%)
    Episodes: 45 total, 8 since last check
`

## Benefits

1. **Debugging:** See exactly what the agent sees
2. **Verification:** Confirm LLM has accurate spatial context
3. **Analysis:** Track exploration patterns over time
4. **Transparency:** Complete observability of the system

## Next Steps

1. Train with: python train_ppo.py --hybrid
2. Check logs: python view_llm_logs.py
3. Verify map observations match objectives
4. Analyze spatial patterns

---

**Status:**  Implemented and tested
**Last Updated:** 2025-11-13
