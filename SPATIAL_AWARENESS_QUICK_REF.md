# Spatial Awareness Quick Reference

## What It Does

The Spatial Awareness System analyzes the 7x7 map around the player to:
-  Detect doors, NPCs, exits
-  Calculate proximity to objectives
-  Auto-complete missions when reached
-  Provide spatial context to LLM

## Quick Usage

### 1. Analyzing Surroundings

`python
from agent.spatial_analyzer import SpatialAnalyzer

analyzer = SpatialAnalyzer()
spatial_info = analyzer.analyze_surroundings(map_observation)

print(spatial_info['nearby_doors'])      # [('north', 2.0), ...]
print(spatial_info['closest_door'])      # ('north', 2.0)
print(spatial_info['passable_directions'])  # ['north', 'east', 'west']
`

### 2. Proximity Rewards

`python
# Automatically gives higher rewards when close to objective
multiplier = analyzer.calculate_proximity_reward(
    spatial_info,
    objective_type='location',
    target={'description': 'door'},
    base_reward=10.0
)
# Distance 1: 20.0 (2.0x)
# Distance 2: 15.0 (1.5x)
# Distance 3: 12.0 (1.2x)
`

### 3. Mission Completion

`python
is_complete, reason = analyzer.detect_mission_completion(
    spatial_info,
    objective={'type': 'location', 'target': {'map': 'PETALBURG_CITY'}},
    game_state={'map': 'PETALBURG_CITY'}
)
# Returns: (True, "Standing at door on PETALBURG_CITY")
`

### 4. LLM Context

`python
context = analyzer.get_spatial_context_for_llm(spatial_info, game_state)
# Output:
#  Location: LITTLEROOT_TOWN
#    Position: (15, 20)
#  Nearby doors: north (2.0 tiles)
#  Can move: north, east, west
`

## Map Constants

`python
# Behavior values (Channel 2)
BEHAVIOR_DOOR = 0x69   # Door
BEHAVIOR_NPC = 0xE0    # NPC
BEHAVIOR_EXIT = 0x62   # Exit
BEHAVIOR_GRASS = 0x02  # Grass

# Collision values (Channel 1)
COLLISION_PASSABLE = 0
COLLISION_WALL = 1
COLLISION_WATER = 3
`

## Integration in Training

Already integrated in:
-  gent/drl_env.py - Runs every step
-  gent/hybrid_llm_callback.py - Adds to LLM context
-  Objective reward calculation - Proximity bonuses

## Testing

`ash
python test_spatial_standalone.py
`

## Key Data Structure

`python
spatial_info = {
    'nearby_doors': [(direction, distance), ...],
    'nearby_npcs': [(direction, distance), ...],
    'nearby_exits': [(direction, distance), ...],
    'closest_door': (direction, distance) or None,
    'obstacles': {'north': bool, 'south': bool, 'east': bool, 'west': bool},
    'passable_directions': ['north', ...],
    'in_grass': bool,
    'near_water': bool,
    'num_obstacles': int,
    'num_passable': int
}
`

## Directions

`
     N
     |
W -- P -- E
     |
     S

P = Player (always at center 3,3)
`

Diagonal directions: 'north-east', 'south-west', etc.

## Common Patterns

### Check if path is clear to north
`python
if 'north' in spatial_info['passable_directions']:
    # Can move north
`

### Find closest feature
`python
if spatial_info['nearby_doors']:
    direction, distance = spatial_info['nearby_doors'][0]  # Already sorted
`

### Check if stuck
`python
if spatial_info['num_passable'] == 0:
    # Agent is surrounded by obstacles
`

### Detect specific location type
`python
if spatial_info['in_grass']:
    # In tall grass, expect wild Pokemon battles
if spatial_info['near_water']:
    # Near water, might need Surf HM
`

## Performance Tips

- Spatial analysis is fast (~0.1ms per step)
- No need to cache unless calling multiple times per step
- CNN feature extraction is optional (only for advanced use)

## Files Modified

`
NEW:  agent/spatial_analyzer.py           (390 lines)
MOD:  agent/drl_env.py                    (+20 lines)
MOD:  agent/hybrid_llm_callback.py        (+15 lines)
NEW:  test_spatial_standalone.py          (180 lines)
NEW:  SPATIAL_AWARENESS_SYSTEM.md         (Full documentation)
`

## What Changed in Training

**Before:**
- Agent only knew global position (x, y, map_name)
- No awareness of nearby features
- Rewards based only on reaching exact coordinates

**After:**
- Agent knows what's nearby (doors, NPCs, obstacles)
- Proximity rewards guide toward objectives
- Auto-completes missions when spatially achieved
- LLM gets spatial context for better planning

## Example Training Scenario

**Objective:** "Go to Professor Birch's lab"

**Without spatial awareness:**
- Agent wanders randomly
- No reward until reaching exact coordinates
- May get stuck at door without knowing

**With spatial awareness:**
- Agent gets +1.5x reward when door in sight
- Gets +2.0x reward when 1 tile from door
- Mission auto-completes when at door
- LLM sees: " Nearby doors: north (1.5 tiles)" and guides agent

## Next Actions

1. Train with: python train_ppo.py
2. Monitor spatial bonuses in logs
3. Check LLM context: python view_llm_logs.py
4. Tune proximity multipliers if needed

---

**Status:**  Ready to use
**Last Updated:** 2025-11-13 21:38
