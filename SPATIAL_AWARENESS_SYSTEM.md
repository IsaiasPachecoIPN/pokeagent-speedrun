# Spatial Awareness System Documentation

## Overview

The **Spatial Awareness System** enhances the Pokemon Emerald agent by providing deep understanding of its surroundings. It leverages the CNN's 7x7 map observation to detect nearby features (doors, NPCs, obstacles) and uses this information for:

1. **Proximity-based reward shaping** - Rewards increase as agent approaches objectives
2. **Mission completion detection** - Automatically detects when objectives are achieved
3. **LLM context enhancement** - Provides spatial information to help LLM make better decisions

## Architecture

### Core Components

`
agent/
  spatial_analyzer.py      # Main spatial analysis module
  drl_env.py              # Environment integration (step function)
  hybrid_llm_callback.py  # LLM integration (spatial context)
  cnn_policy.py           # CNN feature extraction
`

### Map Observation Format

The agent observes a **7x7 grid** centered on the player:

`
Channel 0: metatile_id  (visual appearance)
Channel 1: collision    (0=passable, 1=wall, 3=water, 0x60=door)
Channel 2: behavior     (0x69=door, 0xE0=NPC, 0x62=exit, 0x02=grass)
`

**Player position:** Always at center (3, 3)

## Key Features

### 1. Proximity Detection

`python
spatial_info = spatial_analyzer.analyze_surroundings(map_obs)

# Returns:
{
    'nearby_doors': [('north', 2.0), ('east', 3.5)],  # (direction, distance)
    'nearby_npcs': [('south-west', 1.4)],
    'nearby_exits': [],
    'closest_door': ('north', 2.0),
    'obstacles': {'north': False, 'south': True, 'east': False, 'west': False},
    'passable_directions': ['north', 'east', 'west'],
    'in_grass': False,
    'near_water': True,
    'num_obstacles': 1,
    'num_passable': 3
}
`

**Use case:** Know what's around the agent and where it can move.

### 2. Proximity-Based Reward Shaping

`python
multiplier = spatial_analyzer.calculate_proximity_reward(
    spatial_info,
    objective_type='location',
    target={'description': 'Go to the door', 'map': 'LITTLEROOT_TOWN'},
    base_reward=10.0
)

# If door is 1 tile away: multiplier = 2.0x  reward becomes 20.0
# If door is 2 tiles away: multiplier = 1.5x  reward becomes 15.0
# If door is 3 tiles away: multiplier = 1.2x  reward becomes 12.0
`

**Use case:** Guide agent toward objectives by increasing rewards as it gets closer.

### 3. Mission Completion Detection

`python
is_complete, reason = spatial_analyzer.detect_mission_completion(
    spatial_info,
    objective={'type': 'location', 'target': {'map': 'PETALBURG_CITY'}, 'description': 'Go to the gym entrance'},
    game_state={'map': 'PETALBURG_CITY', 'position': {'x': 50, 'y': 30}}
)

# Output:
# is_complete = True
# reason = "Standing at door on PETALBURG_CITY"
`

**Use case:** Automatically mark objectives as complete when spatial conditions are met.

### 4. LLM Context Generation

`python
context = spatial_analyzer.get_spatial_context_for_llm(spatial_info, game_state)

# Output:
'''
 Location: LITTLEROOT_TOWN
   Position: (15, 20)
 Nearby doors: north (2.0 tiles), east (3.5 tiles)
 Nearby NPCs: south-west (1.4 tiles)
 Can move: north, east, west
 Environment: tall grass
'''
`

**Use case:** Provide LLM with spatial awareness for better strategic planning.

## Integration Points

### A. Environment Step Function (drl_env.py)

Spatial analysis runs every step:

`python
def step(self, action: int):
    # ... execute action ...
    observation = self.state_reader.get_observation_for_drl(map_radius=3)
    
    #  Analyze spatial context
    if 'map' in observation:
        self.last_spatial_info = self.spatial_analyzer.analyze_surroundings(observation['map'])
    
    # Calculate reward with spatial awareness
    reward = self._calculate_reward_from_lightweight(prev_state, current_state)
`

### B. Objective Rewards (drl_env.py)

Spatial bonuses applied to objectives:

`python
def _calculate_objective_rewards(self, prev_state, current_state):
    for objective in active_objectives:
        # Standard objective rewards...
        
        #  Apply spatial proximity multiplier
        if self.last_spatial_info:
            spatial_multiplier = self.spatial_analyzer.calculate_proximity_reward(
                self.last_spatial_info, obj_type, target, base_reward=1.0
            )
            objective_reward *= spatial_multiplier
            
            # Check for spatial completion
            is_complete, reason = self.spatial_analyzer.detect_mission_completion(
                self.last_spatial_info, objective.__dict__, current_state
            )
            if is_complete:
                objective_reward += 50.0 * weight  # Big bonus!
`

### C. LLM Callback (hybrid_llm_callback.py)

Spatial context added to LLM prompts:

`python
def _get_game_state_summary(self):
    # ... training stats ...
    
    #  Add spatial context
    if env.last_spatial_info:
        spatial_context = env.spatial_analyzer.get_spatial_context_for_llm(
            env.last_spatial_info, game_state
        )
        summary += f'\nSPATIAL CONTEXT:\n{spatial_context}\n'
    
    # ... objectives ...
`

## Pokemon Emerald Map Constants

These are extracted from the game's memory:

`python
# Collision Types (Channel 1)
COLLISION_PASSABLE = 0      # Normal ground
COLLISION_WALL = 1          # Walls, trees, buildings
COLLISION_WATER = 3         # Water tiles
COLLISION_DOOR = 0x60       # Door collision

# Behavior Types (Channel 2)
BEHAVIOR_DOOR = 0x69        # Door marker
BEHAVIOR_NPC = 0xE0         # NPC sprite
BEHAVIOR_EXIT = 0x62        # Map exit/warp
BEHAVIOR_GRASS = 0x02       # Tall grass (wild Pokemon)
BEHAVIOR_SIGN = 0x6C        # Readable sign
`

## Example: Spatial-Aware Objective

`json
{
  "id": "obj_001",
  "type": "location",
  "description": "Go to Professor Birch's lab",
  "target": {
    "map": "LITTLEROOT_TOWN",
    "x": 18,
    "y": 12,
    "description": "The lab entrance door"
  },
  "reward_weight": 1.5,
  "status": "active"
}
`

**How spatial awareness helps:**

1. **Approaching:** As agent moves closer to (18, 12), proximity rewards increase
2. **Door detection:** When agent sees door in 7x7 vision, bonus multiplier applied
3. **Completion:** When agent reaches door (distance < 1.0), objective auto-completes
4. **LLM guidance:** LLM sees " Nearby doors: north (1.5 tiles)" and can guide agent

## Performance Characteristics

- **Analysis speed:** ~0.1ms per step (negligible overhead)
- **Memory usage:** ~100 bytes per spatial_info dict
- **CNN feature extraction:** Optional, only when needed (not on every step)

## Testing

Run the standalone test:

`ash
python test_spatial_standalone.py
`

Expected output:
`
 ALL TESTS PASSED!
The spatial awareness system is working!
Core features verified:
   Door detection and proximity
   NPC detection
   Obstacle mapping
   Direction analysis
`

## Advanced Usage

### Extracting CNN Features

`python
# Get intermediate CNN activations for spatial analysis
cnn_features = spatial_analyzer.extract_cnn_features(
    map_obs=torch.tensor(observation['map']),
    layer='conv2'  # Options: 'conv1', 'conv2', 'flattened'
)

# Shape: (batch, 64, 7, 7) for conv2 layer
# Use for attention mechanisms, spatial embeddings, etc.
`

### Custom Proximity Logic

`python
# Add custom proximity detection
def detect_custom_feature(spatial_info, game_state):
    # Check if near a specific landmark
    if spatial_info['nearby_doors'] and spatial_info['near_water']:
        return True, "Found dock entrance"
    return False, ""
`

## Future Enhancements

1. **Attention mechanism:** Highlight important spatial regions in CNN
2. **Path planning:** Use A* with spatial obstacles for navigation
3. **Spatial memory:** Remember visited locations across episodes
4. **Multi-scale analysis:** Process 7x7 (local) + 15x15 (regional) maps
5. **Temporal tracking:** Track movement patterns and predict player intent

## Troubleshooting

### Issue: Spatial info is None

**Cause:** Map observation not available in current step

**Fix:** Check that observation['map'] exists before analyzing

### Issue: No doors/NPCs detected

**Cause:** Feature constants may be incorrect for game version

**Fix:** Verify BEHAVIOR_DOOR and BEHAVIOR_NPC values using memory viewer

### Issue: Proximity rewards not working

**Cause:** Objective description doesn't match proximity logic

**Fix:** Ensure objective description contains keywords like "door", "npc", etc.

## Integration Checklist

- [x] SpatialAnalyzer class created
- [x] Integrated into drl_env.py step function
- [x] Proximity rewards added to objective calculation
- [x] Spatial context added to LLM prompts
- [x] Mission completion detection implemented
- [x] Test script created and passing
- [x] Documentation complete

## Next Steps

1. Train agent with spatial awareness enabled
2. Monitor LLM logs to see spatial context in action
3. Tune proximity reward multipliers based on training performance
4. Add more spatial features (grass detection, water proximity, etc.)
5. Implement spatial memory for cross-episode learning

---

**Status:**  Ready for training

**Last Updated:** 2025-11-13 21:37
