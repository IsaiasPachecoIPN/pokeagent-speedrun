"""
Test Map Observation Capture

Quick test to verify that the map observation capture works correctly.
"""

import numpy as np
import json
from datetime import datetime

# Simulate the _capture_map_observation logic
def test_map_capture():
    print(' Testing Map Observation Capture\n')
    
    # Create a sample 7x7x3 map observation
    map_array = np.zeros((7, 7, 3), dtype=np.float32)
    
    # Add some features
    # Channel 0: metatile_id
    map_array[:, :, 0] = np.random.randint(0, 100, (7, 7))
    
    # Channel 1: collision (walls on edges)
    map_array[0, :, 1] = 1  # North wall
    map_array[6, :, 1] = 1  # South wall
    
    # Channel 2: behavior (door to north, NPC to east)
    map_array[1, 3, 2] = 0x69  # BEHAVIOR_DOOR
    map_array[3, 4, 2] = 0xE0  # BEHAVIOR_NPC
    
    # Convert to JSON-serializable format
    map_data = {
        'shape': list(map_array.shape),
        'channels': {
            'metatile_id': map_array[:, :, 0].tolist(),
            'collision': map_array[:, :, 1].tolist(),
            'behavior': map_array[:, :, 2].tolist()
        },
        'center_position': [3, 3],
        'description': '7x7 grid centered on player - [y][x] indexed'
    }
    
    # Sample spatial analysis
    spatial_data = {
        'nearby_doors': [('north', 2.0)],
        'nearby_npcs': [('east', 1.0)],
        'nearby_exits': [],
        'closest_door': ('north', 2.0),
        'obstacles': {'north': False, 'south': False, 'east': False, 'west': False},
        'passable_directions': ['north', 'south', 'east', 'west'],
        'in_grass': False,
        'near_water': False,
        'num_obstacles': 0,
        'num_passable': 4
    }
    
    # Create the full observation object
    observation = {
        'map_array': map_data,
        'spatial_analysis': spatial_data,
        'ascii_visualization': create_ascii_map(map_array, spatial_data),
        'timestamp': datetime.now().isoformat()
    }
    
    # Test JSON serialization
    try:
        json_str = json.dumps(observation, indent=2)
        print(' JSON serialization successful!')
        print(f'   Size: {len(json_str)} bytes')
        
        # Verify deserialization
        parsed = json.loads(json_str)
        print(' JSON deserialization successful!')
        
        # Display the ASCII map
        print('\n ASCII Visualization:')
        print(observation['ascii_visualization'])
        
        # Check spatial data
        print('\n Spatial Analysis:')
        print(f'   Nearby doors: {len(spatial_data["nearby_doors"])}')
        print(f'   Nearby NPCs: {len(spatial_data["nearby_npcs"])}')
        print(f'   Passable directions: {spatial_data["num_passable"]}/4')
        
        print('\n All tests passed!')
        return True
        
    except Exception as e:
        print(f' Test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def create_ascii_map(map_array, spatial_data):
    BEHAVIOR_DOOR = 0x69
    BEHAVIOR_NPC = 0xE0
    BEHAVIOR_EXIT = 0x62
    BEHAVIOR_GRASS = 0x02
    COLLISION_WALL = 1
    COLLISION_WATER = 3
    
    lines = []
    lines.append('\n7x7 Map Observation (Player at center "P"):')
    lines.append('=' * 43)
    
    for y in range(7):
        row = ''
        for x in range(7):
            if y == 3 and x == 3:
                row += ' P '
                continue
            
            behavior = map_array[y, x, 2]
            collision = map_array[y, x, 1]
            
            if behavior == BEHAVIOR_DOOR:
                row += ' D '
            elif behavior == BEHAVIOR_NPC:
                row += ' N '
            elif behavior == BEHAVIOR_EXIT:
                row += ' E '
            elif behavior == BEHAVIOR_GRASS:
                row += ' G '
            elif collision == COLLISION_WALL:
                row += ' # '
            elif collision == COLLISION_WATER:
                row += ' ~ '
            else:
                row += ' . '
        
        lines.append(f'|{row}|')
    
    lines.append('=' * 43)
    lines.append('Legend: P=Player D=Door N=NPC E=Exit G=Grass #=Wall ~=Water .=Passable')
    
    if spatial_data:
        lines.append('\nSpatial Analysis:')
        if spatial_data.get('nearby_doors'):
            lines.append(f'   Doors: {spatial_data["nearby_doors"]}')
        if spatial_data.get('nearby_npcs'):
            lines.append(f'   NPCs: {spatial_data["nearby_npcs"]}')
        if spatial_data.get('passable_directions'):
            lines.append(f'   Can move: {", ".join(spatial_data["passable_directions"])}')
    
    return '\n'.join(lines)

if __name__ == '__main__':
    test_map_capture()
