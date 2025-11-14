'''
Test Spatial Awareness System

This script demonstrates how the SpatialAnalyzer extracts spatial information
from map observations and provides context for reward shaping and mission completion.
'''

import numpy as np
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from agent.spatial_analyzer import SpatialAnalyzer, BEHAVIOR_DOOR, BEHAVIOR_NPC, COLLISION_WALL

def create_test_map_with_door():
    '''Create a test 7x7 map with a door to the north.'''
    map_obs = np.zeros((7, 7, 3), dtype=np.float32)
    
    # Channel 0: metatile_id (visual appearance)
    map_obs[:, :, 0] = np.random.randint(0, 100, (7, 7))
    
    # Channel 1: collision (walls around edges)
    map_obs[0, :, 1] = COLLISION_WALL  # North wall
    map_obs[6, :, 1] = COLLISION_WALL  # South wall
    map_obs[:, 0, 1] = COLLISION_WALL  # West wall
    map_obs[:, 6, 1] = COLLISION_WALL  # East wall
    
    # Channel 2: behavior (door at position north)
    map_obs[1, 3, 2] = BEHAVIOR_DOOR  # Door 2 tiles north of player
    
    return map_obs

def create_test_map_with_npc():
    '''Create a test 7x7 map with an NPC nearby.'''
    map_obs = np.zeros((7, 7, 3), dtype=np.float32)
    
    # Channel 0: metatile_id
    map_obs[:, :, 0] = np.random.randint(0, 100, (7, 7))
    
    # Channel 1: collision (some walls)
    map_obs[0, :, 1] = COLLISION_WALL
    map_obs[6, :, 1] = COLLISION_WALL
    
    # Channel 2: NPC to the east
    map_obs[3, 4, 2] = BEHAVIOR_NPC  # NPC 1 tile east
    
    return map_obs

def test_basic_analysis():
    '''Test basic spatial analysis with door.'''
    print('\n' + '='*60)
    print('TEST 1: Basic Spatial Analysis (Door Detection)')
    print('='*60)
    
    analyzer = SpatialAnalyzer()
    map_obs = create_test_map_with_door()
    
    spatial_info = analyzer.analyze_surroundings(map_obs)
    
    print(f'\nNearby doors: {spatial_info["nearby_doors"]}')
    print(f'Closest door: {spatial_info["closest_door"]}')
    print(f'Obstacles: {spatial_info["obstacles"]}')
    print(f'Passable directions: {spatial_info["passable_directions"]}')
    print(f'Number of obstacles: {spatial_info["num_obstacles"]}')
    
    assert len(spatial_info['nearby_doors']) > 0, 'Should detect door'
    assert spatial_info['closest_door'] is not None, 'Should find closest door'
    print('\n Door detection working!')

def test_npc_detection():
    '''Test NPC detection and proximity.'''
    print('\n' + '='*60)
    print('TEST 2: NPC Proximity Detection')
    print('='*60)
    
    analyzer = SpatialAnalyzer()
    map_obs = create_test_map_with_npc()
    
    spatial_info = analyzer.analyze_surroundings(map_obs)
    
    print(f'\nNearby NPCs: {spatial_info["nearby_npcs"]}')
    print(f'Passable directions: {spatial_info["passable_directions"]}')
    
    assert len(spatial_info['nearby_npcs']) > 0, 'Should detect NPC'
    assert spatial_info['nearby_npcs'][0][1] < 2.0, 'NPC should be close'
    print('\n NPC detection working!')

def test_proximity_reward():
    '''Test proximity-based reward calculation.'''
    print('\n' + '='*60)
    print('TEST 3: Proximity Reward Calculation')
    print('='*60)
    
    analyzer = SpatialAnalyzer()
    map_obs = create_test_map_with_door()
    
    spatial_info = analyzer.analyze_surroundings(map_obs)
    
    # Test location objective with door
    objective_type = 'location'
    target = {'description': 'Go through the door to next area', 'map': 'TestMap'}
    
    base_reward = 10.0
    modified_reward = analyzer.calculate_proximity_reward(
        spatial_info,
        objective_type,
        target,
        base_reward
    )
    
    print(f'\nBase reward: {base_reward:.2f}')
    print(f'Modified reward (with proximity): {modified_reward:.2f}')
    print(f'Multiplier: {modified_reward / base_reward:.2f}x')
    
    assert modified_reward > base_reward, 'Proximity should increase reward'
    print('\n Proximity reward shaping working!')

def test_mission_completion():
    '''Test spatial mission completion detection.'''
    print('\n' + '='*60)
    print('TEST 4: Mission Completion Detection')
    print('='*60)
    
    analyzer = SpatialAnalyzer()
    map_obs = create_test_map_with_door()
    
    spatial_info = analyzer.analyze_surroundings(map_obs)
    
    # Simulate being on the target map with a door objective
    objective = {
        'type': 'location',
        'target': {'map': 'TestMap', 'description': 'entrance'},
        'description': 'Go to the door entrance'
    }
    
    game_state = {
        'map': 'TestMap',
        'position': {'x': 50, 'y': 50}
    }
    
    # Test when far from door
    is_complete, reason = analyzer.detect_mission_completion(
        spatial_info,
        objective,
        game_state
    )
    
    print(f'\nMission complete: {is_complete}')
    print(f'Reason: {reason}')
    
    # Modify spatial info to simulate being at the door
    spatial_info['closest_door'] = ('north', 0.5)  # Very close to door
    
    is_complete2, reason2 = analyzer.detect_mission_completion(
        spatial_info,
        objective,
        game_state
    )
    
    print(f'\nAfter moving to door:')
    print(f'Mission complete: {is_complete2}')
    print(f'Reason: {reason2}')
    
    assert is_complete2 == True, 'Should detect completion when at door'
    print('\n Mission completion detection working!')

def test_llm_context():
    '''Test LLM-friendly spatial context generation.'''
    print('\n' + '='*60)
    print('TEST 5: LLM Spatial Context Generation')
    print('='*60)
    
    analyzer = SpatialAnalyzer()
    map_obs = create_test_map_with_door()
    
    spatial_info = analyzer.analyze_surroundings(map_obs)
    
    game_state = {
        'map': 'LITTLEROOT_TOWN',
        'position': {'x': 15, 'y': 20}
    }
    
    context = analyzer.get_spatial_context_for_llm(spatial_info, game_state)
    
    print('\nLLM Context:')
    print('-' * 40)
    print(context)
    print('-' * 40)
    
    assert 'LITTLEROOT_TOWN' in context, 'Should include map name'
    assert 'door' in context.lower(), 'Should mention nearby door'
    print('\n LLM context generation working!')

def main():
    '''Run all tests.'''
    print('\n Starting Spatial Awareness System Tests\n')
    
    try:
        test_basic_analysis()
        test_npc_detection()
        test_proximity_reward()
        test_mission_completion()
        test_llm_context()
        
        print('\n' + '='*60)
        print(' ALL TESTS PASSED!')
        print('='*60)
        print('\nThe spatial awareness system is working correctly!')
        print('It can:')
        print('   Detect doors, NPCs, and obstacles')
        print('   Calculate proximity-based rewards')
        print('   Detect mission completion spatially')
        print('   Generate LLM-friendly context')
        print('\nReady for integration with training!')
        
    except AssertionError as e:
        print(f'\n TEST FAILED: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'\n ERROR: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
