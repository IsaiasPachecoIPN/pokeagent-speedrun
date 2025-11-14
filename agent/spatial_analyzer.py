"""
Spatial Awareness System for Pokemon Agent

This module provides spatial analysis capabilities that leverage the CNN's
map processing to understand the agent's location and surroundings.

Features:
- Proximity detection (doors, NPCs, obstacles, exits)
- Spatial feature extraction from CNN activations
- Direction-aware analysis (what's north/south/east/west)
- Mission completion detection based on spatial patterns
- Reward shaping based on proximity to objectives
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


# === Pokemon Emerald Map Constants ===
# These are specific to Pokemon Emerald's collision/behavior system

# Collision types (channel 1 in map observation)
COLLISION_PASSABLE = 0
COLLISION_WALL = 1
COLLISION_WATER = 3
COLLISION_DOOR = 0x60  # Doors are special collision types

# Behavior types (channel 2 in map observation)
BEHAVIOR_DOOR = 0x69  # Door behavior
BEHAVIOR_NPC = 0xE0  # NPC marker
BEHAVIOR_EXIT = 0x62  # Map exit
BEHAVIOR_GRASS = 0x02  # Tall grass
BEHAVIOR_SIGN = 0x6C  # Readable sign


class SpatialAnalyzer:
    '''
    Analyzes spatial context from map observations and CNN features.
    
    The agent observes a 7x7 grid centered on the player:
    - Channel 0: metatile_id (visual appearance)
    - Channel 1: collision (0=passable, 1=wall, 3=water, etc.)
    - Channel 2: behavior (door, NPC, exit markers)
    
    Player is always at the center: position (3, 3)
    '''
    
    def __init__(self, cnn_extractor=None):
        '''
        Args:
            cnn_extractor: PokemonCNNExtractor instance for feature extraction
        '''
        self.cnn_extractor = cnn_extractor
        self.center_pos = (3, 3)  # Player is always at center of 7x7 grid
        
    def analyze_surroundings(self, map_obs: np.ndarray) -> Dict[str, Any]:
        '''
        Comprehensive spatial analysis of the 7x7 map observation.
        
        Args:
            map_obs: Shape (7, 7, 3) - [metatile_id, collision, behavior]
            
        Returns:
            Dictionary with spatial information:
            {
                'nearby_doors': [(direction, distance), ...],
                'nearby_npcs': [(direction, distance), ...],
                'nearby_exits': [(direction, distance), ...],
                'obstacles': {'north': bool, 'south': bool, 'east': bool, 'west': bool},
                'passable_directions': [str, ...],
                'closest_door': (direction, distance) or None,
                'in_grass': bool,
                'near_water': bool
            }
        '''
        if map_obs.shape != (7, 7, 3):
            logger.error(f'Invalid map shape: {map_obs.shape}, expected (7, 7, 3)')
            return self._empty_analysis()
        
        collision_map = map_obs[:, :, 1]
        behavior_map = map_obs[:, :, 2]
        
        # === Detect nearby features ===
        nearby_doors = self._find_nearby_features(behavior_map, BEHAVIOR_DOOR)
        nearby_npcs = self._find_nearby_features(behavior_map, BEHAVIOR_NPC)
        nearby_exits = self._find_nearby_features(behavior_map, BEHAVIOR_EXIT)
        
        # === Check immediate surroundings (4 cardinal directions) ===
        cx, cy = self.center_pos
        obstacles = {
            'north': self._is_obstacle(collision_map, cx, cy - 1),
            'south': self._is_obstacle(collision_map, cx, cy + 1),
            'east': self._is_obstacle(collision_map, cx + 1, cy),
            'west': self._is_obstacle(collision_map, cx - 1, cy)
        }
        
        passable_directions = [dir for dir, blocked in obstacles.items() if not blocked]
        
        # === Find closest door ===
        closest_door = min(nearby_doors, key=lambda x: x[1]) if nearby_doors else None
        
        # === Environmental context ===
        in_grass = behavior_map[cx, cy] == BEHAVIOR_GRASS
        near_water = self._check_nearby(collision_map, COLLISION_WATER, max_distance=2)
        
        return {
            'nearby_doors': nearby_doors,
            'nearby_npcs': nearby_npcs,
            'nearby_exits': nearby_exits,
            'obstacles': obstacles,
            'passable_directions': passable_directions,
            'closest_door': closest_door,
            'in_grass': in_grass,
            'near_water': near_water,
            'num_obstacles': sum(obstacles.values()),
            'num_passable': len(passable_directions)
        }
    
    def _find_nearby_features(
        self, 
        behavior_map: np.ndarray, 
        feature_type: int,
        max_distance: int = 3
    ) -> List[Tuple[str, float]]:
        '''
        Find all instances of a feature type within max_distance.
        
        Returns:
            List of (direction, distance) tuples
        '''
        cx, cy = self.center_pos
        features = []
        
        for y in range(7):
            for x in range(7):
                if behavior_map[y, x] == feature_type:
                    dx = x - cx
                    dy = y - cy
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    if distance <= max_distance and distance > 0:
                        direction = self._get_direction(dx, dy)
                        features.append((direction, distance))
        
        return sorted(features, key=lambda x: x[1])  # Sort by distance
    
    def _get_direction(self, dx: int, dy: int) -> str:
        '''Convert dx, dy offsets to compass direction.'''
        # Determine primary direction
        if abs(dx) > abs(dy):
            primary = 'east' if dx > 0 else 'west'
            secondary = 'south' if dy > 0 else 'north' if dy != 0 else ''
        else:
            primary = 'south' if dy > 0 else 'north' if dy != 0 else ''
            secondary = 'east' if dx > 0 else 'west' if dx != 0 else ''
        
        # Combine if diagonal
        if secondary:
            return f'{secondary}-{primary}'
        return primary
    
    def _is_obstacle(self, collision_map: np.ndarray, x: int, y: int) -> bool:
        '''Check if position is blocked.'''
        if not (0 <= x < 7 and 0 <= y < 7):
            return True  # Out of bounds = obstacle
        
        collision = collision_map[y, x]
        return collision == COLLISION_WALL or collision == COLLISION_WATER
    
    def _check_nearby(
        self, 
        map_channel: np.ndarray, 
        value: int, 
        max_distance: int = 2
    ) -> bool:
        '''Check if value exists within max_distance of center.'''
        cx, cy = self.center_pos
        
        for y in range(max(0, cy - max_distance), min(7, cy + max_distance + 1)):
            for x in range(max(0, cx - max_distance), min(7, cx + max_distance + 1)):
                if map_channel[y, x] == value:
                    distance = abs(x - cx) + abs(y - cy)  # Manhattan distance
                    if distance <= max_distance:
                        return True
        return False
    
    def _empty_analysis(self) -> Dict[str, Any]:
        '''Return empty analysis dict when input is invalid.'''
        return {
            'nearby_doors': [],
            'nearby_npcs': [],
            'nearby_exits': [],
            'obstacles': {'north': True, 'south': True, 'east': True, 'west': True},
            'passable_directions': [],
            'closest_door': None,
            'in_grass': False,
            'near_water': False,
            'num_obstacles': 4,
            'num_passable': 0
        }
    
    def calculate_proximity_reward(
        self,
        spatial_info: Dict[str, Any],
        objective_type: str,
        target: Dict[str, Any],
        base_reward: float = 1.0
    ) -> float:
        '''
        Calculate reward bonus based on spatial proximity to objective.
        
        Args:
            spatial_info: Output from analyze_surroundings()
            objective_type: Type of objective ('location', 'npc', 'door', etc.)
            target: Target information from objective
            base_reward: Base reward to multiply
            
        Returns:
            Reward multiplier (e.g., 1.5 if close to door objective)
        '''
        multiplier = 1.0
        
        if objective_type == 'location':
            # If objective is to reach a location with a door
            if 'door' in target.get('description', '').lower():
                if spatial_info['closest_door']:
                    _, distance = spatial_info['closest_door']
                    # Closer to door = higher reward
                    # distance=1  2.0x, distance=2  1.5x, distance=3  1.2x
                    multiplier = max(1.0, 2.0 - (distance * 0.3))
                    logger.info(f' Proximity bonus: {multiplier:.2f}x (door distance: {distance:.1f})')
        
        elif objective_type == 'npc':
            # Reward for approaching NPCs
            if spatial_info['nearby_npcs']:
                closest_npc_distance = spatial_info['nearby_npcs'][0][1]
                multiplier = max(1.0, 2.0 - (closest_npc_distance * 0.3))
                logger.info(f' Proximity bonus: {multiplier:.2f}x (NPC distance: {closest_npc_distance:.1f})')
        
        elif objective_type == 'explore':
            # Reward for moving through passable areas
            if spatial_info['num_passable'] >= 3:
                multiplier = 1.2  # Open area = good for exploration
        
        return base_reward * multiplier
    
    def detect_mission_completion(
        self,
        spatial_info: Dict[str, Any],
        objective: Dict[str, Any],
        game_state: Dict[str, Any]
    ) -> Tuple[bool, str]:
        '''
        Detect if mission is spatially complete.
        
        Args:
            spatial_info: Spatial analysis
            objective: Objective dict with type, target, description
            game_state: Current game state with position, map, etc.
            
        Returns:
            (is_complete, reason)
        '''
        obj_type = objective.get('type', '')
        target = objective.get('target', {})
        description = objective.get('description', '').lower()
        
        # === Location-based completion ===
        if obj_type == 'location':
            target_map = target.get('map', '')
            current_map = game_state.get('map', '')
            
            # Check if on correct map
            if target_map and current_map == target_map:
                # If target has specific coordinates
                if 'x' in target and 'y' in target:
                    curr_pos = game_state.get('position', {})
                    if curr_pos.get('x') == target['x'] and curr_pos.get('y') == target['y']:
                        return True, f'Reached exact location: {target_map} ({target["x"]}, {target["y"]})'
                
                # If description mentions door and we're at a door
                if 'door' in description or 'entrance' in description:
                    if spatial_info['closest_door'] and spatial_info['closest_door'][1] < 1.0:
                        return True, f'Standing at door on {target_map}'
                
                # Just being on the map might be sufficient
                return True, f'Reached map: {target_map}'
        
        # === NPC interaction completion ===
        elif obj_type == 'npc':
            # Check if standing next to NPC (distance < 1.5)
            if spatial_info['nearby_npcs']:
                closest_npc = spatial_info['nearby_npcs'][0]
                if closest_npc[1] < 1.5:
                    return True, f'Adjacent to NPC ({closest_npc[0]} direction)'
        
        return False, ''
    
    def get_spatial_context_for_llm(
        self,
        spatial_info: Dict[str, Any],
        game_state: Dict[str, Any]
    ) -> str:
        '''
        Generate human-readable spatial context for LLM decision-making.
        
        Returns:
            String description of surroundings
        '''
        lines = []
        lines.append(f' Location: {game_state.get("map", "Unknown")}')
        lines.append(f'   Position: ({game_state.get("position", {}).get("x", "?")}, {game_state.get("position", {}).get("y", "?")})')
        
        # Nearby features
        if spatial_info['nearby_doors']:
            door_info = ', '.join([f'{dir} ({dist:.1f} tiles)' for dir, dist in spatial_info['nearby_doors'][:2]])
            lines.append(f' Nearby doors: {door_info}')
        
        if spatial_info['nearby_npcs']:
            npc_info = ', '.join([f'{dir} ({dist:.1f} tiles)' for dir, dist in spatial_info['nearby_npcs'][:2]])
            lines.append(f' Nearby NPCs: {npc_info}')
        
        if spatial_info['nearby_exits']:
            exit_info = ', '.join([f'{dir} ({dist:.1f} tiles)' for dir, dist in spatial_info['nearby_exits'][:2]])
            lines.append(f' Exits: {exit_info}')
        
        # Movement options
        if spatial_info['passable_directions']:
            lines.append(f' Can move: {", ".join(spatial_info["passable_directions"])}')
        else:
            lines.append(' Surrounded by obstacles')
        
        # Environment
        env_tags = []
        if spatial_info['in_grass']:
            env_tags.append('tall grass')
        if spatial_info['near_water']:
            env_tags.append('near water')
        if env_tags:
            lines.append(f' Environment: {", ".join(env_tags)}')
        
        return '\n'.join(lines)
    
    def extract_cnn_features(
        self, 
        map_obs: torch.Tensor,
        layer: str = 'conv2'
    ) -> torch.Tensor:
        '''
        Extract intermediate CNN features for spatial analysis.
        
        Args:
            map_obs: Tensor of shape (batch, 7, 7, 3) or (7, 7, 3)
            layer: Which layer to extract ('conv1', 'conv2', 'flattened')
            
        Returns:
            Extracted features from specified layer
        '''
        if self.cnn_extractor is None:
            logger.warning('No CNN extractor provided, cannot extract features')
            return None
        
        # Ensure correct shape
        if map_obs.dim() == 3:
            map_obs = map_obs.unsqueeze(0)  # Add batch dimension
        
        # Transpose to PyTorch format (batch, C, H, W)
        if map_obs.shape[-1] == 3:
            map_obs = map_obs.permute(0, 3, 1, 2)
        
        # Extract features based on requested layer
        with torch.no_grad():
            if layer == 'conv1':
                # First 3 layers: Conv2d, ReLU, BatchNorm
                features = self.cnn_extractor.map_cnn[:3](map_obs)
            elif layer == 'conv2':
                # First 6 layers: through second conv block
                features = self.cnn_extractor.map_cnn[:6](map_obs)
            elif layer == 'flattened':
                # Full CNN output
                features = self.cnn_extractor.map_cnn(map_obs)
            else:
                raise ValueError(f'Unknown layer: {layer}')
        
        return features


# === Helper Functions ===

def compute_spatial_distance(
    pos1: Dict[str, int],
    pos2: Dict[str, int],
    map1: str,
    map2: str
) -> float:
    '''
    Compute spatial distance between two positions.
    
    Returns:
        Distance in tiles, or inf if on different maps
    '''
    if map1 != map2:
        return float('inf')
    
    dx = pos1.get('x', 0) - pos2.get('x', 0)
    dy = pos1.get('y', 0) - pos2.get('y', 0)
    
    return np.sqrt(dx**2 + dy**2)
