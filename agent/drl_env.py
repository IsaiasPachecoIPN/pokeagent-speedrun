"""
Deep Reinforcement Learning Environment for Pokemon Emerald
Compatible with Stable Baselines3 (PPO, DQN, etc.)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

from pokemon_env.emulator import EmeraldEmulator
from agent.lightweight_state_reader import LightweightStateReader
from agent.spatial_analyzer import SpatialAnalyzer

logger = logging.getLogger(__name__)


class PokemonEmeraldEnv(gym.Env):
    """
    Gymnasium environment for training DRL agents on Pokemon Emerald.
    
    Features:
    - Uses structured game state (not raw pixels)
    - Automatic reset to saved state
    - Reward shaping for speedrun objectives
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(
        self,
        rom_path: str = "Emerald-GBAdvance/rom.gba",
        initial_state_path: str = "Emerald-GBAdvance/quick_start_save.state",
        render_mode: Optional[str] = None,
        max_steps: int = 10000,
        frame_skip: int = 36,  # Execute action for N frames (6 frames = 10 decisions/sec at 60fps)
        turbo_mode: bool = True,  # If True, runs emulator at max speed (no throttling)
        state_read_interval: int = 1  # Read full game state every N steps (1=every step, 5=every 5 steps)
    ):
        """
        Initialize the Pokemon Emerald DRL environment.
        
        Args:
            rom_path: Path to the ROM file
            initial_state_path: Path to the initial save state
            render_mode: 'human' or 'rgb_array' for visualization
            max_steps: Maximum steps per episode
            frame_skip: Number of frames to execute each action
                       - Lower = more decisions/sec but buttons may not register
                       - Higher = fewer decisions/sec but more reliable
                       - 6 is a good balance (10 decisions/sec at 60fps)
            turbo_mode: Run emulator at maximum speed (no sleep/throttling)
            state_read_interval: Read full game state every N steps for speed
                       - 1 = read every step (slow but accurate)
                       - 5 = read every 5 steps (5x faster, slightly stale data)
        """
        super().__init__()
        
        self.rom_path = rom_path
        self.initial_state_path = initial_state_path
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.turbo_mode = turbo_mode
        self.state_read_interval = state_read_interval
        
        # Initialize emulator
        logger.info(f"Initializing emulator with ROM: {rom_path}")
        self.emulator = EmeraldEmulator(rom_path=rom_path, headless=(render_mode != 'human'))
        self.emulator.initialize()
        logger.info("Emulator initialized successfully")
        
                # Initialize lightweight state reader for fast observations
        logger.info("Initializing lightweight state reader for DRL...")
        self.state_reader = LightweightStateReader(self.emulator.memory_reader)
        logger.info("Lightweight reader ready")
        
        # Load initial state ONCE - critical for training speed!
        logger.info(f"Loading initial state from: {initial_state_path}")
        self.emulator.load_state(initial_state_path)
        # Now save it to memory so we can restore quickly
        logger.info("Saving state to memory for fast resets...")
        self.initial_state_bytes = self.emulator.save_state()
        logger.info(f"State cached in memory: {len(self.initial_state_bytes)} bytes")
        
        # Get initial game state and cache it too
        logger.info("Getting initial game state...")
        self.initial_game_state = self.emulator.get_comprehensive_state()
        logger.info(f"Initial game state cached")
        
        # Action space: 7 GBA buttons (START removed to prevent save menu crash)
        # 0=A, 1=B, 2=SELECT, 3=RIGHT, 4=LEFT, 5=UP, 6=DOWN
        self.action_space = spaces.Discrete(6)
        
        self._action_map = {
            0: "a",
            1: "b", 
            #2: "select",
            2: "right",
            3: "left",
            4: "up",
            5: "down"
        }
        
        # Observation space: Dictionary with map image and game state vector
        # Map: 7x7x3 channels (metatile_id, collision, behavior) - for CNN
        # Vector: player position (2) + party info (12) + game state (4) + milestone_count (1) = 19 features
        self.observation_space = spaces.Dict({
            'map': spaces.Box(
                low=0,
                high=1.0,
                shape=(7, 7, 3),  # Height x Width x Channels
                dtype=np.float32
            ),
            'vector': spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(19,),  # Aumentado de 18 a 19 para milestone_count
                dtype=np.float32
            )
        })
        
        # Internal state tracking
        self.prev_game_state = None
        self.current_step = 0
        self.episode_reward = 0.0
        self.cached_game_state = None  # Cache for state_read_interval optimization
        
        # Tracking for reward calculation
        self.prev_location = None
        self.visited_locations = set()
        self.prev_position = None
        self.stationary_steps = 0
        
        # LLM reward shaping (GLOBAL VARIABLES approach)
        self.llm_reward_multiplier = 1.0  # LLM modifies this (0.0 to 2.0)
        self.llm_advice = ""  # Current strategic advice from LLM
        self.llm_last_update_step = 0  # Last step when LLM was consulted
        self.last_milestone_count = 0  # Track milestone progress for LLM
        
        # 🆕 Caché de diálogos (para capturar texto cuando aparece)
        self.last_dialog = ""  # Último diálogo detectado
        self.last_dialog_step = 0  # Step cuando se detectó
        self.dialog_cache_duration = 500  # Mantener diálogo por 500 steps
        
        # Directional reward shaping (PROXIMITY-BASED)
        self.directional_multiplier = 1.0  # Based on distance to objectives
        self.directional_advice = ""  # Direction guidance
        
        # 🆕 Hybrid DRL+LLM: Objectives manager integration
        self.objectives_manager = None  # Set by callback if using hybrid mode
        self.enable_dialogue_capture = False  # Enable via callback
        self.captured_dialogues = set()  # Track dialogues to avoid duplicates
        
        # 🆕 Spatial Awareness System
        self.spatial_analyzer = SpatialAnalyzer()  # Initialized without CNN (will be set later)
        self.last_spatial_info = None  # Cache last spatial analysis

        # Departure penalty configuration (strong penalty for leaving area with unfinished objectives)
        self.unfinished_departure_penalty = -5.0  # Base penalty magnitude (scaled by objective.reward_weight)
        self._penalized_departures = set()  # Track (objective_id, from_location)

        logger.info(f"Environment created - Action space: {self.action_space}, Observation space: {self.observation_space.shape}")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to the initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            observation: Initial observation vector
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # RESET REAL: Cargar el estado inicial del emulador
        logger.info(f"Resetting environment - Loading initial state: {self.initial_state_path}")
        self.emulator.load_state(self.initial_state_path)
        
        # Get lightweight state DESPUÉS de resetear el emulador
        lightweight_state = self.state_reader.get_drl_state(map_radius=3)
        self.prev_game_state = lightweight_state
        logger.info(f"Reset complete - emulator restored to initial state")
        
        # Reset tracking variables
        logger.info("Resetting tracking variables...")
        self.current_step = 0
        self.episode_reward = 0.0
        position = lightweight_state.get('position', {})
        self.prev_position = (position.get('x', 0), position.get('y', 0))
        self.stationary_steps = 0
        
        # 🆕 Limpiar caché de diálogos (pueden ser "stale" del estado guardado)
        self.last_dialog = ""
        self.last_dialog_step = 0
        self.captured_dialogues.clear()  # Clear captured dialogues for new episode
        logger.debug("Dialog cache cleared on reset")
        
        logger.info(f"Initial position after reset: {self.prev_position}")
        
        # Extract observation using lightweight reader
        logger.info("Extracting observation...")
        observation = self.state_reader.get_observation_for_drl(map_radius=3)
        logger.info(f"Observation extracted - Map shape: {observation['map'].shape}, Vector shape: {observation['vector'].shape}")
        
        info = {
            'location': 'Unknown',
            'badges': lightweight_state.get('badges', 0),
            'party_size': len(lightweight_state.get('party', []))
        }
        
        logger.info(f"Environment reset - Party size: {info['party_size']}")
        
        return observation, info
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one action in the environment with frame skip (Atari-style).
        
        Args:
            action: Action index (0-7)
            
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off
            info: Additional information
        """
        self.current_step += 1
        
        # Execute action for frame_skip frames
        button = self._action_map[action]
        total_reward = 0.0
        
        # Repeat action for frame_skip frames (Atari-style)
        for _ in range(self.frame_skip):
            self.emulator.run_frame_with_buttons([button])
        
        # Get lightweight observation directly (much faster than get_comprehensive_state!)
        observation = self.state_reader.get_observation_for_drl(map_radius=3)
        
        # For reward calculation, we still need some game state info
        # But we can get it from the lightweight reader
        lightweight_state = self.state_reader.get_drl_state(map_radius=3)
        
        # 🆕 Spatial awareness: Analyze surroundings from map observation
        # This gives us proximity info for doors, NPCs, obstacles, etc.
        if observation is not None and 'map' in observation:
            self.last_spatial_info = self.spatial_analyzer.analyze_surroundings(observation['map'])
        else:
            self.last_spatial_info = None
        
        # Calculate reward using lightweight state
        reward = self._calculate_reward_from_lightweight(self.prev_game_state, lightweight_state)
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = self._check_terminated_from_lightweight(lightweight_state)
        truncated = self.current_step >= self.max_steps
        
        # Prepare info
        info = {
            'location': 'Unknown',  # Location reading is slow, skip for now
            'badges': lightweight_state.get('badges', 0),
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'action_taken': button,
            'frames_executed': self.frame_skip
        }
        
        # Update previous state
        # Departure penalty (location change with unfinished objectives)
        try:
            current_location = None
            if hasattr(self.emulator, 'memory_reader') and self.emulator.memory_reader:
                current_location = self.emulator.memory_reader.read_location()
            prev_location = self.prev_location
            applied_departure_penalty = 0.0
            if current_location and prev_location and current_location != prev_location and self.objectives_manager:
                # Identify location objectives tied to prev_location
                unfinished_objs = self.objectives_manager.get_location_objectives_for(prev_location)
                for obj in unfinished_objs:
                    # Only penalize if not completed and not already penalized for this departure
                    key = (obj.id, prev_location)
                    if not obj.completed and key not in self._penalized_departures:
                        penalty = self.unfinished_departure_penalty * obj.reward_weight
                        reward += penalty  # Apply penalty directly to this step's reward
                        applied_departure_penalty += penalty
                        self._penalized_departures.add(key)
                        logger.info(f"🚫 Departure penalty applied: left '{prev_location}' with unfinished objective '{obj.name}' ({penalty:.2f})")
                if applied_departure_penalty != 0.0:
                    info['departure_penalty'] = applied_departure_penalty
            # Update prev_location to current after processing
            if current_location:
                self.prev_location = current_location
        except Exception as e:
            logger.debug(f"Failed to apply departure penalty: {e}")

        self.prev_game_state = lightweight_state
        
        # 🆕 Chequear y cachear diálogo en cada step (solo si hay texto)
        # Esto es muy ligero - solo lee memoria/screenshot si es necesario
        if self.current_step % 5 == 0:  # Chequear cada 5 steps para no sobrecargar
            self._cache_dialog_if_present()
        
        # 🆕 Hybrid DRL+LLM: Capture dialogues for LLM analysis
        # This must come AFTER _cache_dialog_if_present() to get the latest dialogue
        if self.enable_dialogue_capture and self.objectives_manager:
            dialog = self._get_current_dialog()
            if dialog and dialog not in self.captured_dialogues:
                # New dialogue detected - record it
                location = lightweight_state.get('location', 'Unknown')
                self.objectives_manager.add_dialogue(dialog, location=location)
                logger.info(f"📝 Captured dialogue at {location}: {dialog[:50]}...")
                self.captured_dialogues.add(dialog)  # Track to avoid duplicates
        
        # 🆕 Check location-based objectives periodically
        # Read actual location from memory and verify if objectives are completed
        if self.enable_dialogue_capture and self.objectives_manager and self.current_step % 20 == 0:
            self._check_location_objectives()
        
        return observation, reward, terminated, truncated, info
    
    def _extract_observation(self, game_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Convert game state to observation for the agent.
        
        Returns a dictionary with:
        - 'map': 7x7x3 array for CNN (metatile_id, collision, behavior)
        - 'vector': 18-element vector (position, party, game state)
        """
        player = game_state.get('player', {})
        position = player.get('position', {})
        
        # === Extract map as 7x7x3 image for CNN ===
        map_data = game_state.get('map', {})
        tiles = map_data.get('tiles', [])
        
        # Initialize 7x7x3 map array
        map_array = np.zeros((7, 7, 3), dtype=np.float32)
        
        if tiles and len(tiles) > 0:
            # Get center 7x7 from the full tile grid
            center_i = len(tiles) // 2
            center_j = len(tiles[0]) // 2 if len(tiles) > 0 else 0
            start_i = max(0, center_i - 3)
            start_j = max(0, center_j - 3)
            
            for i in range(7):
                for j in range(7):
                    tile_i = start_i + i
                    tile_j = start_j + j
                    
                    if tile_i < len(tiles) and tile_j < len(tiles[tile_i]):
                        tile = tiles[tile_i][tile_j]
                        if tile and len(tile) >= 3:
                            # Channel 0: metatile_id (normalized)
                            map_array[i, j, 0] = tile[0] / 1000.0
                            
                            # Channel 1: behavior (extract value from enum if needed)
                            behavior = tile[1]
                            if hasattr(behavior, 'value'):
                                behavior_val = behavior.value
                            else:
                                behavior_val = int(behavior) if behavior is not None else 0
                            map_array[i, j, 1] = behavior_val / 255.0
                            
                            # Channel 2: collision (0 or 1)
                            map_array[i, j, 2] = float(tile[2])
        
        # === Extract vector features (18 total) ===
        vector_features = []
        
        # Player position (2 features)
        vector_features.append(position.get('x', 0) / 100.0)
        vector_features.append(position.get('y', 0) / 100.0)
        
        # Party Pokemon (12 features: 6 Pokemon x 2 values)
        party = player.get('party', [])
        for i in range(6):
            if i < len(party):
                pokemon = party[i]
                vector_features.append(pokemon.get('level', 0) / 100.0)
                current_hp = pokemon.get('current_hp', 0)
                max_hp = pokemon.get('max_hp', 1)
                hp_ratio = current_hp / max(max_hp, 1)
                vector_features.append(hp_ratio)
            else:
                vector_features.extend([0.0, 0.0])
        
        # Game state (4 features)
        game = game_state.get('game', {})
        vector_features.append(min(game.get('money', 0) / 999999.0, 1.0))
        
        # Handle badges - can be int or list
        badges = game.get('badges', 0)
        if isinstance(badges, list):
            badge_count = sum(1 for b in badges if b)
        else:
            badge_count = int(badges) if isinstance(badges, (int, float)) else 0
        vector_features.append(badge_count / 8.0)
        
        vector_features.append(1.0 if game.get('is_in_battle', False) else 0.0)
        vector_features.append(min(game.get('pokedex_caught', 0) / 200.0, 1.0))
        
        return {
            'map': map_array,
            'vector': np.array(vector_features, dtype=np.float32)
        }
    
    def _calculate_reward(
        self,
        prev_state: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> float:
        """
        Calculate reward based on state transition.
        """
        reward = 0.0
        
        # Check if game is in a "blocked" state (dialogue, cutscene, etc.)
        # In these states, player can't move, so don't penalize
        is_blocked = self._is_game_blocked(current_state)
        
        # Badge rewards (primary objective)
        prev_badges = self._get_badges(prev_state)
        curr_badges = self._get_badges(current_state)
        if curr_badges > prev_badges:
            reward += 1000.0
            logger.info(f"🏆 Badge obtained! Total badges: {curr_badges}")
        
        # Level up rewards
        prev_levels = self._get_total_party_level(prev_state)
        curr_levels = self._get_total_party_level(current_state)
        if curr_levels > prev_levels:
            reward += 50.0
        
        # Location exploration rewards
        curr_location = self._get_location(current_state)
        if curr_location != self.prev_location:
            if curr_location not in self.visited_locations:
                reward += 20.0
                self.visited_locations.add(curr_location)
            else:
                reward += 5.0
            self.prev_location = curr_location
        
        # Movement rewards (only if not blocked)
        curr_position = self._get_position(current_state)
        if not is_blocked:
            if curr_position != self.prev_position:
                reward += 0.5
                self.stationary_steps = 0
            else:
                self.stationary_steps += 1
                # Penalize being stuck, but less harshly at first
                reward -= 0.05 * min(self.stationary_steps, 20)
        else:
            # In blocked state, reset stationary counter
            self.stationary_steps = 0
        
        self.prev_position = curr_position
        
        # HP penalties (only if not in blocked state to avoid unfair penalties during battles/cutscenes)
        if not is_blocked:
            party = current_state.get('player', {}).get('party', [])
            for pokemon in party:
                current_hp = pokemon.get('current_hp', 0)
                max_hp = pokemon.get('max_hp', 1)
                hp_ratio = current_hp / max(max_hp, 1)
                if hp_ratio < 0.2:
                    reward -= 5.0
                elif hp_ratio < 0.5:
                    reward -= 1.0
        
        return reward
    
    def _is_game_blocked(self, game_state: Dict[str, Any]) -> bool:
        """
        Check if the game is in a state where player input is blocked.
        This includes dialogues, cutscenes, menus, etc.
        """
        game = game_state.get('game_state', {})
        
        # Check for dialogue
        if game.get('in_dialogue', False):
            return True
        
        # Check for battle (partially blocked - different action space)
        if game.get('is_in_battle', False):
            return True
        
        # Check for menu
        if game.get('in_menu', False):
            return True
        
        # Could add more checks:
        # - Animations playing
        # - Cutscenes
        # - Scrolling text
        # For now, these three are the main ones
        
        return False
    
    def _check_terminated(self, game_state: Dict[str, Any]) -> bool:
        """Check if episode should terminate."""
        party = game_state.get('player', {}).get('party', [])
        if party:
            all_fainted = all(p.get('current_hp', 0) == 0 for p in party)
            if all_fainted:
                return True
        
        badges = self._get_badges(game_state)
        if badges >= 8:
            return True
        
        return False
    
    def _get_position(self, game_state: Dict[str, Any]) -> Tuple[int, int]:
        """Get player position as tuple."""
        pos = game_state.get('player', {}).get('position', {})
        return (pos.get('x', 0), pos.get('y', 0))
    
    def _get_location(self, game_state: Dict[str, Any]) -> str:
        """Get current location name."""
        location = game_state.get('player', {}).get('location', '')
        if isinstance(location, dict):
            return location.get('map_name', 'UNKNOWN')
        return str(location) if location else 'UNKNOWN'
    
    def _get_badges(self, game_state: Dict[str, Any]) -> int:
        """Get number of badges."""
        badges = game_state.get('game', {}).get('badges', 0)
        if isinstance(badges, list):
            return sum(1 for b in badges if b)
        return int(badges) if isinstance(badges, int) else 0
    
    def _get_total_party_level(self, game_state: Dict[str, Any]) -> int:
        """Get sum of all party Pokemon levels."""
        party = game_state.get('player', {}).get('party', [])
        return sum(p.get('level', 0) for p in party)
    
    # === Lightweight methods for fast DRL training ===
    
    def _calculate_reward_from_lightweight(
        self,
        prev_state: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> float:
        """
        Calculate reward based on lightweight state (faster than full state).
        Simplified reward function that doesn't require expensive state reads.
        """
        reward = 0.0
        
        # Badge rewards (primary objective)
        prev_badges = prev_state.get('badges', 0) if prev_state else 0
        curr_badges = current_state.get('badges', 0)
        if curr_badges > prev_badges:
            reward += 1000.0
            logger.info(f"🏆 Badge obtained! Total badges: {curr_badges}")
        
        # Level up rewards
        prev_party = prev_state.get('party', []) if prev_state else []
        curr_party = current_state.get('party', [])
        prev_levels = sum(p.get('level', 0) for p in prev_party)
        curr_levels = sum(p.get('level', 0) for p in curr_party)
        if curr_levels > prev_levels:
            reward += 50.0
        
        # Movement rewards
        prev_pos = prev_state.get('position', {}) if prev_state else {}
        curr_pos = current_state.get('position', {})
        prev_coords = (prev_pos.get('x', 0), prev_pos.get('y', 0))
        curr_coords = (curr_pos.get('x', 0), curr_pos.get('y', 0))
        
        # Check if in battle (movement doesn't matter in battle)
        in_battle = current_state.get('in_battle', False)
        
        if not in_battle:
            if curr_coords != prev_coords:
                reward += 0.5
                self.stationary_steps = 0
            else:
                self.stationary_steps += 1
                # Penalize being stuck
                reward -= 0.05 * min(self.stationary_steps, 20)
        else:
            # In battle, reset stationary counter
            self.stationary_steps = 0
        
        # HP penalties (only if not in battle)
        if not in_battle:
            for pokemon in curr_party:
                current_hp = pokemon.get('current_hp', 0)
                max_hp = pokemon.get('max_hp', 1)
                hp_ratio = current_hp / max(max_hp, 1)
                if hp_ratio < 0.2:
                    reward -= 5.0
                elif hp_ratio < 0.5:
                    reward -= 1.0
        
        # 🆕 Hybrid DRL+LLM: Objective-based rewards
        # Check if agent is making progress toward active objectives
        if self.objectives_manager:
            objective_reward = self._calculate_objective_rewards(prev_state, current_state)
            reward += objective_reward
        
        # Apply COMBINED reward multipliers:
        # 1. LLM multiplier (milestone-based, every 1000 steps)
        # 2. Directional multiplier (proximity-based, every 100 steps)
        base_reward = reward
        
        # Combinar ambos multiplicadores (multiplicativo)
        combined_multiplier = self.llm_reward_multiplier * self.directional_multiplier
        reward = reward * combined_multiplier
        
        # Log cuando hay reward shaping activo
        if abs(combined_multiplier - 1.0) > 0.1 and abs(base_reward) > 0.1:
            parts = []
            if self.llm_reward_multiplier != 1.0:
                parts.append(f"LLM:{self.llm_reward_multiplier:.2f}")
            if self.directional_multiplier != 1.0:
                parts.append(f"Dir:{self.directional_multiplier:.2f}")
            
            logger.warning(
                f"💰 Reward shaping: {base_reward:.2f} → {reward:.2f} "
                f"({' × '.join(parts) if parts else f'×{combined_multiplier:.2f}'})"
            )
        
        return reward
    
    def _check_terminated_from_lightweight(self, lightweight_state: Dict[str, Any]) -> bool:
        """Check if episode should terminate using lightweight state."""
        # Check if all Pokemon fainted
        party = lightweight_state.get('party', [])
        if party:
            all_fainted = all(p.get('current_hp', 0) == 0 for p in party)
            if all_fainted:
                return True
        
        # Check if got all badges
        badges = lightweight_state.get('badges', 0)
        if badges >= 8:
            return True
        
        return False
    
    def _calculate_objective_rewards(
        self,
        prev_state: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> float:
        """
        🆕 Calculate rewards based on active objectives set by LLM.
        
        This is where the LLM's strategic planning influences the agent's learning.
        Each objective type has different reward logic.
        """
        objective_reward = 0.0
        
        try:
            active_objectives = self.objectives_manager.get_active_objectives()
            
            if not active_objectives:
                return 0.0
            
            curr_pos = current_state.get('position', {})
            curr_x, curr_y = curr_pos.get('x', 0), curr_pos.get('y', 0)
            
            for objective in active_objectives:
                obj_type = objective.type
                target = objective.target
                weight = objective.reward_weight
                
                # === LOCATION OBJECTIVES ===
                if obj_type == 'location':
                    target_map = target.get('map', '')
                    target_x = target.get('x')
                    target_y = target.get('y')
                    
                    # Reward for being on the target map
                    current_map = self._get_current_map_name()
                    if current_map == target_map:
                        objective_reward += 5.0 * weight
                        
                        # If specific coordinates specified, reward proximity
                        if target_x is not None and target_y is not None:
                            distance = abs(curr_x - target_x) + abs(curr_y - target_y)
                            proximity_reward = max(0, (20 - distance) * 0.5)
                            objective_reward += proximity_reward * weight
                            
                            # Big reward for reaching exact location
                            if distance <= 2:
                                objective_reward += 20.0 * weight
                                # Mark objective as progressing
                                self.objectives_manager.update_progress(objective.id, 0.8)
                        
                        # 🆕 SPATIAL BONUS: Use spatial awareness for proximity rewards
                        if self.last_spatial_info:
                            # Apply spatial proximity multiplier
                            spatial_multiplier = self.spatial_analyzer.calculate_proximity_reward(
                                self.last_spatial_info,
                                obj_type,
                                target,
                                base_reward=1.0
                            )
                            if spatial_multiplier > 1.0:
                                objective_reward *= spatial_multiplier
                            
                            # Check for spatial mission completion
                            is_complete, reason = self.spatial_analyzer.detect_mission_completion(
                                self.last_spatial_info,
                                objective.__dict__,  # Convert to dict
                                current_state
                            )
                            if is_complete:
                                logger.info(f"🎯 Spatial completion detected: {reason}")
                                objective_reward += 50.0 * weight  # Big bonus for completion
                                self.objectives_manager.update_progress(objective.id, 1.0)
                
                # === DIALOGUE OBJECTIVES ===
                elif obj_type == 'dialogue':
                    # Reward for encountering new dialogues
                    if self.last_dialog and (self.current_step - self.last_dialog_step < 10):
                        # Recent dialogue detected
                        objective_reward += 10.0 * weight
                        
                        # Check if dialogue matches target NPC
                        target_npc = target.get('npc', '')
                        if target_npc.lower() in self.last_dialog.lower():
                            objective_reward += 30.0 * weight
                            self.objectives_manager.update_progress(objective.id, 0.9)
                    
                    # 🆕 SPATIAL BONUS: Reward approaching NPCs
                    if self.last_spatial_info and self.last_spatial_info.get('nearby_npcs'):
                        closest_npc_distance = self.last_spatial_info['nearby_npcs'][0][1]
                        if closest_npc_distance < 2.0:
                            proximity_bonus = (2.0 - closest_npc_distance) * 5.0
                            objective_reward += proximity_bonus * weight
                            logger.debug(f"📍 NPC proximity bonus: +{proximity_bonus:.1f} (distance: {closest_npc_distance:.1f})")
                
                # === ITEM OBJECTIVES ===
                elif obj_type == 'item':
                    # Check if item count increased (would need item tracking)
                    # For now, placeholder
                    pass
                
                # === BATTLE OBJECTIVES ===
                elif obj_type == 'battle':
                    in_battle = current_state.get('in_battle', False)
                    if in_battle:
                        objective_reward += 5.0 * weight
                
                # === CUSTOM OBJECTIVES ===
                elif obj_type == 'custom':
                    # Custom objectives can have arbitrary conditions
                    # For now, small encouragement reward
                    objective_reward += 1.0 * weight
            
            # Log significant objective rewards
            if objective_reward > 5.0:
                logger.info(f"🎯 Objective reward: +{objective_reward:.1f} from {len(active_objectives)} objectives")
        
        except Exception as e:
            logger.debug(f"Error calculating objective rewards: {e}")
        
        return objective_reward
    
    # === End lightweight methods ===
    
    def render(self):
        """Render the environment.
        
        Note: For training with visualization, use PygameRenderCallback in train_ppo.py
        instead of calling this method directly. This avoids the PIL.Image.show() 
        issue which spawns external processes and causes system resource exhaustion.
        """
        if self.render_mode == 'human':
            # Don't use screenshot.show() - it spawns external processes
            # Instead, return the screenshot for external rendering (e.g., pygame)
            screenshot = self.emulator.get_screenshot()
            if screenshot:
                return np.array(screenshot)
        elif self.render_mode == 'rgb_array':
            screenshot = self.emulator.get_screenshot()
            if screenshot:
                return np.array(screenshot)
        return None
    
    # Helper methods for LLM reward callback (SubprocVecEnv compatibility)
    def _get_stationary_steps(self) -> int:
        """Get current stationary steps count (for remote access)."""
        return self.stationary_steps
    
    def _get_milestone_count(self) -> int:
        """Get number of completed milestones (for remote access)."""
        if hasattr(self.emulator, 'milestone_tracker'):
            return len(self.emulator.milestone_tracker.milestones)
        return 0
    
    def _get_last_milestone_count(self) -> int:
        """Get last known milestone count (for remote access)."""
        return self.last_milestone_count
    
    def _set_llm_multiplier(self, multiplier: float, advice: str, milestone_count: int):
        """Set LLM reward multiplier and advice (for remote access)."""
        self.llm_reward_multiplier = multiplier
        self.llm_advice = advice
        # Actualizar last_milestone_count para evitar boost persistente
        self.last_milestone_count = milestone_count
    
    def _set_last_milestone_count(self, count: int):
        """Update last milestone count (for remote access)."""
        self.last_milestone_count = count
    
    # Helper methods for Directional reward callback (proximity-based)
    def _get_player_position(self) -> tuple:
        """Get current player position (x, y) for directional guidance."""
        if hasattr(self, 'prev_game_state') and self.prev_game_state:
            pos = self.prev_game_state.get('position', {})
            return (pos.get('x', 0), pos.get('y', 0))
        return (0, 0)
    
    def _get_current_map_name(self) -> str:
        """Get current map name for objective detection."""
        try:
            # Intentar obtener del memory reader
            if hasattr(self.emulator, 'memory_reader'):
                map_name = self.emulator.memory_reader.get_map_name()
                if map_name and map_name != "UNKNOWN":
                    return map_name
            
            # Fallback: usar location si está disponible
            if hasattr(self, 'prev_location') and self.prev_location:
                return self.prev_location
            
            return "UNKNOWN"
        except Exception as e:
            logger.debug(f"Could not get map name: {e}")
            return "UNKNOWN"
    
    def _cache_dialog_if_present(self):
        """
        🆕 Chequear si hay diálogo en pantalla y guardarlo en caché.
        
        Se llama cada 5 steps desde step() para capturar diálogos cuando aparecen,
        sin depender de cuándo el LLM callback decida leerlos.
        
        Usa read_dialog() directo en lugar de OCR fallback para mayor velocidad.
        """
        try:
            if not hasattr(self.emulator, 'memory_reader'):
                return
            
            # 🆕 Ignorar diálogos "stale" en los primeros 50 steps después del reset
            # (pueden ser diálogos viejos del estado guardado que quedaron en memoria)
            if self.current_step < 50:
                logger.debug(f"Ignoring potential stale dialog at step {self.current_step}")
                return
            
            # Leer diálogo directo de memoria (rápido y simple)
            dialog = self.emulator.memory_reader.read_dialog()
            
            # 🆕 Filtrar textos ambientales (NO son útiles como objetivos)
            # PERO mantener: nombres de ubicaciones, nombres de NPCs, diálogos importantes
            if dialog and dialog.strip():
                dialog_lower = dialog.lower()
                
                # Textos que SÍ queremos capturar (locations, NPCs, story text)
                important_patterns = [
                    "town", "route", "city", "lab", "center", "mart", "house",  # Location names
                    "professor", "prof.", "birch", "gym", "leader", "rival",  # NPCs
                    "welcome", "help", "save", "pokemon", "poké", "mom", "dad",  # Story keywords
                    "dangerous", "wild", "grass"  # Important warnings
                ]
                
                # Si contiene patrones importantes, SIEMPRE capturar
                is_important = any(pattern in dialog_lower for pattern in important_patterns)
                
                # Textos triviales que NO queremos (solo si NO son importantes)
                if not is_important:
                    ambient_text_patterns = [
                        "there is a movie on tv", "two men are dancing", "four boys are playing",
                        "it's a nintendo", "game boy", "it's a poster", "it's a map",
                        "it's a bookshelf", "there are books", "it's a clock",
                        "it's a pc", "someone's pc", "it's a trash", "it's a plant",
                        "the water is", "it's a beautiful", "nothing here",
                        "took a closer look", "checked", "examined"
                    ]
                    
                    if any(pattern in dialog_lower for pattern in ambient_text_patterns):
                        logger.debug(f"🚫 Ignoring ambient text: '{dialog[:40]}...'")
                        return  # No cachear texto ambiental
            
            # Solo guardar si hay texto nuevo y diferente al anterior
            if dialog and dialog.strip() and dialog != self.last_dialog:
                self.last_dialog = dialog
                self.last_dialog_step = self.current_step
                # Mark if this is a location text
                is_location = any(word in dialog.lower() for word in ["town", "route", "city"])
                logger.info(f"{'📍' if is_location else '💬'} [Step {self.current_step}] NEW TEXT: '{dialog[:60]}...'")
        except Exception as e:
            # Log errores para saber si hay problemas
            logger.debug(f"Error caching dialog: {e}")
    
    def _get_current_dialog(self) -> str:
        """
        Get current dialogue text from cache (for LLM callback).
        
        Ya NO lee directamente - solo retorna el caché actualizado por _cache_dialog_if_present().
        El caché se actualiza cada 5 steps en step(), capturando diálogos cuando aparecen.
        """
        steps_since_dialog = self.current_step - self.last_dialog_step
        
        if self.last_dialog and steps_since_dialog < self.dialog_cache_duration:
            logger.debug(f"Returning cached dialog from {steps_since_dialog} steps ago")
            return self.last_dialog
        else:
            logger.debug(f"No valid dialog in cache")
            return ""
    
    def _set_directional_multiplier(self, multiplier: float, advice: str):
        """Set directional reward multiplier (from callback)."""
        self.directional_multiplier = multiplier
        self.directional_advice = advice
    
    def _check_location_objectives(self):
        """
        🆕 Read current location from memory and check if it completes any objectives.
        
        This method:
        1. Reads the actual map location from memory (map_bank, map_number)
        2. Gets the location name (e.g., "LITTLEROOT TOWN")
        3. Calls objectives_manager to check if any location-based objectives are completed
        
        Called every 20 steps from step() method.
        """
        try:
            if not hasattr(self.emulator, 'memory_reader'):
                return
            
            memory_reader = self.emulator.memory_reader
            
            # Read raw map coordinates
            map_bank = memory_reader._read_u8(memory_reader.addresses.MAP_BANK)
            map_number = memory_reader._read_u8(memory_reader.addresses.MAP_NUMBER)
            
            # Read location name (uses MapLocation enum internally)
            location_name = memory_reader.read_location()
            
            # Check if this location completes any objectives
            if self.objectives_manager and location_name:
                self.objectives_manager.check_location_objectives(
                    location_name=location_name,
                    map_bank=map_bank,
                    map_number=map_number
                )
                
        except Exception as e:
            logger.debug(f"Failed to check location objectives: {e}")
    
    # 🆕 Hybrid DRL+LLM methods
    def enable_hybrid_mode(self, objectives_manager):
        """Enable hybrid mode with LLM-based objective setting."""
        self.objectives_manager = objectives_manager
        self.enable_dialogue_capture = True
        logger.info("🤖 Hybrid DRL+LLM mode enabled - dialogues will be captured")
    
    def get_current_objectives(self):
        """Get active objectives (for reward shaping)."""
        if self.objectives_manager:
            return self.objectives_manager.get_active_objectives()
        return []
    
    def close(self):
        """Clean up resources."""
        self.emulator.close()
    
    def close(self):
        """Clean up resources."""
        self.emulator.stop()

    # === LLM Snapshot Integration ===
    def get_llm_snapshot(self) -> Dict[str, Any]:
        """Collect a structured snapshot of environment/game state for LLM strategic planning.
        Safe, lightweight aggregation using existing emulator/memory_reader helpers.
        Returns a dictionary with keys: progress, navigation, party, battle, resources, dialogue, scores.
        """
        snapshot: Dict[str, Any] = {
            'progress': {}, 'navigation': {}, 'party': {}, 'battle': {},
            'resources': {}, 'dialogue': {}, 'scores': {}
        }

        try:
            emu = getattr(self, 'emulator', None)
            reader = emu.memory_reader if emu and hasattr(emu, 'memory_reader') else None

            # --- Progress ---
            badges = reader.read_badges() if reader else []
            milestone_tracker = getattr(emu, 'milestone_tracker', None)
            milestones_completed = []
            if milestone_tracker and getattr(milestone_tracker, 'milestones', None):
                milestones_completed = [m for m, data in milestone_tracker.milestones.items() if data.get('completed')]
            snapshot['progress'] = {
                'badge_count': len(badges),
                'badges': badges,
                'milestone_count': len(milestones_completed),
                'milestones_completed': milestones_completed,
                'latest_milestone': getattr(milestone_tracker, 'latest_milestone', None),
                'latest_split': getattr(milestone_tracker, 'latest_split_time', '00:00:00')
            }

            # --- Navigation ---
            position = emu.get_player_position() if emu else None
            location = reader.read_location() if reader else 'UNKNOWN'
            facing = reader.read_player_facing() if reader else None
            map_tiles = emu.get_map_tiles(radius=3) if emu else None
            doors = 0; grass_tiles = 0; water_tiles = 0; passable = 0; blocked = 0
            if map_tiles:
                for row in map_tiles:
                    for tile in row:
                        # tile tuple layout: (metatile_id, behavior, collision, elevation) (may vary)
                        behavior = tile[1] if len(tile) > 1 else None
                        collision = tile[2] if len(tile) > 2 else 0
                        if hasattr(behavior, 'name'):
                            name = behavior.name
                        else:
                            name = str(behavior)
                        if 'DOOR' in name: doors += 1
                        if 'GRASS' in name: grass_tiles += 1
                        if 'WATER' in name: water_tiles += 1
                        if collision == 0: passable += 1
                        else: blocked += 1
            snapshot['navigation'] = {
                'position': position,
                'location': location,
                'facing': facing,
                'doors_nearby': doors,
                'encounter_grass_tiles': grass_tiles,
                'water_tiles': water_tiles,
                'passable_ratio': (passable / (passable + blocked)) if (passable + blocked) > 0 else None
            }

            # --- Party ---
            party = emu.get_party_pokemon() if emu else []
            party_summary = []
            diversity_types = set()
            total_level = 0
            for p in party or []:
                species = p.get('species')
                level = p.get('level', 0)
                max_hp = p.get('max_hp', 1) or 1
                hp_ratio = p.get('current_hp', 0)/max_hp
                status = p.get('status', 'OK')
                types = p.get('types', [])
                for t in types: diversity_types.add(t)
                total_level += level
                party_summary.append({
                    'species': species,
                    'level': level,
                    'hp_ratio': round(hp_ratio, 3),
                    'status': status,
                    'types': types
                })
            avg_level = total_level / len(party) if party else 0
            snapshot['party'] = {
                'size': len(party or []),
                'avg_level': avg_level,
                'type_coverage': sorted(list(diversity_types)),
                'members': party_summary
            }

            # --- Battle ---
            in_battle = reader.is_in_battle() if reader else False
            snapshot['battle'] = {
                'in_battle': in_battle
            }

            # --- Resources ---
            money = emu.get_money() if emu else 0
            items = reader.read_items() if reader else []
            item_dict = {name: qty for name, qty in items}
            pokedex_caught = reader.read_pokedex_caught_count() if reader else 0
            snapshot['resources'] = {
                'money': money,
                'items': item_dict,
                'pokedex_caught': pokedex_caught
            }

            # --- Dialogue ---
            current_dialog = ''
            try:
                if hasattr(self, '_get_current_dialog'):
                    current_dialog = self._get_current_dialog()
            except Exception:
                pass
            snapshot['dialogue'] = {
                'last_dialogue': current_dialog,
                'dialogue_active': bool(current_dialog and len(current_dialog.strip()) > 3)
            }

            # --- Scores (simple composites) ---
            badge_score = min(len(badges)/8.0, 1.0)
            milestone_score = min(len(milestones_completed)/30.0, 1.0)
            progress_score = round(0.6*badge_score + 0.4*milestone_score, 3)
            avg_hp_ratio = 0.0
            if party_summary:
                avg_hp_ratio = sum(m['hp_ratio'] for m in party_summary)/len(party_summary)
            risk_score = round(1.0 - avg_hp_ratio, 3) if party_summary else 0.0
            resource_health = round(min(money/50000.0, 1.0), 3)
            snapshot['scores'] = {
                'progress_score': progress_score,
                'risk_score': risk_score,
                'resource_health': resource_health,
                'badge_score': round(badge_score, 3),
                'milestone_score': round(milestone_score, 3)
            }
        except Exception as e:
            snapshot['error'] = f"snapshot_failed: {e}"

        return snapshot