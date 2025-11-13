"""
Objectives Manager for Hybrid DRL+LLM System

Manages game objectives that guide the DRL agent's learning.
The LLM can read and modify objectives based on game progress.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class Objective:
    """Represents a single game objective."""
    
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        type: str,  # 'location', 'dialogue', 'item', 'battle', 'custom'
        target: Dict[str, Any],  # Specific target data (location coords, NPC name, etc.)
        reward_weight: float = 1.0,
        completed: bool = False,
        progress: float = 0.0,
        created_at: Optional[str] = None,
        completed_at: Optional[str] = None
    ):
        self.id = id
        self.name = name
        self.description = description
        self.type = type
        self.target = target
        self.reward_weight = reward_weight
        self.completed = completed
        self.progress = progress
        self.created_at = created_at or datetime.now().isoformat()
        self.completed_at = completed_at
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'type': self.type,
            'target': self.target,
            'reward_weight': self.reward_weight,
            'completed': self.completed,
            'progress': self.progress,
            'created_at': self.created_at,
            'completed_at': self.completed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Objective':
        """Create objective from dictionary."""
        return cls(**data)


class ObjectivesManager:
    """
    Manages the current objectives for the DRL agent.
    
    The LLM can use this to:
    - Read current objectives
    - Add new objectives based on game progress
    - Mark objectives as completed
    - Adjust reward weights
    """
    
    def __init__(self, objectives_file: str = "agent/current_objectives.json"):
        self.objectives_file = Path(objectives_file)
        self.objectives: Dict[str, Objective] = {}
        self.dialogue_history: List[Dict[str, Any]] = []
        self.load()
        
    def load(self):
        """Load objectives from file."""
        if self.objectives_file.exists():
            try:
                with open(self.objectives_file, 'r') as f:
                    data = json.load(f)
                    self.objectives = {
                        obj_id: Objective.from_dict(obj_data)
                        for obj_id, obj_data in data.get('objectives', {}).items()
                    }
                    self.dialogue_history = data.get('dialogue_history', [])
                    logger.info(f"Loaded {len(self.objectives)} objectives from {self.objectives_file}")
            except Exception as e:
                logger.error(f"Failed to load objectives: {e}")
                self._initialize_default()
        else:
            logger.info("No objectives file found, initializing with defaults")
            self._initialize_default()
    
    def save(self):
        """Save objectives to file."""
        self.objectives_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'objectives': {
                obj_id: obj.to_dict()
                for obj_id, obj in self.objectives.items()
            },
            'dialogue_history': self.dialogue_history,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.objectives_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.objectives)} objectives to {self.objectives_file}")
    
    def _initialize_default(self):
        """Initialize with more achievable starting objectives based on actual game text."""
        # Objective 1: Explore the house (easy to complete)
        explore_house = Objective(
            id="explore_starting_house",
            name="Explore Starting House",
            description="Look around the house and interact with objects (PC, TV, stairs)",
            type="exploration",
            target={
                "keywords": ["PC", "TV", "HOUSE", "upstairs", "MOM"],
                "min_interactions": 3
            },
            reward_weight=1.5
        )
        self.objectives[explore_house.id] = explore_house
        
        # Objective 2: Meet Professor Birch (based on Mom's dialogue)
        meet_birch = Objective(
            id="meet_prof_birch",
            name="Meet Professor Birch",
            description="Go next door to Prof. Birch's house or lab as Mom suggested",
            type="location",
            target={
                "keywords": ["PROF. BIRCH", "BIRCH'S HOUSE", "BIRCH'S LAB", "POKéMON LAB"],
                "any_match": True
            },
            reward_weight=2.0
        )
        self.objectives[meet_birch.id] = meet_birch
        
        self.save()
    
    def add_objective(self, objective: Objective) -> bool:
        """Add a new objective."""
        if objective.id in self.objectives:
            logger.warning(f"Objective {objective.id} already exists")
            return False
        self.objectives[objective.id] = objective
        self.save()
        logger.info(f"Added objective: {objective.name}")
        return True
    
    def complete_objective(self, objective_id: str) -> bool:
        """Mark an objective as completed."""
        if objective_id not in self.objectives:
            logger.warning(f"Objective {objective_id} not found")
            return False
        
        obj = self.objectives[objective_id]
        if not obj.completed:
            obj.completed = True
            obj.progress = 1.0
            obj.completed_at = datetime.now().isoformat()
            self.save()
            logger.info(f"Completed objective: {obj.name}")
            return True
        return False
    
    def update_progress(self, objective_id: str, progress: float):
        """Update progress for an objective (0.0 to 1.0)."""
        if objective_id in self.objectives:
            self.objectives[objective_id].progress = max(0.0, min(1.0, progress))
            self.save()
    
    def get_active_objectives(self) -> List[Objective]:
        """Get all incomplete objectives."""
        return [obj for obj in self.objectives.values() if not obj.completed]
    
    def get_completed_objectives(self) -> List[Objective]:
        """Get all completed objectives."""
        return [obj for obj in self.objectives.values() if obj.completed]
    
    def add_dialogue(self, text: str, npc: Optional[str] = None, location: Optional[str] = None):
        """Record a dialogue that the agent encountered and check objective completion."""
        dialogue_entry = {
            'text': text,
            'npc': npc,
            'location': location,
            'timestamp': datetime.now().isoformat()
        }
        self.dialogue_history.append(dialogue_entry)
        
        # 🆕 Auto-check objectives based on new dialogue
        self._check_dialogue_objectives(text)
        
        # Keep only last 50 dialogues to avoid file bloat
        if len(self.dialogue_history) > 50:
            self.dialogue_history = self.dialogue_history[-50:]
        self.save()
    
    def get_recent_dialogues(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the N most recent dialogues."""
        return self.dialogue_history[-n:]
    
    def to_json_str(self) -> str:
        """Get a JSON string representation for LLM consumption."""
        return json.dumps({
            'active_objectives': [obj.to_dict() for obj in self.get_active_objectives()],
            'completed_objectives': [obj.to_dict() for obj in self.get_completed_objectives()],
            'recent_dialogues': self.get_recent_dialogues()
        }, indent=2)
    
    def _check_dialogue_objectives(self, text: str):
        """
        Check if new dialogue completes any objectives.
        Auto-completes keyword-based objectives.
        """
        text_upper = text.upper()
        
        for obj_id, obj in self.objectives.items():
            if obj.completed:
                continue
            
            # Check keyword-based objectives
            target = obj.target
            if isinstance(target, dict) and 'keywords' in target:
                keywords = target['keywords']
                any_match = target.get('any_match', False)
                
                if any_match:
                    # Complete if ANY keyword is found
                    if any(keyword.upper() in text_upper for keyword in keywords):
                        logger.info(f"✅ Objective '{obj.name}' completed by dialogue: '{text[:50]}...'")
                        self.complete_objective(obj_id)
                else:
                    # Count how many unique keywords we've seen
                    matched_keywords = sum(1 for kw in keywords if any(kw.upper() in d['text'].upper() for d in self.dialogue_history))
                    min_interactions = target.get('min_interactions', len(keywords))
                    
                    progress = min(1.0, matched_keywords / min_interactions)
                    self.update_progress(obj_id, progress)
                    
                    if progress >= 1.0:
                        logger.info(f"✅ Objective '{obj.name}' completed ({matched_keywords}/{min_interactions} interactions)")
                        self.complete_objective(obj_id)
    
    def check_location_objectives(self, location_name: str, map_bank: int = None, map_number: int = None):
        """
        Check if current location completes any objectives.
        
        Args:
            location_name: Name of the current location (e.g., "LITTLEROOT TOWN")
            map_bank: Raw map bank value (optional)
            map_number: Raw map number value (optional)
        """
        if not location_name:
            return
        
        location_upper = location_name.upper()
        
        for obj_id, obj in self.objectives.items():
            if obj.completed:
                continue
            
            # Check location-based objectives
            if obj.type == 'location':
                target = obj.target
                
                # Method 1: Check by location name keywords
                if isinstance(target, dict) and 'keywords' in target:
                    keywords = target['keywords']
                    any_match = target.get('any_match', True)  # Default to any_match for locations
                    
                    if any_match:
                        # Complete if ANY keyword matches location
                        if any(keyword.upper() in location_upper for keyword in keywords):
                            logger.info(f"✅ Objective '{obj.name}' completed by reaching location: '{location_name}'")
                            self.complete_objective(obj_id)
                
                # Method 2: Check by exact location name match
                elif isinstance(target, dict) and 'location' in target:
                    target_location = target['location'].upper()
                    if target_location in location_upper or location_upper in target_location:
                        logger.info(f"✅ Objective '{obj.name}' completed by reaching: '{location_name}'")
                        self.complete_objective(obj_id)
                
                # Method 3: Check by map coordinates (if provided)
                elif isinstance(target, dict) and 'map_bank' in target and map_bank is not None:
                    if (target.get('map_bank') == map_bank and 
                        target.get('map_number') == map_number):
                        logger.info(f"✅ Objective '{obj.name}' completed by reaching map {map_bank:02X}_{map_number:02X}")
                        self.complete_objective(obj_id)
    
    def reset(self, keep_dialogues: bool = False):
        """
        Reset objectives to default state.
        
        Args:
            keep_dialogues: If True, keep dialogue history; if False, clear it
        """
        logger.info("🔄 Resetting objectives to default state...")
        
        # Clear all objectives
        self.objectives.clear()
        
        # Clear dialogue history unless specified to keep
        if not keep_dialogues:
            self.dialogue_history.clear()
        
        # Reinitialize with default objective
        self._initialize_default()
        
        logger.info(f"✅ Objectives reset complete. Dialogues {'kept' if keep_dialogues else 'cleared'}.")
