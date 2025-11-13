"""
LLM Tools for Hybrid DRL+LLM System

Tools that the LLM can use to interact with the training system:
- Read objectives
- Write new objectives
- Read dialogues
- Update policy parameters
"""

import json
import logging
from typing import Dict, List, Any, Optional
from agent.objectives_manager import ObjectivesManager, Objective

logger = logging.getLogger(__name__)


class LLMTools:
    """
    Tool collection for LLM to interact with the training system.
    These tools are exposed to the LLM via function calling.
    """
    
    def __init__(self, objectives_manager: ObjectivesManager):
        self.objectives_manager = objectives_manager
        self.policy_params = {
            'reward_weight_exploration': 1.0,
            'reward_weight_dialogue': 1.5,
            'reward_weight_milestone': 2.0,
            'penalty_stationary': -0.1,
            'penalty_revisit': -0.05
        }
    
    # ===== TOOL: Read Objectives =====
    def read_objectives(self) -> dict:
        """
        Read the current objectives and their status.
        
        Returns a JSON object containing:
        - active_objectives: List of incomplete objectives
        - completed_objectives: List of completed objectives
        - recent_dialogues: Recent in-game dialogues
        """
        return {
            'active_objectives': [obj.to_dict() for obj in self.objectives_manager.get_active_objectives()],
            'completed_objectives': [obj.to_dict() for obj in self.objectives_manager.get_completed_objectives()],
            'recent_dialogues': self.objectives_manager.get_recent_dialogues(10)
        }
    
    # ===== TOOL: Write Objective =====
    def write_objective(
        self,
        name: str,
        description: str,
        type: str,
        target: dict,
        reward_weight: float = 1.0
    ) -> dict:
        """
        Create a new objective for the agent to pursue.
        
        Args:
            name: Short name for the objective (e.g., "Talk to Professor Birch")
            description: Detailed description of what needs to be done
            type: Type of objective - one of: 'location', 'dialogue', 'item', 'battle', 'custom'
            target: Dictionary with specific target data. Examples:
                - For location: {"map": "ROUTE_101", "x": 10, "y": 5}
                - For dialogue: {"npc": "PROF_BIRCH", "trigger": "talk"}
                - For item: {"item_id": "POKE_BALL", "count": 5}
            reward_weight: How much to weight this objective (default 1.0, higher = more important)
        
        Returns:
            Success/failure status
        """
        # Generate unique ID
        obj_id = name.lower().replace(' ', '_')
        
        # Create objective
        objective = Objective(
            id=obj_id,
            name=name,
            description=description,
            type=type,
            target=target,
            reward_weight=reward_weight
        )
        
        success = self.objectives_manager.add_objective(objective)
        
        return {
            'success': success,
            'objective_id': obj_id,
            'message': f"Objective '{name}' created successfully" if success else f"Failed to create objective '{name}'"
        }
    
    # ===== TOOL: Mark Objective Complete =====
    def complete_objective(self, objective_id: str) -> dict:
        """
        Mark an objective as completed.
        
        Args:
            objective_id: The ID of the objective to complete
        
        Returns:
            Success/failure status
        """
        success = self.objectives_manager.complete_objective(objective_id)
        return {
            'success': success,
            'message': f"Objective {objective_id} marked complete" if success else f"Failed to complete objective {objective_id}"
        }
    
    # ===== TOOL: Read Recent Dialogues =====
    def read_dialogues(self, count: int = 10) -> dict:
        """
        Read recent in-game dialogues that the agent has encountered.
        
        Args:
            count: Number of recent dialogues to retrieve (default 10)
        
        Returns:
            List of dialogue entries with text, NPC, location, and timestamp
        """
        dialogues = self.objectives_manager.get_recent_dialogues(count)
        return {
            'dialogues': dialogues,
            'count': len(dialogues)
        }
    
    # ===== TOOL: Update Policy Parameters =====
    def update_policy(
        self,
        reward_weight_exploration: Optional[float] = None,
        reward_weight_dialogue: Optional[float] = None,
        reward_weight_milestone: Optional[float] = None,
        penalty_stationary: Optional[float] = None,
        penalty_revisit: Optional[float] = None
    ) -> dict:
        """
        Update policy parameters to adjust the agent's learning behavior.
        
        Args:
            reward_weight_exploration: Weight for exploring new areas (default 1.0)
            reward_weight_dialogue: Weight for engaging in dialogues (default 1.5)
            reward_weight_milestone: Weight for completing milestones (default 2.0)
            penalty_stationary: Penalty for staying in one place (default -0.1)
            penalty_revisit: Penalty for revisiting same locations (default -0.05)
        
        Returns:
            Updated policy parameters
        """
        if reward_weight_exploration is not None:
            self.policy_params['reward_weight_exploration'] = reward_weight_exploration
        if reward_weight_dialogue is not None:
            self.policy_params['reward_weight_dialogue'] = reward_weight_dialogue
        if reward_weight_milestone is not None:
            self.policy_params['reward_weight_milestone'] = reward_weight_milestone
        if penalty_stationary is not None:
            self.policy_params['penalty_stationary'] = penalty_stationary
        if penalty_revisit is not None:
            self.policy_params['penalty_revisit'] = penalty_revisit
        
        logger.info(f"Policy parameters updated: {self.policy_params}")
        
        return {
            'success': True,
            'policy_params': self.policy_params
        }
    
    # ===== TOOL: Get Training Stats =====
    def get_training_stats(
        self,
        episode_rewards: List[float],
        episode_lengths: List[int],
        completed_milestones: int
    ) -> dict:
        """
        Get current training statistics.
        
        Args:
            episode_rewards: List of recent episode rewards
            episode_lengths: List of recent episode lengths
            completed_milestones: Number of milestones completed
        
        Returns:
            Training statistics summary
        """
        if not episode_rewards:
            return {
                'avg_reward': 0,
                'avg_length': 0,
                'completed_milestones': completed_milestones
            }
        
        return {
            'avg_reward': sum(episode_rewards) / len(episode_rewards),
            'max_reward': max(episode_rewards),
            'min_reward': min(episode_rewards),
            'avg_length': sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0,
            'completed_milestones': completed_milestones,
            'total_episodes': len(episode_rewards)
        }


def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions in Ollama's function calling format.
    These define what tools the LLM can use.
    """
    return [
        {
            'type': 'function',
            'function': {
                'name': 'read_objectives',
                'description': 'Read the current objectives and their completion status. Use this to understand what the agent is trying to accomplish.',
                'parameters': {
                    'type': 'object',
                    'properties': {},
                    'required': []
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'write_objective',
                'description': 'Create a new objective for the agent to pursue. Use this after reading dialogues or observing game progress.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'name': {
                            'type': 'string',
                            'description': 'Short name for the objective (e.g., "Talk to Professor Birch")'
                        },
                        'description': {
                            'type': 'string',
                            'description': 'Detailed description of what needs to be done'
                        },
                        'type': {
                            'type': 'string',
                            'enum': ['location', 'dialogue', 'item', 'battle', 'custom'],
                            'description': 'Type of objective'
                        },
                        'target': {
                            'type': 'object',
                            'description': 'Specific target data (varies by type)'
                        },
                        'reward_weight': {
                            'type': 'number',
                            'description': 'How important this objective is (default 1.0, higher = more important)'
                        }
                    },
                    'required': ['name', 'description', 'type', 'target']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'complete_objective',
                'description': 'Mark an objective as completed when the agent has achieved it.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'objective_id': {
                            'type': 'string',
                            'description': 'The ID of the objective to complete'
                        }
                    },
                    'required': ['objective_id']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'read_dialogues',
                'description': 'Read recent in-game dialogues to understand story context and set new objectives.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'count': {
                            'type': 'integer',
                            'description': 'Number of recent dialogues to retrieve (default 10)'
                        }
                    },
                    'required': []
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'update_policy',
                'description': 'Adjust policy parameters to change how the agent learns. Use this to emphasize/de-emphasize different behaviors.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'reward_weight_exploration': {
                            'type': 'number',
                            'description': 'Weight for exploring new areas (higher = more exploration)'
                        },
                        'reward_weight_dialogue': {
                            'type': 'number',
                            'description': 'Weight for engaging in dialogues (higher = seek more NPCs)'
                        },
                        'reward_weight_milestone': {
                            'type': 'number',
                            'description': 'Weight for completing milestones (higher = focus on objectives)'
                        },
                        'penalty_stationary': {
                            'type': 'number',
                            'description': 'Penalty for staying in one place (more negative = discourage standing still)'
                        },
                        'penalty_revisit': {
                            'type': 'number',
                            'description': 'Penalty for revisiting locations (more negative = encourage exploring new areas)'
                        }
                    },
                    'required': []
                }
            }
        }
    ]
