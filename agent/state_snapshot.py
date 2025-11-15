"""State Snapshot Aggregator

Provides a lightweight structured view of emulator + environment state for
LLM-guided strategic planning and reward shaping. Designed to minimize
memory reads and reuse existing helper methods.

Usage:
    aggregator = StateSnapshotAggregator()
    data = aggregator.collect(env)

The snapshot focuses on seven domains:
    progress, navigation, party, battle, resources, dialogue, scores.

All fields are defensive: failures produce nulls but keep schema stable.
"""
from __future__ import annotations

from typing import Any, Dict


class StateSnapshotAggregator:
    def __init__(self):
        self.last_position = None
        self.last_money = None
        self.last_badge_count = 0
        self.last_milestone_count = 0

    def collect(self, env) -> Dict[str, Any]:
        emu = getattr(env, 'emulator', None)
        reader = emu.memory_reader if emu and hasattr(emu, 'memory_reader') else None
        snapshot: Dict[str, Any] = {
            'progress': {}, 'navigation': {}, 'party': {}, 'battle': {},
            'resources': {}, 'dialogue': {}, 'scores': {}
        }

        try:
            # Progress
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

            # Navigation
            position = emu.get_player_position() if emu else None
            location = reader.read_location() if reader else 'UNKNOWN'
            facing = reader.read_player_facing() if reader else None
            # Movement delta magnitude (simple heuristic)
            move_delta = None
            if position and self.last_position:
                try:
                    move_delta = abs(position['x'] - self.last_position['x']) + abs(position['y'] - self.last_position['y'])
                except Exception:
                    move_delta = None
            self.last_position = position
            snapshot['navigation'] = {
                'position': position,
                'location': location,
                'facing': facing,
                'move_delta': move_delta
            }

            # Party
            party = emu.get_party_pokemon() if emu else []
            party_summary = []
            diversity = set()
            total_level = 0
            total_hp_ratio = 0.0
            for p in party or []:
                level = p.get('level', 0)
                max_hp = p.get('max_hp', 1) or 1
                hp_ratio = (p.get('current_hp', 0) / max_hp)
                total_level += level
                total_hp_ratio += hp_ratio
                types = p.get('types', [])
                for t in types: diversity.add(t)
                party_summary.append({
                    'species': p.get('species'),
                    'level': level,
                    'hp_ratio': round(hp_ratio, 3),
                    'status': p.get('status', 'OK'),
                    'types': types
                })
            avg_level = total_level / len(party) if party else 0
            avg_hp_ratio = total_hp_ratio / len(party) if party else 0
            snapshot['party'] = {
                'size': len(party),
                'avg_level': avg_level,
                'avg_hp_ratio': round(avg_hp_ratio, 3),
                'type_coverage': sorted(list(diversity)),
                'members': party_summary
            }

            # Battle
            in_battle = reader.is_in_battle() if reader else False
            snapshot['battle'] = {
                'in_battle': in_battle
            }

            # Resources
            money = emu.get_money() if emu else 0
            money_delta = money - self.last_money if self.last_money is not None else 0
            self.last_money = money
            items = reader.read_items() if reader else []
            item_dict = {name: qty for name, qty in items}
            pokedex_caught = reader.read_pokedex_caught_count() if reader else 0
            snapshot['resources'] = {
                'money': money,
                'money_delta': money_delta,
                'items': item_dict,
                'pokedex_caught': pokedex_caught
            }

            # Dialogue
            dialog_text = ''
            if hasattr(env, '_get_current_dialog'):
                try:
                    dialog_text = env._get_current_dialog()
                except Exception:
                    dialog_text = ''
            snapshot['dialogue'] = {
                'active': bool(dialog_text and len(dialog_text.strip()) > 3),
                'text': dialog_text[:300] if dialog_text else ''
            }

            # Scores (basic composites)
            badge_score = min(len(badges)/8.0, 1.0)
            milestone_score = min(len(milestones_completed)/30.0, 1.0)
            progress_score = round(0.6*badge_score + 0.4*milestone_score, 3)
            risk_score = round(1.0 - avg_hp_ratio, 3) if party else 0.0
            resource_health = round(min(money/50000.0, 1.0), 3)
            snapshot['scores'] = {
                'progress_score': progress_score,
                'risk_score': risk_score,
                'resource_health': resource_health,
                'badge_score': round(badge_score, 3),
                'milestone_score': round(milestone_score, 3)
            }

        except Exception as e:
            snapshot['error'] = f'aggregation_failed: {e}'

        return snapshot
