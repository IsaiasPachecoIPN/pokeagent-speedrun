"""
Simple standalone test for per-objective dialogue system
(No game dependencies required)
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Simplified Objective class for demo
class SimpleObjective:
    def __init__(self, id, name, description, dialogue_history=None):
        self.id = id
        self.name = name
        self.description = description
        self.dialogue_history = dialogue_history or []
        self.completed = False
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'dialogue_history': self.dialogue_history,
            'completed': self.completed,
            'created_at': self.created_at
        }

def test_per_objective_dialogues():
    """Demonstrate the per-objective dialogue tracking system."""
    
    print("=" * 80)
    print("🧪 Testing Per-Objective Dialogue History System")
    print("=" * 80)
    
    # Simulated manager state
    objectives = {}
    global_dialogue_history = []
    
    # === PHASE 1: First objective ===
    print("\n📋 PHASE 1: Creating first objective")
    print("-" * 80)
    
    obj1 = SimpleObjective(
        id="explore_house",
        name="Explore House",
        description="Look around the starting house"
    )
    objectives[obj1.id] = obj1
    print(f"✅ Created objective: {obj1.name}")
    print(f"   Dialogue history: {len(obj1.dialogue_history)} dialogues")
    
    # Simulate adding dialogues
    print("\n💬 Adding dialogues while 'explore_house' is active...")
    dialogues_phase1 = [
        {"text": "There is a movie on TV.", "location": "Player's House"},
        {"text": "JOHNNY booted up the PC.", "location": "Player's House"},
        {"text": "MOM: See you, honey!", "location": "Player's House"}
    ]
    
    for dialogue in dialogues_phase1:
        dialogue['timestamp'] = datetime.now().isoformat()
        global_dialogue_history.append(dialogue)
        # Add to active objectives only
        for obj in objectives.values():
            if not obj.completed:
                obj.dialogue_history.append(dialogue)
    
    print(f"\n📊 After dialogues:")
    print(f"   Global dialogue history: {len(global_dialogue_history)} dialogues")
    print(f"   'explore_house' dialogue history: {len(obj1.dialogue_history)} dialogues")
    print(f"   Dialogues for explore_house:")
    for i, d in enumerate(obj1.dialogue_history, 1):
        print(f"      {i}. {d['text']}")
    
    # Complete first objective
    print(f"\n✅ Marking objective as completed: {obj1.name}")
    obj1.completed = True
    
    # === PHASE 2: Second objective (should have empty dialogue history) ===
    print("\n\n📋 PHASE 2: Creating second objective")
    print("-" * 80)
    
    obj2 = SimpleObjective(
        id="meet_prof_birch",
        name="Meet Professor Birch",
        description="Go to Professor Birch's lab"
    )
    objectives[obj2.id] = obj2
    print(f"✅ Created objective: {obj2.name}")
    print(f"   Dialogue history: {len(obj2.dialogue_history)} dialogues (should be 0)")
    print(f"   ⭐ This objective starts FRESH with no previous dialogues!")
    
    # Add new dialogues while second objective is active
    print("\n💬 Adding NEW dialogues while 'meet_prof_birch' is active...")
    dialogues_phase2 = [
        {"text": "PROF. BIRCH'S POKéMON LAB", "location": "Outside Lab"},
        {"text": "PROF. BIRCH: Welcome to the world of Pokemon!", "location": "Lab"}
    ]
    
    for dialogue in dialogues_phase2:
        dialogue['timestamp'] = datetime.now().isoformat()
        global_dialogue_history.append(dialogue)
        # Add to active objectives only (obj1 is completed, so only obj2)
        for obj in objectives.values():
            if not obj.completed:
                obj.dialogue_history.append(dialogue)
    
    print(f"\n📊 After new dialogues:")
    print(f"   Global dialogue history: {len(global_dialogue_history)} dialogues (accumulates)")
    print(f"   'explore_house' (completed) dialogue history: {len(obj1.dialogue_history)} dialogues")
    print(f"   'meet_prof_birch' (active) dialogue history: {len(obj2.dialogue_history)} dialogues")
    
    print(f"\n   Dialogues for meet_prof_birch (only recent ones):")
    for i, d in enumerate(obj2.dialogue_history, 1):
        print(f"      {i}. {d['text']}")
    
    # === PHASE 3: LLM Perspective ===
    print("\n\n🤖 PHASE 3: What the LLM sees when calling read_objectives()")
    print("-" * 80)
    
    active_objectives = [obj.to_dict() for obj in objectives.values() if not obj.completed]
    completed_objectives = [obj.to_dict() for obj in objectives.values() if obj.completed]
    
    print(f"\n   Active objectives: {len(active_objectives)}")
    for obj_data in active_objectives:
        print(f"   📌 {obj_data['name']}: {len(obj_data['dialogue_history'])} dialogues in context")
        print(f"      Dialogues:")
        for d in obj_data['dialogue_history']:
            print(f"         • {d['text']}")
    
    print(f"\n   Completed objectives: {len(completed_objectives)}")
    for obj_data in completed_objectives:
        print(f"   ✅ {obj_data['name']}: {len(obj_data['dialogue_history'])} dialogues (frozen)")
    
    print("\n" + "=" * 80)
    print("🎯 KEY INSIGHT:")
    print("=" * 80)
    print("\nWhen LLM evaluates 'meet_prof_birch', it ONLY sees:")
    for d in obj2.dialogue_history:
        print(f"   • {d['text']}")
    
    print("\n❌ It does NOT see dialogues from 'explore_house':")
    for d in obj1.dialogue_history:
        print(f"   • {d['text']}")
    
    print("\n✅ This prevents context pollution!")
    print("✅ LLM makes focused decisions based on current objective's dialogues only!")
    
    # === Summary ===
    print("\n\n" + "=" * 80)
    print("📋 SUMMARY: How This Helps the LLM")
    print("=" * 80)
    
    print("\n🎯 Benefits:")
    print("   1. ✨ FOCUSED CONTEXT - Each objective has only relevant dialogues")
    print("   2. 🚫 NO POLLUTION - Past objectives don't confuse current decisions")
    print("   3. 🎪 CLEAR DECISIONS - LLM evaluates based on current objective's progress")
    print("   4. 🆕 FRESH START - New objectives begin with empty dialogue_history")
    print("   5. 📚 GLOBAL ACCESS - LLM can still call read_dialogues() for general context")
    
    print("\n🔧 Workflow:")
    print("   1. LLM: read_objectives() → sees 'meet_prof_birch' with 2 dialogues")
    print("   2. LLM: evaluate_objective_completion('meet_prof_birch', reasoning)")
    print("   3. LLM analyzes: 'PROF. BIRCH' keyword found in dialogue → complete!")
    print("   4. LLM: complete_objective('meet_prof_birch')")
    print("   5. LLM: write_objective('choose_starter') → new objective, empty dialogues")
    print("   6. System: Future dialogues added to 'choose_starter' only")
    
    print("\n🚀 Result:")
    print("   • Each objective represents ONE story beat")
    print("   • LLM creates objectives sequentially based on story progression")
    print("   • No confusion from mixing different story phases")
    print("   • Clean, focused AI decision-making!")
    
    print("\n✅ Test complete!\n")


if __name__ == "__main__":
    test_per_objective_dialogues()
