"""
Test script to demonstrate per-objective dialogue history system
Shows how each objective tracks its own dialogues independently
"""

import json
from agent.objectives_manager import ObjectivesManager, Objective

def test_per_objective_dialogues():
    """Demonstrate the per-objective dialogue tracking system."""
    
    print("=" * 80)
    print("🧪 Testing Per-Objective Dialogue History System")
    print("=" * 80)
    
    # Create manager
    manager = ObjectivesManager("test_objectives_demo.json")
    manager.objectives.clear()
    manager.dialogue_history.clear()
    
    # === PHASE 1: First objective ===
    print("\n📋 PHASE 1: Creating first objective")
    print("-" * 80)
    
    obj1 = Objective(
        id="explore_house",
        name="Explore House",
        description="Look around the starting house",
        type="exploration",
        target={"keywords": ["TV", "PC", "MOM"], "min_interactions": 2}
    )
    manager.objectives[obj1.id] = obj1
    print(f"✅ Created objective: {obj1.name}")
    print(f"   Dialogue history: {len(obj1.dialogue_history)} dialogues")
    
    # Add some dialogues while this objective is active
    print("\n💬 Adding dialogues while 'explore_house' is active...")
    manager.add_dialogue("There is a movie on TV.", location="Player's House")
    manager.add_dialogue("JOHNNY booted up the PC.", location="Player's House")
    manager.add_dialogue("MOM: See you, honey!", npc="MOM", location="Player's House")
    
    print(f"\n📊 After dialogues:")
    print(f"   Global dialogue history: {len(manager.dialogue_history)} dialogues")
    print(f"   'explore_house' dialogue history: {len(obj1.dialogue_history)} dialogues")
    print(f"   Dialogues for explore_house:")
    for i, d in enumerate(obj1.dialogue_history, 1):
        print(f"      {i}. {d['text'][:50]}")
    
    # Complete first objective
    print(f"\n✅ Completing objective: {obj1.name}")
    manager.complete_objective(obj1.id)
    
    # === PHASE 2: Second objective (should have empty dialogue history) ===
    print("\n\n📋 PHASE 2: Creating second objective")
    print("-" * 80)
    
    obj2 = Objective(
        id="meet_prof_birch",
        name="Meet Professor Birch",
        description="Go to Professor Birch's lab",
        type="location",
        target={"keywords": ["PROF. BIRCH", "LAB"], "any_match": True}
    )
    manager.objectives[obj2.id] = obj2
    print(f"✅ Created objective: {obj2.name}")
    print(f"   Dialogue history: {len(obj2.dialogue_history)} dialogues (should be 0)")
    print(f"   ⭐ This objective starts FRESH with no previous dialogues!")
    
    # Add new dialogues while second objective is active
    print("\n💬 Adding NEW dialogues while 'meet_prof_birch' is active...")
    manager.add_dialogue("PROF. BIRCH'S POKéMON LAB", location="Outside Lab")
    manager.add_dialogue("PROF. BIRCH: Welcome to the world of Pokemon!", npc="PROF. BIRCH", location="Lab")
    
    print(f"\n📊 After new dialogues:")
    print(f"   Global dialogue history: {len(manager.dialogue_history)} dialogues (accumulates)")
    print(f"   'explore_house' (completed) dialogue history: {len(obj1.dialogue_history)} dialogues")
    print(f"   'meet_prof_birch' (active) dialogue history: {len(obj2.dialogue_history)} dialogues")
    
    print(f"\n   Dialogues for meet_prof_birch (only recent ones):")
    for i, d in enumerate(obj2.dialogue_history, 1):
        print(f"      {i}. {d['text'][:50]}")
    
    # === PHASE 3: LLM Perspective ===
    print("\n\n🤖 PHASE 3: What the LLM sees")
    print("-" * 80)
    
    print("\n1️⃣ When LLM calls read_objectives():")
    llm_view = {
        'active_objectives': [obj.to_dict() for obj in manager.get_active_objectives()],
        'completed_objectives': [obj.to_dict() for obj in manager.get_completed_objectives()]
    }
    
    print(f"\n   Active objectives: {len(llm_view['active_objectives'])}")
    for obj_data in llm_view['active_objectives']:
        print(f"   - {obj_data['name']}: {len(obj_data['dialogue_history'])} dialogues in context")
        print(f"     Dialogues:")
        for d in obj_data['dialogue_history']:
            print(f"       • {d['text'][:50]}")
    
    print(f"\n   Completed objectives: {len(llm_view['completed_objectives'])}")
    for obj_data in llm_view['completed_objectives']:
        print(f"   - {obj_data['name']}: {len(obj_data['dialogue_history'])} dialogues (frozen at completion)")
    
    print("\n2️⃣ When LLM evaluates 'meet_prof_birch':")
    print("   It ONLY sees these dialogues:")
    print("   - 'PROF. BIRCH'S POKéMON LAB'")
    print("   - 'PROF. BIRCH: Welcome to the world of Pokemon!'")
    print("   ⭐ It does NOT see dialogues from 'explore_house' objective!")
    print("   ⭐ This prevents context pollution and focuses the LLM's analysis!")
    
    print("\n3️⃣ When LLM creates a new objective:")
    print("   - New objective starts with dialogue_history = []")
    print("   - It will only track dialogues from that point forward")
    print("   - Previous objectives' dialogues are isolated")
    
    # === Summary ===
    print("\n\n" + "=" * 80)
    print("✅ SUMMARY: Per-Objective Dialogue Tracking")
    print("=" * 80)
    print("\n🎯 Key Benefits:")
    print("   1. Each objective has FOCUSED context (only relevant dialogues)")
    print("   2. No context pollution from previous objectives")
    print("   3. LLM can make clear decisions based on CURRENT objective's progress")
    print("   4. New objectives start fresh (empty dialogue_history)")
    print("   5. Global history still available via read_dialogues() if needed")
    
    print("\n🔧 How it works:")
    print("   - When dialogue appears: added to ALL active objectives + global history")
    print("   - When objective completes: its dialogue_history is frozen")
    print("   - When new objective created: starts with empty dialogue_history")
    print("   - LLM reads objective: sees ONLY that objective's dialogue_history")
    
    print("\n🚀 Usage in training:")
    print("   1. LLM calls read_objectives() → sees each objective's dialogue_history")
    print("   2. LLM calls evaluate_objective_completion(id) → analyzes that objective's dialogues")
    print("   3. LLM calls complete_objective(id) → if criteria met")
    print("   4. LLM calls write_objective() → creates new objective (empty dialogue_history)")
    
    # Save final state
    manager.save()
    print(f"\n💾 Saved test state to: test_objectives_demo.json")
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_per_objective_dialogues()
