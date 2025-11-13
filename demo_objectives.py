#!/usr/bin/env python3
"""Show new objectives system"""
import sys
sys.path.insert(0, '.')

from agent.objectives_manager import ObjectivesManager

print("=" * 70)
print(" New Objectives System Demo")
print("=" * 70)
print()

# Create new objectives manager (will generate new defaults)
manager = ObjectivesManager("agent/current_objectives.json")

print(" Default Objectives:")
print()
for obj in manager.get_active_objectives():
    print(f" {obj.name}")
    print(f"   ID: {obj.id}")
    print(f"   Description: {obj.description}")
    print(f"   Type: {obj.type}")
    print(f"   Target: {obj.target}")
    print(f"   Reward Weight: {obj.reward_weight}")
    print(f"   Progress: {obj.progress * 100:.0f}%")
    print()

print("=" * 70)
print(" Testing Dialogue Capture")
print("=" * 70)
print()

# Simulate capturing dialogues
test_dialogues = [
    "There is a movie on TV.",
    "JOHNNY booted up the PC.",
    "MOM: See you, honey!",
    "PROF. BIRCH''S POKéMON LAB"
]

for i, dialogue in enumerate(test_dialogues, 1):
    print(f"{i}. Adding dialogue: ''{dialogue}''")
    manager.add_dialogue(dialogue, location="Starting House")
    
    # Check progress
    for obj in manager.get_active_objectives():
        if obj.progress > 0 or obj.completed:
            status = " COMPLETED" if obj.completed else f" {obj.progress * 100:.0f}%"
            print(f"    {obj.name}: {status}")
    print()

print("=" * 70)
print(" Final Status")
print("=" * 70)
print()
print(f"Active objectives: {len(manager.get_active_objectives())}")
print(f"Completed objectives: {len(manager.get_completed_objectives())}")
print(f"Dialogues captured: {len(manager.dialogue_history)}")
print()

if manager.get_completed_objectives():
    print(" Completed:")
    for obj in manager.get_completed_objectives():
        print(f"    {obj.name}")
    print()

if manager.get_active_objectives():
    print(" Still active:")
    for obj in manager.get_active_objectives():
        print(f"    {obj.name} ({obj.progress * 100:.0f}% complete)")
    print()

print("=" * 70)
