"""
Test Dialogue Capture System

This script tests if dialogues are being captured correctly from the game.
Run this to diagnose why dialogue_history might be empty.
"""

import logging
import time
from agent.drl_env import PokemonEmeraldEnv
from agent.objectives_manager import ObjectivesManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_dialogue_capture():
    """Test dialogue capture in the environment."""
    
    print("="*60)
    print("🧪 Testing Dialogue Capture System")
    print("="*60)
    print()
    
    # Step 1: Create objectives manager
    print("📋 Step 1: Creating objectives manager...")
    objectives_manager = ObjectivesManager("agent/test_dialogue_capture.json")
    print(f"✅ Objectives manager created")
    print(f"   Current dialogues: {len(objectives_manager.dialogue_history)}")
    print()
    
    # Step 2: Create environment
    print("🎮 Step 2: Creating environment...")
    env = PokemonEmeraldEnv(
        rom_path="Emerald-GBAdvance/rom.gba",
        initial_state_path="Emerald-GBAdvance/quick_start_save.state",
        render_mode=None,  # No visualization for test
        max_steps=1000
    )
    print("✅ Environment created")
    print()
    
    # Step 3: Enable hybrid mode
    print("🤖 Step 3: Enabling hybrid mode...")
    env.enable_hybrid_mode(objectives_manager)
    print(f"✅ Hybrid mode enabled")
    print(f"   enable_dialogue_capture: {env.enable_dialogue_capture}")
    print(f"   objectives_manager: {env.objectives_manager is not None}")
    print()
    
    # Step 4: Reset environment
    print("🔄 Step 4: Resetting environment...")
    obs, info = env.reset()
    print("✅ Environment reset")
    print(f"   Initial state loaded")
    print()
    
    # Step 5: Run some steps and check for dialogues
    print("🎯 Step 5: Running game steps and checking for dialogues...")
    print("   (This will take ~10 seconds)")
    print()
    
    dialogues_captured = 0
    
    for step in range(100):  # Run 100 steps
        # Random action (just to move around)
        action = step % 6  # Cycle through actions
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if dialogue was captured
        current_dialog = env._get_current_dialog()
        if current_dialog:
            print(f"   Step {step}: 📝 Dialogue detected: '{current_dialog[:50]}...'")
            dialogues_captured += 1
        
        # Check objectives manager
        if len(objectives_manager.dialogue_history) > 0:
            latest = objectives_manager.dialogue_history[-1]
            print(f"   ✅ Dialogue saved to manager: '{latest['text'][:50]}...'")
        
        if terminated or truncated:
            print(f"   Episode ended at step {step}")
            break
        
        # Small delay for readability
        if current_dialog:
            time.sleep(0.1)
    
    print()
    print("="*60)
    print("📊 Test Results")
    print("="*60)
    print(f"Total steps: 100")
    print(f"Dialogues detected in cache: {dialogues_captured}")
    print(f"Dialogues saved to manager: {len(objectives_manager.dialogue_history)}")
    print()
    
    if len(objectives_manager.dialogue_history) > 0:
        print("✅ SUCCESS - Dialogues are being captured!")
        print()
        print("Recent dialogues:")
        for i, dialog in enumerate(objectives_manager.dialogue_history[-5:], 1):
            print(f"  {i}. [{dialog.get('location', 'Unknown')}] {dialog['text'][:60]}...")
    else:
        print("⚠️  NO DIALOGUES CAPTURED")
        print()
        print("Possible reasons:")
        print("  1. Game state doesn't have active dialogues yet")
        print("  2. Need to interact with NPCs or objects")
        print("  3. Dialogue reading from memory isn't working")
        print()
        print("Try:")
        print("  - Use a save state that has dialogues on screen")
        print("  - Run training for longer (agent will eventually find NPCs)")
        print("  - Check if memory reading is working")
    
    print()
    
    # Cleanup
    env.close()
    print("🧹 Environment closed")
    print()


def check_current_objectives_file():
    """Check what's in the current objectives file."""
    
    print("="*60)
    print("📂 Checking Current Objectives File")
    print("="*60)
    print()
    
    try:
        manager = ObjectivesManager("agent/current_objectives.json")
        
        print(f"Active objectives: {len(manager.get_active_objectives())}")
        for obj in manager.get_active_objectives():
            print(f"  - {obj.name} (weight: {obj.reward_weight})")
        
        print()
        print(f"Completed objectives: {len(manager.get_completed_objectives())}")
        for obj in manager.get_completed_objectives():
            print(f"  - {obj.name}")
        
        print()
        print(f"Dialogue history: {len(manager.dialogue_history)} entries")
        if len(manager.dialogue_history) > 0:
            print("Recent dialogues:")
            for dialog in manager.dialogue_history[-3:]:
                print(f"  - [{dialog.get('location', '?')}] {dialog['text'][:50]}...")
        else:
            print("  (empty - no dialogues captured yet)")
        
        print()
        print("💡 TIP: Dialogue history will populate during training")
        print("   when the agent encounters NPCs and dialogues in the game.")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
    
    print()


if __name__ == "__main__":
    print()
    print("🔍 Dialogue Capture Diagnostic Tool")
    print()
    print("This tool will:")
    print("1. Check your current objectives file")
    print("2. Test if dialogue capture is working")
    print()
    
    input("Press Enter to start...")
    print()
    
    # First, check what's in the file
    check_current_objectives_file()
    
    # Then test dialogue capture
    response = input("Run dialogue capture test? (y/n): ")
    if response.lower() == 'y':
        print()
        test_dialogue_capture()
    else:
        print("\nTest skipped. Run this script again to test.")
