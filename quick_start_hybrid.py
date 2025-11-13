"""
Quick Start: Hybrid DRL+LLM Training

This script demonstrates the complete workflow:
1. Initialize objectives with a simple goal
2. Add sample dialogues
3. Start training with hybrid mode
4. LLM evaluates and creates new objectives
"""

import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_initial_objectives():
    """Create initial objectives file with starter goal."""
    
    objectives_file = Path("agent/current_objectives.json")
    
    # Check if already exists
    if objectives_file.exists():
        logger.info(f"✅ Objectives file already exists: {objectives_file}")
        response = input("Overwrite with fresh objectives? (y/n): ")
        if response.lower() != 'y':
            logger.info("Keeping existing objectives")
            return
    
    # Create starter objectives
    objectives_data = {
        "objectives": {
            "exit_starting_room": {
                "id": "exit_starting_room",
                "name": "Exit Starting Room",
                "description": "Leave the starting bedroom and explore the house",
                "type": "location",
                "target": {
                    "map": "LITTLEROOT_TOWN",
                    "condition": "outside_starting_room"
                },
                "reward_weight": 2.0,
                "completed": False,
                "progress": 0.0,
                "created_at": "2025-11-13T00:00:00"
            }
        },
        "dialogue_history": [
            {
                "text": "Welcome to the world of Pokemon!",
                "npc": "System",
                "location": "Unknown",
                "timestamp": "2025-11-13T00:00:00"
            }
        ],
        "last_updated": "2025-11-13T00:00:00"
    }
    
    # Ensure directory exists
    objectives_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to file
    with open(objectives_file, 'w') as f:
        json.dump(objectives_data, f, indent=2)
    
    logger.info(f"✅ Created initial objectives: {objectives_file}")
    logger.info(f"   - 1 starter objective: Exit Starting Room")
    logger.info(f"   - Reward weight: 2.0 (high priority)")


def check_prerequisites():
    """Check if all prerequisites are met."""
    
    logger.info("🔍 Checking prerequisites...")
    
    # Check ROM file
    rom_path = Path("Emerald-GBAdvance/rom.gba")
    if not rom_path.exists():
        logger.error(f"❌ ROM file not found: {rom_path}")
        logger.error("   Please ensure Pokemon Emerald ROM is in Emerald-GBAdvance/")
        return False
    logger.info(f"✅ ROM file found: {rom_path}")
    
    # Check state file
    state_path = Path("Emerald-GBAdvance/quick_start_save.state")
    if not state_path.exists():
        logger.warning(f"⚠️  State file not found: {state_path}")
        logger.warning("   Training will start from ROM boot (slower)")
    else:
        logger.info(f"✅ State file found: {state_path}")
    
    # Check Ollama
    import subprocess
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("✅ Ollama is running")
            if 'qwen3:8b' in result.stdout or 'qwen2.5' in result.stdout:
                logger.info("✅ Model qwen3:8b is available")
            else:
                logger.warning("⚠️  Model qwen3:8b not found")
                logger.warning("   Run: ollama pull qwen3:8b")
                return False
        else:
            logger.error("❌ Ollama not responding")
            return False
    except FileNotFoundError:
        logger.error("❌ Ollama not installed")
        logger.error("   Install from: https://ollama.ai")
        return False
    except subprocess.TimeoutExpired:
        logger.error("❌ Ollama timeout (is it running?)")
        return False
    
    return True


def print_training_info():
    """Print information about what will happen."""
    
    print("\n" + "="*60)
    print("🚀 HYBRID DRL+LLM TRAINING")
    print("="*60)
    print()
    print("What will happen:")
    print()
    print("1. 🎮 DRL Agent starts learning basic movements")
    print("   - Objective: Exit starting room")
    print("   - Learning: Which buttons to press")
    print("   - Duration: ~5000 steps (few minutes)")
    print()
    print("2. 🤖 LLM Evaluates progress")
    print("   - Reads current objectives")
    print("   - Reads captured dialogues")
    print("   - Creates new objectives based on game context")
    print("   - Adjusts policy parameters")
    print()
    print("3. 🔄 Cycle repeats every 5000 steps")
    print("   - DRL learns low-level actions")
    print("   - LLM guides high-level strategy")
    print()
    print("Files that will be created:")
    print("  - agent/current_objectives.json (objectives and dialogues)")
    print("  - logs/checkpoints/*.zip (model checkpoints)")
    print("  - tensorboard_logs/ (training metrics)")
    print()
    print("="*60)
    print()


def main():
    """Main function to set up and explain the training process."""
    
    print("="*60)
    print("🎯 Quick Start: Hybrid DRL+LLM System")
    print("="*60)
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return
    
    print()
    
    # Setup objectives
    setup_initial_objectives()
    
    print()
    
    # Print training info
    print_training_info()
    
    # Ask to start
    print("Ready to start training?")
    print()
    print("Command to run:")
    print("  python train_ppo.py --hybrid --visualize --timesteps 50000")
    print()
    print("Or without visualization (faster):")
    print("  python train_ppo.py --hybrid --timesteps 50000")
    print()
    
    response = input("Start training now? (y/n): ")
    
    if response.lower() == 'y':
        import subprocess
        print("\n🚀 Starting training...\n")
        subprocess.run([
            'python', 'train_ppo.py',
            '--hybrid',
            '--visualize',
            '--timesteps', '50000',
            '--n-envs', '1'
        ])
    else:
        print("\n✅ Setup complete! Run the command above when ready.")
        print("\nTo monitor objectives in real-time:")
        print("  # Windows PowerShell:")
        print("  while ($true) { Clear-Host; Get-Content agent/current_objectives.json | ConvertFrom-Json | ConvertTo-Json -Depth 10; Start-Sleep 5 }")
        print("\n  # Linux/Mac:")
        print("  watch -n 5 'cat agent/current_objectives.json | jq'")


if __name__ == "__main__":
    main()
