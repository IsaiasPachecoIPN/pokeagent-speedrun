#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test Text Capture from Simulator - With Intelligent Exploration"""
import json
import logging
import argparse
import random
from datetime import datetime
from agent.drl_env import PokemonEmeraldEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_text_capture(rom_path: str = "Emerald-GBAdvance/rom.gba",
                     state_path: str = "Emerald-GBAdvance/quick_start_save.state",
                     num_steps: int = 500,
                     output_file: str = "captured_text.json"):
    """Run environment with intelligent exploration to capture text."""
    print("=" * 70)
    print(" Text Capture Test (Intelligent Exploration)")
    print("=" * 70)
    print(f"ROM: {rom_path}")
    print(f"State: {state_path}")
    print(f"Steps: {num_steps}")
    print(f"Output: {output_file}")
    print("=" * 70)
    print()
    
    # Create environment
    print(" Creating environment...")
    env = PokemonEmeraldEnv(
        rom_path=rom_path,
        initial_state_path=state_path,
        render_mode=None,
        max_steps=num_steps + 100
    )
    print(" Environment created")
    print()
    
    # Storage
    captured_texts = []
    text_set = set()
    stats = {
        "total_steps": 0,
        "total_texts_captured": 0,
        "unique_texts": 0,
        "text_by_category": {
            "location": 0,
            "dialogue": 0,
            "npc": 0,
            "story": 0,
            "other": 0
        }
    }
    
    print(" Running simulation with intelligent exploration...")
    print("   Strategy: Press A to interact, move randomly, press B to close menus")
    print()
    
    # Reset
    obs, info = env.reset()
    last_text = ""
    
    # Exploration strategy
    action_sequence = []
    for step in range(num_steps):
        # Smart exploration:
        # - 30% A button (interact with objects/NPCs)
        # - 10% B button (close menus/dialogues)
        # - 60% movement (explore the map)
        rand = random.random()
        if rand < 0.3:
            action = 0  # A button - interact
        elif rand < 0.4:
            action = 1  # B button - back/cancel
        else:
            action = random.choice([2, 3, 4, 5])  # Random movement
        
        action_sequence.append(action)
        
        # Execute
        obs, reward, terminated, truncated, info = env.step(action)
        stats["total_steps"] += 1
        
        # Try to read text
        try:
            if hasattr(env.emulator, 'memory_reader'):
                text = env.emulator.memory_reader.read_dialog()
                
                if text and text.strip() and text != last_text:
                    stats["total_texts_captured"] += 1
                    
                    # Categorize
                    text_lower = text.lower()
                    category = "other"
                    
                    if any(word in text_lower for word in ["town", "route", "city"]):
                        category = "location"
                        stats["text_by_category"]["location"] += 1
                    elif any(word in text_lower for word in ["professor", "prof.", "gym", "leader"]):
                        category = "npc"
                        stats["text_by_category"]["npc"] += 1
                    elif any(word in text_lower for word in ["lab", "center", "mart"]):
                        category = "location"
                        stats["text_by_category"]["location"] += 1
                    elif any(word in text_lower for word in ["welcome", "help", "save", "pokemon", "poké"]):
                        category = "story"
                        stats["text_by_category"]["story"] += 1
                    elif any(word in text_lower for word in ["mom", "dad", "said", "asked"]):
                        category = "dialogue"
                        stats["text_by_category"]["dialogue"] += 1
                    else:
                        stats["text_by_category"]["other"] += 1
                    
                    entry = {
                        "step": step,
                        "timestamp": datetime.now().isoformat(),
                        "text": text,
                        "category": category,
                        "location": info.get('location', 'Unknown'),
                        "text_length": len(text)
                    }
                    
                    captured_texts.append(entry)
                    text_set.add(text)
                    
                    emoji = {
                        "location": "",
                        "npc": "",
                        "dialogue": "",
                        "story": "",
                        "other": ""
                    }.get(category, "")
                    
                    print(f"{emoji} [Step {step:4d}] {category.upper():10s}: {text[:60]}...")
                    last_text = text
        
        except Exception as e:
            logger.debug(f"Error reading text at step {step}: {e}")
        
        if terminated or truncated:
            print(f"\n  Episode ended at step {step}")
            break
        
        if (step + 1) % 100 == 0:
            print(f"   Progress: {step + 1}/{num_steps} steps ({len(text_set)} unique texts)")
    
    stats["unique_texts"] = len(text_set)
    env.close()
    
    print()
    print("=" * 70)
    print(" Test Results")
    print("=" * 70)
    print(f"Total steps executed: {stats['total_steps']}")
    print(f"Total texts captured: {stats['total_texts_captured']}")
    print(f"Unique texts: {stats['unique_texts']}")
    print()
    print("Text by category:")
    for category, count in stats['text_by_category'].items():
        print(f"  {category:12s}: {count:4d}")
    print()
    
    # Save
    output_data = {
        "test_info": {
            "rom_path": rom_path,
            "state_path": state_path,
            "num_steps": num_steps,
            "timestamp": datetime.now().isoformat()
        },
        "statistics": stats,
        "captured_texts": captured_texts
    }
    
    print(f" Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f" Results saved!")
    print()
    
    if captured_texts:
        print(" Sample of captured texts (first 5):")
        print("-" * 70)
        for i, entry in enumerate(captured_texts[:5], 1):
            print(f"{i}. [{entry['category'].upper()}] {entry['text'][:60]}...")
        print("-" * 70)
    
    print()
    print("=" * 70)
    print(" Test Complete!")
    print("=" * 70)
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test text capture with intelligent exploration")
    parser.add_argument("--rom", type=str, default="Emerald-GBAdvance/rom.gba")
    parser.add_argument("--state", type=str, default="Emerald-GBAdvance/quick_start_save.state")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=str, default="captured_text.json")
    
    args = parser.parse_args()
    test_text_capture(args.rom, args.state, args.steps, args.output)
