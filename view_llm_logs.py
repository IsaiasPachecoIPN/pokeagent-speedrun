#!/usr/bin/env python3
"""View LLM Response Logs"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime

def view_latest_log(log_dir="logs/llm_responses"):
    """View the most recent LLM log."""
    
    # Find all log files
    log_files = sorted(Path(log_dir).glob("llm_call_*.json"))
    
    if not log_files:
        print(f"No logs found in {log_dir}")
        return
    
    latest_log = log_files[-1]
    
    print("=" * 70)
    print(f" Latest LLM Log: {latest_log.name}")
    print("=" * 70)
    print()
    
    with open(latest_log, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
    
    print(f"📋 Call Number: {log_data['call_number']}")
    print(f"⏰ Timestamp: {log_data['timestamp']}")
    print(f"🎮 Training Step: {log_data['training_step']}")
    print(f"🤖 Model: {log_data['model']}")
    print()
    
    # 🆕 Show reward statistics
    if log_data.get('reward_statistics'):
        print("📊 Reward Statistics:")
        print("-" * 70)
        reward_stats = log_data['reward_statistics']
        print(f"   Total Episodes: {reward_stats.get('total_episodes', 0)}")
        print(f"   Avg Reward (Last 10): {reward_stats.get('avg_reward_last_10', 0):.3f}")
        print(f"   Avg Reward (Last 50): {reward_stats.get('avg_reward_last_50', 0):.3f}")
        print(f"   Avg Reward (All Time): {reward_stats.get('avg_reward_all_time', 0):.3f}")
        print(f"   Trend: {reward_stats.get('reward_trend', 'N/A')}")
        recent = reward_stats.get('recent_rewards', [])
        if recent:
            print(f"   Recent (last 10): {', '.join([f'{r:.2f}' for r in recent[-5:]])}...")
        print()
    
    print("🎲 Game State:")
    print("-" * 70)
    print(log_data['game_state_summary'])
    print()
    
    # 🆕 Show map observation
    if log_data.get('map_observation'):
        map_obs = log_data['map_observation']
        if 'error' not in map_obs:
            print("🗺️  Map Observation:")
            print("-" * 70)
            
            # Show ASCII visualization
            if map_obs.get('ascii_visualization'):
                print(map_obs['ascii_visualization'])
            
            # Show spatial analysis summary
            spatial = map_obs.get('spatial_analysis', {})
            if spatial:
                print("\n📍 Spatial Summary:")
                if spatial.get('nearby_doors'):
                    print(f"   🚪 Doors: {spatial['nearby_doors']}")
                if spatial.get('nearby_npcs'):
                    print(f"   👤 NPCs: {spatial['nearby_npcs']}")
                if spatial.get('passable_directions'):
                    print(f"   ✅ Can move: {', '.join(spatial['passable_directions'])}")
            print()
    
    if log_data.get('tool_calls_summary'):
        print("🛠️  Tools Used:")
        print("-" * 70)
        for i, tool in enumerate(log_data['tool_calls_summary'], 1):
            print(f"{i}. {tool['tool']}")
            if tool.get('arguments'):
                args_str = json.dumps(tool['arguments'], indent=2)
                print(f"   Arguments: {args_str}")
        print()
    
    if log_data.get('final_response'):
        print("💬 LLM Final Response:")
        print("-" * 70)
        print(log_data['final_response'])
        print()
    
    print("🎯 Objectives After:")
    print("-" * 70)
    objectives = log_data['objectives_after']
    if isinstance(objectives, str):
        objectives = json.loads(objectives)
    
    if objectives.get('objectives'):
        for obj_id, obj in objectives['objectives'].items():
            status = "✅" if obj['completed'] else "📌"
            print(f"{status} {obj['name']}")
            print(f"   Progress: {obj['progress'] * 100:.0f}%")
            print(f"   Weight: {obj['reward_weight']}")
            print()
    
    print("=" * 70)


def list_all_logs(log_dir="logs/llm_responses"):
    """List all LLM logs."""
    
    log_files = sorted(Path(log_dir).glob("llm_call_*.json"))
    
    if not log_files:
        print(f"No logs found in {log_dir}")
        return
    
    print("=" * 70)
    print(f" All LLM Logs ({len(log_files)} total)")
    print("=" * 70)
    print()
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            tools_used = len(log_data.get('tool_calls_summary', []))
            timestamp = datetime.fromisoformat(log_data['timestamp'])
            
            # 🆕 Get reward info
            reward_info = ""
            if log_data.get('reward_statistics'):
                reward_stats = log_data['reward_statistics']
                avg_10 = reward_stats.get('avg_reward_last_10', 0)
                trend = reward_stats.get('reward_trend', 'N/A')
                reward_info = f" | Reward: {avg_10:.2f} {trend}"
            
            print(f"📋 {log_file.name}")
            print(f"   Step: {log_data['training_step']:,} | Tools: {tools_used} | Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}{reward_info}")
            
            if log_data.get('final_response'):
                response_preview = log_data['final_response'][:80]
                print(f"   Response: {response_preview}...")
            print()
        
        except Exception as e:
            print(f" Error reading {log_file.name}: {e}")
            print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_all_logs()
    else:
        view_latest_log()
