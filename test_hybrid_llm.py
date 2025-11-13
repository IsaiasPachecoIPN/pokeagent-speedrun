"""
Test script for Hybrid DRL+LLM system with tool calling

This script tests the LLM's ability to:
1. Read objectives
2. Read dialogues
3. Create new objectives based on game context
4. Adjust policy parameters
"""

import logging
import sys

# Add agent module to path
sys.path.insert(0, '.')

from agent.hybrid_llm_callback import test_llm_tools

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("="*60)
    print("🧪 Testing Hybrid DRL+LLM System")
    print("="*60)
    print()
    print("This test will:")
    print("1. Create a test objectives manager")
    print("2. Add sample dialogues")
    print("3. Call LLM with tool calling enabled")
    print("4. LLM will read objectives and dialogues")
    print("5. LLM will create new objectives based on context")
    print()
    print("Prerequisites:")
    print("- Ollama must be running")
    print("- Model 'qwen3:8b' must be installed")
    print()
    input("Press Enter to start test...")
    print()
    
    # Run the test
    test_llm_tools()
