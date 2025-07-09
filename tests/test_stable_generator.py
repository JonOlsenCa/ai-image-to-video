#!/usr/bin/env python3
"""Test script for stable video generator"""
import sys
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from services.stable_video_generator import StableVideoGenerator
    print("✅ Successfully imported StableVideoGenerator")
    
    # Create instance
    gen = StableVideoGenerator()
    print("✅ Generator instance created")
    
    # Get available animations
    animations = gen.get_available_animations()
    print("\n📋 Available animations:")
    for category, anims in animations.items():
        print(f"\n{category}:")
        for anim in anims:
            print(f"  - {anim}")
    
    # Test analyze_prompt
    test_prompts = ["person walking", "bounce", "dancing"]
    print("\n🔍 Testing prompt analysis:")
    for prompt in test_prompts:
        result = gen.analyze_prompt(prompt)
        print(f"  '{prompt}' -> type: {result['type']}, intensity: {result['intensity']}, speed: {result['speed']}")
    
    print("\n✅ All basic tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()