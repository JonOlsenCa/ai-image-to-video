#!/usr/bin/env python3
"""Test video generation directly"""
import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_generation():
    try:
        from services.stable_video_generator import StableVideoGenerator
        from PIL import Image
        import numpy as np
        
        print("✅ Imported StableVideoGenerator")
        
        # Create test image if needed
        test_image_path = Path("test_input.jpg")
        if not test_image_path.exists():
            print("Creating test image...")
            img = Image.new('RGB', (512, 512), color='red')
            img.save(test_image_path)
        
        # Create generator
        gen = StableVideoGenerator()
        print("✅ Generator created")
        
        # Test output path
        output_path = Path("test_output.mp4")
        
        # Progress callback
        async def progress_callback(progress, message):
            print(f"Progress: {progress}% - {message}")
        
        # Generate video
        print("\n🎬 Generating test video...")
        result = await gen.generate_video(
            prompt="person walking",
            output_path=str(output_path),
            duration_seconds=2,
            progress_callback=progress_callback,
            image_path=str(test_image_path)
        )
        
        print(f"\n✅ Video generated: {result}")
        print(f"File size: {output_path.stat().st_size} bytes")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_generation())