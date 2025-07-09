#!/usr/bin/env python3
"""Test imageio and video generation"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imageio():
    try:
        import imageio
        print(f"✅ imageio version: {imageio.__version__}")
        
        # Test if ffmpeg is available
        try:
            import imageio_ffmpeg
            print(f"✅ imageio-ffmpeg version: {imageio_ffmpeg.__version__}")
        except:
            print("❌ imageio-ffmpeg not installed")
        
        # Test creating a simple video
        import numpy as np
        frames = []
        for i in range(10):
            # Create a frame with changing color
            frame = np.ones((100, 100, 3), dtype=np.uint8) * (i * 25)
            frames.append(frame)
        
        # Try to save video
        output_path = "test_imageio_output.mp4"
        try:
            imageio.mimwrite(output_path, frames, fps=5, codec='libx264')
            print(f"✅ Video saved successfully to {output_path}")
            import os
            print(f"   File size: {os.path.getsize(output_path)} bytes")
        except Exception as e:
            print(f"❌ Failed to save video: {e}")
            
            # Try with different codec
            try:
                imageio.mimwrite(output_path, frames, fps=5)
                print(f"✅ Video saved with default codec")
            except Exception as e2:
                print(f"❌ Failed with default codec too: {e2}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_imageio()