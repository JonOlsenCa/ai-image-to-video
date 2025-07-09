#!/usr/bin/env python3
"""
Test if the server can start properly
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import fastapi
        print("✅ FastAPI imported")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import uvicorn
        print("✅ Uvicorn imported")
    except ImportError as e:
        print(f"❌ Uvicorn import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch imported (CUDA: {torch.cuda.is_available()})")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    return True

def test_main_import():
    """Test if main.py can be imported"""
    print("\n🔍 Testing main.py import...")
    
    try:
        # Change to backend directory
        os.chdir('backend')
        
        # Import main
        import main
        print("✅ main.py imported successfully")
        print(f"✅ Video generator: {type(main.video_generator).__name__}")
        return True
    except Exception as e:
        print(f"❌ main.py import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 AI Image-to-Video Server Test")
    print("=" * 50)
    
    # Test basic imports
    if not test_imports():
        print("\n❌ Basic imports failed")
        return False
    
    # Test main import
    if not test_main_import():
        print("\n❌ Main import failed")
        return False
    
    print("\n✅ All tests passed! Server should start successfully.")
    print("\nTo start the server, run:")
    print("  start_server_now.bat")
    print("  OR")
    print("  cd backend && python main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
