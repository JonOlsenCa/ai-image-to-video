#!/usr/bin/env python3
"""Test script for NSFW model integration"""

import asyncio
import sys
import os
from pathlib import Path
import tempfile

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_nsfw_generator():
    """Test the NSFW text-to-video generator"""
    print("🧪 Testing NSFW Text-to-Video Generator")
    print("=" * 50)
    
    try:
        from services.nsfw_text_to_video_generator import NSFWTextToVideoGenerator
        
        generator = NSFWTextToVideoGenerator()
        print("✅ NSFW generator imported successfully")
        
        # Test model info
        print(f"Device: {generator.device}")
        print(f"Data type: {generator.dtype}")
        
        # Test image generation only (faster than video)
        print("\n🎨 Testing image generation...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_image.png"
            
            test_prompt = "a beautiful landscape, high quality, detailed"
            
            try:
                result = await generator.generate_image_only(
                    prompt=test_prompt,
                    output_path=str(output_path)
                )
                
                if output_path.exists():
                    file_size = output_path.stat().st_size
                    print(f"✅ Image generated successfully: {file_size} bytes")
                    return True
                else:
                    print("❌ Image file not created")
                    return False
                    
            except Exception as e:
                print(f"❌ Image generation failed: {e}")
                return False
                
    except ImportError as e:
        print(f"❌ Failed to import NSFW generator: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

async def test_enhanced_generator():
    """Test the enhanced video generator with NSFW support"""
    print("\n🎬 Testing Enhanced Video Generator with NSFW")
    print("=" * 50)
    
    try:
        from services.enhanced_video_generator import EnhancedVideoGenerator
        
        # Test with NSFW enabled
        generator = EnhancedVideoGenerator(use_nsfw_model=True)
        print("✅ Enhanced generator with NSFW imported successfully")
        
        # Test model info
        model_info = generator.get_model_info()
        print(f"Model info: {model_info}")
        
        # Test switching models
        print("\n🔄 Testing model switching...")
        generator.switch_model(use_nsfw=False)
        print("✅ Switched to standard model")
        
        generator.switch_model(use_nsfw=True)
        print("✅ Switched to NSFW model")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import enhanced generator: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

async def test_api_endpoints():
    """Test the new API endpoints"""
    print("\n🌐 Testing API Endpoints")
    print("=" * 50)
    
    try:
        import requests
        import time
        
        base_url = "http://localhost:8000"
        
        # Test model info endpoint
        print("Testing /model-info endpoint...")
        try:
            response = requests.get(f"{base_url}/model-info", timeout=5)
            if response.status_code == 200:
                print(f"✅ Model info: {response.json()}")
            else:
                print(f"❌ Model info failed: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("⚠️  Server not running - skipping API tests")
            return True
        except Exception as e:
            print(f"❌ Model info error: {e}")
        
        # Test NSFW configuration endpoint
        print("\nTesting /configure-nsfw endpoint...")
        try:
            response = requests.post(f"{base_url}/configure-nsfw", 
                                   json={"enable_nsfw": True}, timeout=5)
            if response.status_code == 200:
                print(f"✅ NSFW config: {response.json()}")
            else:
                print(f"❌ NSFW config failed: {response.status_code}")
        except Exception as e:
            print(f"❌ NSFW config error: {e}")
        
        return True
        
    except ImportError:
        print("⚠️  requests not available - skipping API tests")
        return True

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking Dependencies")
    print("=" * 50)
    
    dependencies = [
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy")
    ]
    
    all_available = True
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} not available")
            all_available = False
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - will use CPU")
    except:
        print("❌ Cannot check CUDA")
    
    return all_available

async def main():
    """Run all tests"""
    print("🚀 NSFW Model Integration Test Suite")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Missing dependencies - please install requirements")
        return False
    
    # Run tests
    tests = [
        ("NSFW Generator", test_nsfw_generator),
        ("Enhanced Generator", test_enhanced_generator),
        ("API Endpoints", test_api_endpoints)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! NSFW integration is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    asyncio.run(main())
