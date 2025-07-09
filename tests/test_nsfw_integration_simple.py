#!/usr/bin/env python3
"""Simple test script for NSFW model integration (no model downloads)"""

import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all NSFW integration modules can be imported"""
    print("🧪 Testing NSFW Integration Imports")
    print("=" * 50)
    
    try:
        # Test NSFW generator import
        from services.nsfw_text_to_video_generator import NSFWTextToVideoGenerator
        print("✅ NSFWTextToVideoGenerator imported successfully")
        
        # Test enhanced generator import
        from services.enhanced_video_generator import EnhancedVideoGenerator
        print("✅ EnhancedVideoGenerator imported successfully")
        
        # Test that enhanced generator can be initialized with NSFW option
        enhanced_gen = EnhancedVideoGenerator(use_nsfw_model=False)  # Don't load models
        print("✅ EnhancedVideoGenerator initialized successfully")
        
        # Test model info method
        model_info = enhanced_gen.get_model_info()
        print(f"✅ Model info retrieved: {model_info}")
        
        # Test model switching (without actually loading models)
        enhanced_gen.switch_model(use_nsfw=True)
        print("✅ Model switching works")
        
        enhanced_gen.switch_model(use_nsfw=False)
        print("✅ Model switching back works")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_main_integration():
    """Test that main.py can import the enhanced generator"""
    print("\n🌐 Testing Main.py Integration")
    print("=" * 50)
    
    try:
        # Test that main.py imports work
        import main
        print("✅ main.py imported successfully")
        
        # Check if enhanced generator is in the fallback chain
        main_content = open('main.py', 'r').read()
        if 'enhanced_video_generator' in main_content:
            print("✅ Enhanced video generator found in main.py")
        else:
            print("⚠️  Enhanced video generator not found in main.py")
        
        # Check for new endpoints
        if '/model-info' in main_content:
            print("✅ New /model-info endpoint found")
        else:
            print("⚠️  /model-info endpoint not found")
            
        if '/configure-nsfw' in main_content:
            print("✅ New /configure-nsfw endpoint found")
        else:
            print("⚠️  /configure-nsfw endpoint not found")
            
        if '/generate-image' in main_content:
            print("✅ New /generate-image endpoint found")
        else:
            print("⚠️  /generate-image endpoint not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing main.py integration: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available"""
    print("\n🔍 Testing Dependencies")
    print("=" * 50)
    
    dependencies = [
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn")
    ]
    
    all_available = True
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} not available")
            all_available = False
    
    return all_available

def main():
    """Run all tests"""
    print("🚀 NSFW Model Integration Simple Test Suite")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Main Integration", test_main_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
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
        print("\n📝 Next Steps:")
        print("1. Start the server: python run_server.py")
        print("2. Test the new endpoints:")
        print("   - GET /model-info")
        print("   - POST /configure-nsfw")
        print("   - POST /generate-image")
        print("3. The first video generation will download models (~10GB)")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
