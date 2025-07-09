@echo off
title Test AI Video Setup
color 0A

echo.
echo ========================================
echo   Testing AI Video Setup
echo ========================================
echo.

:: Check Python
echo 🔍 Testing Python...
python --version
if errorlevel 1 (
    echo ❌ Python not found
    pause
    exit /b 1
)

:: Check directory structure
echo.
echo 🔍 Testing directory structure...
if exist "backend\main.py" (
    echo ✅ backend\main.py found
) else (
    echo ❌ backend\main.py not found
)

if exist "backend\services" (
    echo ✅ backend\services found
) else (
    echo ❌ backend\services not found
)

:: Test Python imports
echo.
echo 🔍 Testing Python imports...
cd backend

echo Testing FastAPI...
python -c "import fastapi; print('✅ FastAPI available')" 2>nul || echo "❌ FastAPI not available"

echo Testing torch...
python -c "import torch; print('✅ PyTorch available')" 2>nul || echo "❌ PyTorch not available"

echo Testing diffusers...
python -c "import diffusers; print('✅ Diffusers available')" 2>nul || echo "❌ Diffusers not available"

echo Testing services...
python -c "import sys; sys.path.append('.'); from services.enhanced_video_generator import EnhancedVideoGenerator; print('✅ Enhanced generator available')" 2>nul || echo "❌ Enhanced generator not available"

echo Testing NSFW generator...
python -c "import sys; sys.path.append('.'); from services.nsfw_text_to_video_generator import NSFWTextToVideoGenerator; print('✅ NSFW generator available')" 2>nul || echo "❌ NSFW generator not available"

:: Test CUDA
echo.
echo 🔍 Testing CUDA...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Current device:', torch.cuda.current_device() if torch.cuda.is_available() else 'CPU')" 2>nul

echo.
echo 🔍 Testing server startup (quick test)...
timeout /t 1 /nobreak >nul
python -c "from main import app; print('✅ Server can be imported')" 2>nul || echo "❌ Server import failed"

cd ..

echo.
echo ========================================
echo   Test Complete
echo ========================================
echo.
echo If all tests show ✅, you can run:
echo   start_ai_video_simple.bat
echo.
pause
