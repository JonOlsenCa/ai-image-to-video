@echo off
title Install AI Video Dependencies
color 0A

echo.
echo ========================================
echo   Installing AI Video Dependencies
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ from:
    echo https://python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python found
python --version

echo.
echo 📥 Installing core dependencies...

:: Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install core web framework
echo Installing FastAPI and server...
pip install fastapi uvicorn python-multipart aiofiles websockets

:: Install AI/ML libraries
echo Installing AI libraries...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate safetensors

:: Install image/video processing
echo Installing media processing...
pip install imageio[ffmpeg] opencv-python pillow numpy

:: Install additional utilities
echo Installing utilities...
pip install requests pathlib

echo.
echo 🔍 Verifying installation...

:: Test imports
python -c "import fastapi; print('✅ FastAPI')" || echo "❌ FastAPI failed"
python -c "import torch; print('✅ PyTorch')" || echo "❌ PyTorch failed"
python -c "import diffusers; print('✅ Diffusers')" || echo "❌ Diffusers failed"
python -c "import transformers; print('✅ Transformers')" || echo "❌ Transformers failed"
python -c "import imageio; print('✅ ImageIO')" || echo "❌ ImageIO failed"

echo.
echo 🔍 Testing CUDA support...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo.
echo ========================================
echo   Installation Complete
echo ========================================
echo.
echo Next steps:
echo 1. Run: test_setup.bat (to verify everything works)
echo 2. Run: start_ai_video_simple.bat (to start the server)
echo.
pause
