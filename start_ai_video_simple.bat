@echo off
title AI Image-to-Video Generator - Simple Start
color 0A

echo.
echo ========================================
echo   AI Image-to-Video Generator
echo   Simple Start (No Virtual Environment)
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    echo Download from: https://python.org/downloads/
    pause
    exit /b 1
)

:: Check if we're in the right directory
if not exist "backend\main.py" (
    echo ❌ Error: backend\main.py not found
    echo Please run this script from the ai-image-to-video root directory
    pause
    exit /b 1
)

:: Create necessary directories
if not exist "backend\outputs" mkdir "backend\outputs"
if not exist "backend\uploads" mkdir "backend\uploads"
if not exist "backend\models" mkdir "backend\models"

echo 🔍 Checking Python and dependencies...

:: Move to backend directory
cd backend

:: Install/upgrade dependencies directly to system Python
echo 📥 Installing dependencies...
pip install --upgrade pip --quiet
pip install fastapi uvicorn python-multipart --quiet
pip install diffusers transformers torch --quiet
pip install imageio[ffmpeg] accelerate aiofiles websockets opencv-python --quiet

if errorlevel 1 (
    echo ⚠️  Some dependencies may have failed to install
    echo Continuing anyway...
)

:: Check if CUDA is available
echo 🔍 Checking GPU support...
python -c "import torch; print('✅ CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')" 2>nul

echo.
echo 🚀 Starting AI Image-to-Video Server...
echo.
echo 📝 Server will be available at:
echo    http://localhost:8000
echo.
echo 🎯 Features Available:
echo    • Image-to-Video Generation
echo    • Text-to-Video Generation (NSFW model)
echo    • Image-Only Generation (NSFW model)
echo    • Real-time Progress Tracking
echo.
echo 💡 Tips:
echo    • First generation will download models (~10GB)
echo    • Enable NSFW model for text-to-video generation
echo    • Press Ctrl+C to stop the server
echo.

:: Start the server
echo ⚡ Launching server...
echo.
echo 🌐 Opening browser in 3 seconds...
timeout /t 3 /nobreak >nul
start http://localhost:8000
python main.py

:: If we get here, the server stopped
echo.
echo 🛑 Server stopped
echo.
echo 📊 Session Summary:
echo    • Generated videos saved to: outputs/
echo    • Uploaded images saved to: uploads/
echo    • Models cached in: models/
echo.
pause
