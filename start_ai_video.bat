@echo off
title AI Image-to-Video Generator with NSFW Support
color 0A

echo.
echo ========================================
echo   AI Image-to-Video Generator
echo   with NSFW-gen-v2 Support
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
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

:: Move to backend directory
cd backend

echo 🔍 Checking dependencies...

:: Check if virtual environment exists
if not exist "backend\venv" (
    echo 📦 Creating virtual environment...
    python -m venv backend\venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo 🔧 Activating virtual environment...
call backend\venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    echo Trying alternative activation method...
    cd backend
    call venv\Scripts\activate.bat
    if errorlevel 1 (
        echo ❌ Virtual environment activation failed
        echo Continuing without virtual environment...
        goto :skip_venv
    )
    cd ..
)

:skip_venv

:: Install/upgrade dependencies
echo 📥 Installing dependencies...
pip install --upgrade pip
cd backend
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies from requirements.txt
    echo Trying to install core dependencies individually...
    pip install fastapi uvicorn python-multipart diffusers transformers torch imageio accelerate aiofiles websockets opencv-python
)
cd ..

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
echo    • Multiple AI Model Support
echo.
echo 🎯 API Endpoints:
echo    • GET  /                   - Web interface
echo    • POST /generate-video     - Generate video from image/text
echo    • POST /generate-image     - Generate image only (NSFW)
echo    • POST /configure-nsfw     - Enable/disable NSFW model
echo    • GET  /model-info         - Get model information
echo    • GET  /docs               - API documentation
echo.
echo 💡 Tips:
echo    • First generation will download models (~10GB)
echo    • Enable NSFW model for text-to-video generation
echo    • Check /docs for complete API reference
echo    • Press Ctrl+C to stop the server
echo.

:: Start the server
echo ⚡ Launching server...
cd backend
python main.py

:: If we get here, the server stopped
echo.
echo 🛑 Server stopped
echo.
echo 📊 Session Summary:
echo    • Generated videos saved to: backend/outputs/
echo    • Uploaded images saved to: backend/uploads/
echo    • Models cached in: backend/models/
echo.
pause
