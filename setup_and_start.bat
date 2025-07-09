@echo off
title AI Image-to-Video Setup and Start
color 0A

echo.
echo ========================================
echo   AI Image-to-Video Generator
echo   Complete Setup and Start
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

:: Check directory
if not exist "backend\main.py" (
    echo ❌ Error: backend\main.py not found
    echo Please run this script from the ai-image-to-video root directory
    pause
    exit /b 1
)

echo.
echo 🔍 Checking current dependencies...

:: Test if FastAPI is available
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo ❌ FastAPI not installed
    goto :install_deps
) else (
    echo ✅ FastAPI available
)

:: Test if torch is available
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo ❌ PyTorch not installed
    goto :install_deps
) else (
    echo ✅ PyTorch available
)

:: Test if diffusers is available
python -c "import diffusers" >nul 2>&1
if errorlevel 1 (
    echo ❌ Diffusers not installed
    goto :install_deps
) else (
    echo ✅ Diffusers available
)

echo.
echo ✅ All dependencies appear to be installed
goto :start_server

:install_deps
echo.
echo 📥 Installing dependencies...
echo This may take several minutes...
echo.

:: Install core packages
echo Installing FastAPI and server components...
python -m pip install --upgrade pip
python -m pip install fastapi uvicorn python-multipart aiofiles websockets

:: Install AI packages
echo Installing AI/ML packages...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install diffusers transformers accelerate safetensors

:: Install media processing
echo Installing media processing packages...
python -m pip install imageio[ffmpeg] opencv-python pillow numpy

echo.
echo ✅ Dependencies installation complete

:start_server
echo.
echo 🚀 Starting AI Image-to-Video Server...

:: Create directories
if not exist "backend\outputs" mkdir "backend\outputs"
if not exist "backend\uploads" mkdir "backend\uploads"
if not exist "backend\models" mkdir "backend\models"

:: Try to start the full server
cd backend
echo.
echo 🔍 Testing full server startup...
python -c "from main import app; print('✅ Full server can start')" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Full server has issues, starting simple server instead...
    echo.
    echo 📝 Simple server will be available at: http://localhost:8000
    echo 💡 Use this to check status and install remaining dependencies
    echo.
    python simple_server.py
) else (
    echo ✅ Starting full AI server...
    echo.
    echo 📝 Server will be available at: http://localhost:8000
    echo 📚 API documentation at: http://localhost:8000/docs
    echo.
    echo 🎯 Features Available:
    echo    • Image-to-Video Generation
    echo    • Text-to-Video Generation (NSFW model)
    echo    • Real-time Progress Tracking
    echo.
    echo 💡 Tips:
    echo    • First generation will download models (~10GB)
    echo    • Enable NSFW model for text-to-video generation
    echo    • Press Ctrl+C to stop the server
    echo.
    
    :: Open browser after a short delay
    timeout /t 3 /nobreak >nul
    start http://localhost:8000
    
    python main.py
)

cd ..

echo.
echo 🛑 Server stopped
echo.
echo 📊 Session Summary:
echo    • Generated videos saved to: backend/outputs/
echo    • Uploaded images saved to: backend/uploads/
echo    • Models cached in: backend/models/
echo.
pause
