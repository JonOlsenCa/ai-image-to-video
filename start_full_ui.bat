@echo off
title AI Image-to-Video Generator - Full UI
color 0A

echo.
echo ========================================
echo   AI Image-to-Video Generator
echo   Starting Full UI
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

:: Check directory
if not exist "backend\main.py" (
    echo ❌ Error: backend\main.py not found
    echo Please run this script from the ai-image-to-video root directory
    pause
    exit /b 1
)

echo ✅ Python found
echo 🔍 Starting server with full UI...

:: Create directories
if not exist "backend\outputs" mkdir "backend\outputs"
if not exist "backend\uploads" mkdir "backend\uploads"
if not exist "backend\models" mkdir "backend\models"

:: Move to backend and start server
cd backend

echo.
echo 🚀 Starting AI Image-to-Video Server...
echo 📝 Server will be available at: http://localhost:8000
echo 🎬 Full UI with upload, prompts, and generation
echo 🛑 Press Ctrl+C to stop the server
echo.

:: Wait a moment then open browser
timeout /t 2 /nobreak >nul
start http://localhost:8000

:: Start the server
python main.py

:: If we get here, the server stopped
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
