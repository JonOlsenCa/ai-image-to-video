@echo off
title AI Image-to-Video Generator - Full Stack
color 0A

echo.
echo ========================================
echo   AI Image-to-Video Generator
echo   Full Stack (Backend + Frontend)
echo ========================================
echo.

:: Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed
    echo Please install Node.js 14+ and try again
    echo Download from: https://nodejs.org/
    pause
    exit /b 1
)

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

if not exist "frontend\package.json" (
    echo ❌ Error: frontend\package.json not found
    echo Please run this script from the ai-image-to-video root directory
    pause
    exit /b 1
)

echo 🔧 Setting up backend...
cd backend

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating Python virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install Python dependencies
echo 📥 Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt

cd ..

echo 🌐 Setting up frontend...
cd frontend

:: Install Node.js dependencies
echo 📥 Installing Node.js dependencies...
npm install

cd ..

echo.
echo 🚀 Starting Full Stack Application...
echo.
echo 📝 Services will be available at:
echo    Backend API: http://localhost:8000
echo    Frontend:    http://localhost:3000
echo.
echo 💡 Both services will start automatically
echo    Press Ctrl+C in either window to stop
echo.

:: Start backend in a new window
echo 🔧 Starting backend server...
start "AI Video Backend" cmd /k "cd backend && call venv\Scripts\activate.bat && python main.py"

:: Wait a moment for backend to start
timeout /t 3 /nobreak >nul

:: Start frontend in a new window
echo 🌐 Starting frontend server...
start "AI Video Frontend" cmd /k "cd frontend && npm start"

echo.
echo ✅ Both services are starting...
echo 📱 Frontend will open automatically in your browser
echo 🛑 Close both terminal windows to stop the application
echo.

pause
