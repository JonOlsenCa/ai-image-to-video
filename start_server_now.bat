@echo off
title AI Image-to-Video Server - Starting Now
color 0A

echo.
echo ========================================
echo   Starting AI Image-to-Video Server
echo ========================================
echo.

cd backend

echo 🚀 Starting server with full UI...
echo 📝 Server will be available at: http://localhost:8000
echo 🎬 Full UI with upload, prompts, and generation
echo.

:: Open browser after a short delay
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8000"

:: Start the server
python main.py

echo.
echo 🛑 Server stopped
pause
