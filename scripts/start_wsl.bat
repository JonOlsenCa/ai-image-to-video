@echo off
title AI Video Generator - WSL
echo ================================================
echo    AI Image to Video Generator 
echo    With Advanced Person Animation
echo ================================================
echo.

REM Clean up
echo Cleaning up old processes...
wsl -e pkill -f uvicorn 2>nul
wsl -e pkill -f "npm start" 2>nul
taskkill /f /im node.exe 2>nul
timeout /t 2 > nul

REM Backend in WSL
echo.
echo Starting Backend Server (WSL)...
start "AI Video Backend" wsl -e bash -c "cd /mnt/c/github/ai-image-to-video/backend && source venv/bin/activate && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo Waiting for backend...
timeout /t 8 > nul

REM Frontend in WSL
echo.
echo Starting Frontend (WSL)...
start "AI Video Frontend" wsl -e bash -c "cd /mnt/c/github/ai-image-to-video/frontend && npm start"

echo.
echo ================================================
echo Services starting...
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000 
echo ================================================
echo.

timeout /t 10 > nul
start http://localhost:3000

echo.
echo App is running! Person animations:
echo - Walking, Dancing, Waving, Jumping
echo - Smiling, Talking, Nodding
echo.
echo If stuck at "initializing":
echo 1. Check the Backend window for errors
echo 2. Try refreshing the webpage
echo 3. Make sure WSL is running
echo.
echo Press any key to stop all services...
pause > nul

wsl -e pkill -f uvicorn
wsl -e pkill -f "npm start"