@echo off
echo ================================================
echo    Git Commit and Push for AI Video Generator
echo ================================================
echo.

REM Configure git identity
echo Setting git identity...
git config --global user.email "jon@olsenconsulting.ca"
git config --global user.name "JonOlsenCa"
echo Git identity configured!
echo.

REM Add all changes
echo Staging all changes...
git add .
echo.

REM Commit with the comprehensive message
echo Creating commit...
git commit -m "Add advanced person animation and multiple video generators" -m "" -m "- Implemented StableVideoGenerator with sophisticated person-specific animations" -m "  - Walking, dancing, jumping with realistic physics" -m "  - Facial animations (smiling, talking, nodding)" -m "  - Body movements with proper rotation and scaling" -m "- Added multiple video generator backends with fallback pattern" -m "  - Primary: StableVideoGenerator for person animations" -m "  - Fallback: DynamiCrafter, SimpleVideoGenerator" -m "- Enhanced UI with categorized animation buttons" -m "  - AI-powered person animations (green buttons)" -m "  - Camera movements, motion effects, visual effects" -m "- Fixed Windows/WSL networking issues" -m "  - Updated CORS to allow all origins" -m "  - Created start_wsl.bat for proper WSL execution" -m "- Improved error handling and progress tracking" -m "- Added test scripts for debugging" -m "" -m "🤖 Generated with [Claude Code](https://claude.ai/code)" -m "" -m "Co-Authored-By: Claude <noreply@anthropic.com>"

echo.
echo Commit created successfully!
echo.

REM Push to GitHub
echo Pushing to GitHub...
git push origin master

echo.
echo ================================================
echo Done! Your changes have been pushed to GitHub.
echo ================================================
echo.
pause