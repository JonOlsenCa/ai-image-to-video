# Instructions for Claude

## Command Execution Instructions

When providing commands for the user to run, ALWAYS specify:
1. **Where to run it**: WSL terminal, Windows Command Prompt, Windows PowerShell, etc.
2. **Which directory to run from**: The specific directory path where the command should be executed
3. **Admin requirements**: Whether administrator/sudo access is needed

Examples:
- "Run in WSL terminal from /home/user/project: `sudo apt update`" 
- "Run in Windows PowerShell (as Administrator) from any directory: `winget install GitHub.cli`"
- "Run in WSL terminal from /mnt/c/Github/ai-image-to-video (no sudo needed): `gh.exe auth login`"

## Installation Instructions

**IMPORTANT**: Whenever the user needs to install any software, packages, or dependencies, ALWAYS create a batch file (.bat) for them to run. Do not provide command line instructions for installations - always create a batch file instead.

Example approach:
- Create `install_[software_name].bat` in the repository root
- Include clear echo statements showing progress
- Handle any required sudo/admin permissions
- Verify installation success
- Provide clear instructions on how to run the batch file

## Batch File Best Practices

**CRITICAL**: Use this exact pattern for Windows batch files that run WSL commands:

```batch
@echo off
echo Starting [Application Name]...
echo.

echo Starting Backend Server...
start "Backend" wsl -d Ubuntu bash -l -c "cd /path/to/backend && source venv/bin/activate && python server.py"

timeout /t 8 > nul

echo Starting Frontend Server...
start "Frontend" wsl -d Ubuntu bash -l -c "cd /path/to/frontend && npm start"

timeout /t 20 > nul
start http://localhost:3000

echo Done! Check the opened browser window.
pause
```

**Key elements that MUST be included:**
- `wsl -d Ubuntu bash -l -c` (NOT `wsl bash -c`)
- Use `start "WindowName"` to open separate windows
- Use `timeout /t X > nul` for timing delays
- Use double quotes around the entire WSL command
- Use `bash -l` for login shell to load environment properly
- NEVER use complex nested quotes or bash -c with multiple quote levels