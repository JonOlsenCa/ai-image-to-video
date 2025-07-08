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