@echo off
cd /d "%~dp0"
echo Starting photo browser with LAN access (phone on same WiFi) ...
echo The phone URL will be printed below. Allow firewall access if prompted.
start "" http://127.0.0.1:8765
.venv\Scripts\python.exe app.py --host 0.0.0.0 --port 8765
pause
