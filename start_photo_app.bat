@echo off
cd /d "%~dp0"
echo Starting photo browser at http://127.0.0.1:8765 ...
start "" http://127.0.0.1:8765
.venv\Scripts\python.exe app.py --host 127.0.0.1 --port 8765
pause
