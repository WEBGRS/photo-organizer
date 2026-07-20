@echo off
cd /d "%~dp0"
echo Auto-naming scenery cluster folders (needs the D: photo drive connected)...
.venv\Scripts\python.exe label_clusters.py
pause
