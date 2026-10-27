@echo off
rem OSA data viewer (double-click to run)
cd /d "%~dp0python"
start "" "..\.venv\Scripts\pythonw.exe" viewer_app.py
