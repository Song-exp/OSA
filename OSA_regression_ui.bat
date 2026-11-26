@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "python\osa_regression_ui.py"
if errorlevel 1 pause
