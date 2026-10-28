@echo off
rem CBCT segmentation QC viewer (double-click to run, internal use)
cd /d "%~dp0python"
start "" "..\.venv\Scripts\pythonw.exe" qc_viewer.py
