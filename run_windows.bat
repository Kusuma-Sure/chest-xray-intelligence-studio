@echo off
cd /d "%~dp0"
title Chest X-ray Intelligence Studio
echo ================================================================
echo Starting Chest X-ray Intelligence Studio...
echo Opening in browser: http://127.0.0.1:8000
echo (Keep this terminal window open while using the application)
echo ================================================================

rem Automatically open web browser after a short pause
start "" http://127.0.0.1:8000

if exist ".venv\Scripts\activate.bat" (
  call .venv\Scripts\activate.bat
  python app.py
) else (
  py -3.10 app.py || python app.py
)
pause
