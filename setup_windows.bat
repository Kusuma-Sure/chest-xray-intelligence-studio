@echo off
cd /d "%~dp0"
echo Creating Python 3.10 virtual environment...
py -3.10 -m venv .venv
if errorlevel 1 goto error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto error
echo.
echo Install PyTorch and Torchvision. If pip cannot find a matching build, use https://pytorch.org/get-started/locally/
pip install torch torchvision
if errorlevel 1 goto error
pip install -r requirements.txt
if errorlevel 1 goto error
echo.
echo Setup complete. Start the application with run_windows.bat
pause
exit /b 0
:error
echo Setup failed. Copy the error above and share it for troubleshooting.
pause
exit /b 1
