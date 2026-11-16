@echo off
cd /d "%~dp0"
echo ================================================================
echo Chest X-ray Intelligence Studio -- Windows Environment Setup
echo ================================================================

echo Checking Python installation...
py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    echo Using Python 3.11...
    py -3.11 -m venv .venv
    goto venv_created
)

py -3.10 --version >nul 2>&1
if not errorlevel 1 (
    echo Using Python 3.10...
    py -3.10 -m venv .venv
    goto venv_created
)

echo Using default Python...
python -m venv .venv
if errorlevel 1 goto error

:venv_created
call .venv\Scripts\activate.bat
echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 goto error

echo.
echo ================================================================
echo Setup complete! Running system model check...
echo ================================================================
python scripts\check_models.py
echo.
echo You can now start the application with: run_windows.bat
pause
exit /b 0

:error
echo.
echo ================================================================
echo Setup failed. Please check the error message above.
echo ================================================================
pause
exit /b 1
