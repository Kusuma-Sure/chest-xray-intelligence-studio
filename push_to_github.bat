@echo off
title Push Chest X-ray Capstone to GitHub
echo =========================================================================
echo              PUSH CHEXPERT CAPSTONE TO GITHUB
echo =========================================================================
echo.

cd /d "%~dp0"

echo Pushing current branch to GitHub (https://github.com/VEDANTH0517/chest-xray-intelligence-studio)...
git add .
git commit -m "Update CheXpert Capstone codebase"
git push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] Pushed successfully to GitHub!
) else (
    echo.
    echo [ERROR] Push failed. Check your network or GitHub permissions.
)
echo.
pause
