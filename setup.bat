@echo off
echo 🏥 Hybrid Explainable AI Healthcare - Quick Setup
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Run the Python setup script
echo Running setup script...
python setup.py

echo.
echo Setup complete! Run "code ." to open in VS Code.
pause
