@echo off
REM Setup Script for 2026 Global Trend Forecaster
REM This script installs dependencies and trains models

echo.
echo ============================================================
echo  2026 Global Trend Forecaster - Setup Script
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python first from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed successfully
echo.

echo [2/3] Training ML models...
python train_model.py
if errorlevel 1 (
    echo [ERROR] Failed to train models
    pause
    exit /b 1
)
echo [OK] Models trained successfully
echo.

echo [3/3] Starting Streamlit application...
echo.
echo ============================================================
echo  Streamlit app is starting...
echo  It should open automatically in your browser
echo  URL: http://localhost:8501
echo ============================================================
echo.

streamlit run app.py
pause
