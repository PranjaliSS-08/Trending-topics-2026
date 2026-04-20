#!/bin/bash

# Setup Script for 2026 Global Trend Forecaster
# This script installs dependencies and trains models

echo ""
echo "============================================================"
echo "  2026 Global Trend Forecaster - Setup Script"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed"
    echo "Please install Python3 first from https://www.python.org/"
    exit 1
fi

echo "[1/3] Installing dependencies from requirements.txt..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi
echo "[OK] Dependencies installed successfully"
echo ""

echo "[2/3] Training ML models..."
python3 train_model.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to train models"
    exit 1
fi
echo "[OK] Models trained successfully"
echo ""

echo "[3/3] Starting Streamlit application..."
echo ""
echo "============================================================"
echo "  Streamlit app is starting..."
echo "  It should open automatically in your browser"
echo "  URL: http://localhost:8501"
echo "============================================================"
echo ""

streamlit run app.py
