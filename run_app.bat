@echo off
REM Sinhala Answer Evaluator - Quick Start Script for Windows
REM This script runs the Streamlit application

echo.
echo ============================================
echo Sinhala Answer Evaluator - Streamlit UI
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9+ and add it to PATH
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Error: Streamlit is not installed
    echo Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if OLLAMA is running
echo Checking OLLAMA connection...
python -c "import requests; requests.get('http://localhost:11434/api/tags', timeout=2)" >nul 2>&1
if errorlevel 1 (
    echo Warning: OLLAMA does not appear to be running
    echo Please start OLLAMA:
    echo   ollama serve
    echo.
    echo Continuing anyway - you may see connection errors...
    echo.
    pause
)

echo.
echo Starting Streamlit application...
echo Opening at: http://localhost:8501
echo.

python -m streamlit run ui/app.py

pause
