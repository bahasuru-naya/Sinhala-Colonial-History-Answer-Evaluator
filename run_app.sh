#!/bin/bash
# Sinhala Answer Evaluator - Quick Start Script for Linux/Mac
# This script runs the Streamlit application

echo ""
echo "============================================"
echo "Sinhala Answer Evaluator - Streamlit UI"
echo "============================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.9+ using your package manager"
    exit 1
fi

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "Error: Streamlit is not installed"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

# Check if OLLAMA is running
echo "Checking OLLAMA connection..."
if ! python3 -c "import requests; requests.get('http://localhost:11434/api/tags', timeout=2)" 2>/dev/null; then
    echo "Warning: OLLAMA does not appear to be running"
    echo "Please start OLLAMA in another terminal:"
    echo "  ollama serve"
    echo ""
    echo "Continuing anyway - you may see connection errors..."
    echo ""
fi

echo ""
echo "Starting Streamlit application..."
echo "Opening at: http://localhost:8501"
echo ""

python3 -m streamlit run ui/app.py
