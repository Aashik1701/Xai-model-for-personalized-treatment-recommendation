#!/bin/bash

echo "🏥 Hybrid Explainable AI Healthcare - Quick Setup"
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8+ using your system package manager"
    exit 1
fi

echo "✅ Python found"
echo

# Run the Python setup script
echo "Running setup script..."
python3 setup.py

echo
echo "Setup complete! Run 'code .' to open in VS Code."
