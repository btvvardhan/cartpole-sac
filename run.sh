#!/bin/bash

# One-command run script
# This script will set up and run the entire training pipeline

echo "=========================================="
echo "Cart-Pole SAC - Complete Pipeline"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Virtual environment not found. Running setup..."
    bash setup.sh
    if [ $? -ne 0 ]; then
        echo "Setup failed. Please check errors above."
        exit 1
    fi
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Test installation
echo ""
echo "Testing installation..."
python test_installation.py
if [ $? -ne 0 ]; then
    echo ""
    echo "Installation test failed. Please fix errors above."
    exit 1
fi

# Run training
echo ""
echo "Starting training..."
echo ""
python train.py "$@"

echo ""
echo "=========================================="
echo "Pipeline completed!"
echo "=========================================="
echo ""
echo "Results saved in 'results/' directory:"
echo "  - learning_curve.png (for your report)"
echo "  - training_log.txt (detailed statistics)"
echo "  - best_model.pt (trained model)"
echo ""
