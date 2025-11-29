#!/bin/bash

# One-command run script - GPU-ONLY VERSION
# This script will set up and run the entire training pipeline on GPU

echo "=========================================="
echo "Cart-Pole SAC - GPU-Only Pipeline"
echo "=========================================="

# Check if conda environment exists
if ! conda env list | grep -q "^sac "; then
    echo ""
    echo "Conda environment 'sac' not found. Creating..."
    conda create -n sac python=3.10 -y
    eval "$(conda shell.bash hook)"
    conda activate sac
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo ""
    echo "Activating conda environment 'sac'..."
    eval "$(conda shell.bash hook)"
    conda activate sac
fi

# Verify GPU is available
echo ""
echo "Verifying GPU availability..."
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA not available'; print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')" || {
    echo ""
    echo "ERROR: GPU is not available. This script requires GPU execution."
    echo "Please check:"
    echo "  1. GPU drivers: nvidia-smi"
    echo "  2. PyTorch with CUDA support is installed"
    echo "  3. CUDA is properly configured"
    exit 1
}

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
echo "Starting training on GPU..."
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

