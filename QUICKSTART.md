# Quick Reference Guide

## Installation & Setup

```bash
# 1. Navigate to project directory
cd cartpole-sac

# 2. Run setup script (Linux/Mac)
chmod +x setup.sh
./setup.sh

# OR manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Test installation
python test_installation.py
```

## Running the Code

### Basic Training
```bash
python train.py
```

### Custom Training
```bash
# More episodes
python train.py --episodes 500

# More evaluation episodes
python train.py --eval-episodes 20

# Different learning rate
python train.py --lr 1e-4

# Smaller batch size (if GPU memory is limited)
python train.py --batch-size 128

# CPU only
python train.py --device cpu
```

### All Available Arguments
```bash
python train.py --help

Options:
  --episodes        Number of training episodes (default: 300)
  --eval-episodes   Number of evaluation episodes (default: 10)
  --hidden-dim      Hidden layer dimension (default: 256)
  --lr              Learning rate (default: 3e-4)
  --batch-size      Batch size (default: 256)
  --gamma           Discount factor (default: 0.99)
  --tau             Soft update coefficient (default: 0.005)
  --alpha           Entropy temperature (default: 0.2)
  --device          Device to use (default: cuda)
```

## Expected Output

### During Training
```
======================================================================
CONFIGURATION
======================================================================
Environment: dm_control/cartpole-balance-v0
Training Seeds: [0, 1, 2]
Evaluation Seed: 10
Training Episodes: 300
...

======================================================================
Training with seed 0
======================================================================
Episode   10/300 (  3.3%) | Reward:  234.56 | Avg(10):  189.23 | Updates:    450
Episode   20/300 (  6.7%) | Reward:  567.89 | Avg(10):  445.67 | Updates:   1200
...
```

### After Training
You'll find in `results/` directory:
- `learning_curve.png` - Plot for your report
- `training_log.txt` - Detailed statistics
- `best_model.pt` - Best model weights

## File Structure

```
cartpole-sac/
├── README.md                  # Project overview
├── requirements.txt           # Dependencies
├── setup.sh                   # Setup script
├── test_installation.py       # Test script
├── train.py                   # Main training script ← RUN THIS
├── sac_agent.py              # SAC algorithm
├── networks.py               # Neural networks
├── replay_buffer.py          # Experience replay
├── utils.py                  # Helper functions
└── results/                  # Created after training
    ├── learning_curve.png
    ├── training_log.txt
    └── best_model.pt
```

## Hyperparameters Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| hidden_dim | 256 | Size of hidden layers in networks |
| lr | 3e-4 | Learning rate for Adam optimizer |
| gamma | 0.99 | Discount factor for future rewards |
| tau | 0.005 | Soft update rate for target network |
| alpha | 0.2 | Entropy temperature (exploration) |
| batch_size | 256 | Mini-batch size for updates |

## Performance Expectations

**RTX 5070 Ti (Your GPU):**
- Training time: ~5-10 minutes for 300 episodes
- Expected reward: 900-1000
- GPU utilization: ~30-50%
- VRAM usage: ~1-2 GB

**CPU:**
- Training time: ~20-30 minutes for 300 episodes
- Same reward, just slower

## Troubleshooting

### "CUDA out of memory"
```bash
python train.py --batch-size 128
```

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Environment not working
```bash
pip install --upgrade dm_control shimmy gymnasium
```

### Slow training
- Make sure you're using GPU: check with `nvidia-smi`
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

## For Your Report

1. **Code Link**: Upload to GitHub and include link
2. **Learning Curve**: Use `results/learning_curve.png`
3. **Statistics**: Find in `results/training_log.txt`
4. **Hyperparameters**: Listed in training_log.txt

### Report Template

```
Method: Soft Actor-Critic (SAC)
Training Seeds: 0, 1, 2
Evaluation Seed: 10
Episodes: 300

Hyperparameters:
- Learning Rate: 3e-4
- Gamma: 0.99
- Tau: 0.005
- Alpha: 0.2
- Batch Size: 256
- Hidden Dim: 256

Results:
- Training (last 10 episodes): XXX.XX ± YY.YY
- Evaluation (seed=10): XXX.XX ± YY.YY

[Include learning_curve.png]

Code: https://github.com/YOUR_USERNAME/cartpole-sac
```

## Tips

1. Run `test_installation.py` first to verify setup
2. Training is automatic - just run `train.py`
3. All results saved to `results/` folder
4. Use learning curve directly in report
5. Code is well-commented for understanding
