# SETUP INSTRUCTIONS FOR YOUR RTX 5070 Ti

## Step 1: Download/Clone the Repository

Place the `cartpole-sac` folder on your computer.

## Step 2: Navigate to Directory

```bash
cd cartpole-sac
```

## Step 3: Make Scripts Executable

```bash
chmod +x setup.sh run.sh
```

## Step 4: Run Everything (Easiest)

```bash
./run.sh
```

This will:
- Create virtual environment
- Install all dependencies
- Test installation
- Train the agent
- Save all results

## OR Step-by-Step (Manual)

### Option A: Automated Setup
```bash
./setup.sh
source venv/bin/activate
python test_installation.py
python train.py
```

### Option B: Manual Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python test_installation.py
python train.py
```

## Expected Timeline

1. **Setup**: 2-3 minutes (downloading packages)
2. **Testing**: 10 seconds
3. **Training**: 5-10 minutes on your RTX 5070 Ti

**Total: ~10-15 minutes**

## What You Get

After running, check the `results/` folder:

```
results/
├── learning_curve.png      ← USE THIS IN YOUR REPORT
├── training_log.txt        ← STATISTICS FOR YOUR REPORT
└── best_model.pt           ← TRAINED MODEL
```

## For Your Assignment Report

1. **Include the learning curve**: `results/learning_curve.png`
2. **Copy statistics**: From `results/training_log.txt`
3. **Upload code**: To GitHub
4. **Add GitHub link**: In your IEEE format report

## Verification Checklist

After running, verify:
- ✓ `results/` directory exists
- ✓ `learning_curve.png` created
- ✓ `training_log.txt` contains statistics
- ✓ Console shows "Training completed successfully!"

## GPU Verification

Before running, verify your GPU:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Should see:
- RTX 5070 Ti in nvidia-smi
- `True` from Python command

## File Overview

| File | Purpose |
|------|---------|
| `train.py` | Main script - RUN THIS |
| `sac_agent.py` | SAC algorithm implementation |
| `networks.py` | Actor & Critic networks |
| `replay_buffer.py` | Experience replay |
| `utils.py` | Plotting & logging |
| `test_installation.py` | Verify setup |
| `requirements.txt` | Dependencies |

## Common Issues

### Issue: "ModuleNotFoundError"
**Solution**: 
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size
```bash
python train.py --batch-size 128
```

### Issue: Slow training
**Check**: 
```bash
nvidia-smi  # Should show GPU usage
```

## Customization (Optional)

Want different settings?
```bash
python train.py --episodes 500 --lr 1e-4 --batch-size 128
```

See all options:
```bash
python train.py --help
```

## For GitHub Upload

1. Create new repository on GitHub
2. In `cartpole-sac` folder:
```bash
git init
git add .
git commit -m "SAC implementation for cart-pole balance"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cartpole-sac.git
git push -u origin main
```
3. Copy GitHub URL for your report

## Report Template

```latex
\section{Implementation}

The agent was trained using Soft Actor-Critic (SAC) algorithm 
implemented in PyTorch. Code available at: 
\url{https://github.com/YOUR_USERNAME/cartpole-sac}

\subsection{Hyperparameters}
\begin{itemize}
    \item Learning Rate: $3 \times 10^{-4}$
    \item Discount Factor ($\gamma$): 0.99
    \item Soft Update ($\tau$): 0.005
    \item Temperature ($\alpha$): 0.2
    \item Batch Size: 256
    \item Hidden Dimension: 256
\end{itemize}

\subsection{Results}
The agent was trained with seeds 0, 1, 2 for 300 episodes each
and evaluated with seed 10 for 10 episodes.

Training Performance (last 10 episodes): XXX.XX $\pm$ YY.YY
Evaluation Performance (seed=10): XXX.XX $\pm$ YY.YY

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{learning_curve.png}
    \caption{Learning curve showing mean and standard deviation}
\end{figure}
```

## Quick Commands Summary

```bash
# Setup and run everything
./run.sh

# Or step by step
./setup.sh
source venv/bin/activate
python train.py

# Test first
python test_installation.py

# Custom training
python train.py --episodes 500
```

## Expected Output Sample

```
======================================================================
CONFIGURATION
======================================================================
Environment: dm_control/cartpole-balance-v0
Training Seeds: [0, 1, 2]
Training Episodes: 300
...

======================================================================
Training with seed 0
======================================================================
Episode   10/300 | Reward:  234.56 | Avg(10):  189.23
Episode   20/300 | Reward:  567.89 | Avg(10):  445.67
...

✓ Learning curve saved to results/learning_curve.png
✓ Training log saved to results/training_log.txt
✓ Best agent saved to results/best_model.pt

Training completed successfully!
```

## Success Criteria

You're done when you see:
1. "Training completed successfully!" in console
2. Three files in `results/` folder
3. PNG shows nice learning curve
4. TXT file has all your statistics

**Then**: Upload to GitHub, add link to report, done! ✓

## Need Help?

If something doesn't work:
1. Check `python test_installation.py` passes all tests
2. Verify `nvidia-smi` shows your GPU
3. Make sure virtual environment is activated (you should see `(venv)` in terminal)
4. Try reducing batch size if memory issues

Good luck with your assignment! 🚀
