# Cart-Pole Balance using SAC

Implementation of Soft Actor-Critic (SAC) for the DeepMind Control Suite Cart-Pole Balance task.

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- 8GB+ GPU memory (your RTX 5070 Ti is perfect)

## Installation

```bash
# Clone or download this repository
cd cartpole-sac

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Train and evaluate the agent
python train.py

# Or with custom settings
python train.py --episodes 500 --eval-episodes 20
```

## Project Structure

```
cartpole-sac/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── train.py              # Main training script
├── sac_agent.py          # SAC algorithm implementation
├── networks.py           # Actor and Critic networks
├── replay_buffer.py      # Experience replay buffer
├── utils.py              # Utility functions
└── results/              # Output directory (created automatically)
    ├── learning_curve.png
    ├── training_log.txt
    └── best_model.pt
```

## Hyperparameters

Default hyperparameters (optimized for cart-pole):

- **Hidden Dimension**: 256
- **Learning Rate**: 3e-4
- **Discount Factor (γ)**: 0.99
- **Soft Update (τ)**: 0.005
- **Temperature (α)**: 0.2
- **Batch Size**: 256
- **Replay Buffer**: 1,000,000
- **Training Episodes**: 300
- **Training Seeds**: [0, 1, 2]
- **Evaluation Seed**: 10

## Expected Performance

- Training time: ~5-10 minutes on RTX 5070 Ti
- Expected reward: 900-1000 (normalized to [0, 1000])
- Convergence: ~100-150 episodes

## Output

After training, you'll find:
- `results/learning_curve.png` - Training and evaluation curves
- `results/training_log.txt` - Detailed training logs
- `results/best_model.pt` - Best performing model
- Console output with episode rewards and statistics

## Usage for Assignment

1. Run the training script
2. Check `results/learning_curve.png` for your report
3. Check `results/training_log.txt` for statistics
4. Upload code to GitHub and include link in your report

## Customization

Edit `train.py` to modify hyperparameters:

```python
# Training configuration
TRAIN_SEEDS = [0, 1, 2]
EVAL_SEED = 10
NUM_EPISODES = 300

# SAC hyperparameters
agent = SACAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=256,
    lr=3e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    batch_size=256
)
```

## Troubleshooting

**CUDA out of memory**: Reduce batch size in `train.py`
```python
batch_size=128  # Instead of 256
```

**Import errors**: Make sure all dependencies are installed
```bash
pip install -r requirements.txt
```

**Slow training**: Verify GPU is being used
```python
# Should print: cuda
print(torch.cuda.is_available())
```

## References

- [Soft Actor-Critic Paper](https://arxiv.org/abs/1812.05905)
- [DeepMind Control Suite](https://github.com/deepmind/dm_control)
- [Gymnasium](https://gymnasium.farama.org/)

## License

MIT License - Feel free to use for academic purposes
