"""
Utility functions for training and evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Union, Dict


def set_seed(seed: int):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    import torch
    import random
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Make PyTorch operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_results_dir(dirname: str = 'results'):
    """
    Create directory for saving results
    
    Args:
        dirname: Name of directory to create
        
    Returns:
        Path to created directory
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


def plot_learning_curves(
    all_training_rewards: np.ndarray,
    eval_rewards: List[float],
    eval_seed: int,
    save_path: str = 'results/learning_curve.png'
):
    """
    Plot learning curves with mean and standard deviation
    
    Args:
        all_training_rewards: Array of shape (n_seeds, n_episodes) with training rewards
        eval_rewards: List of evaluation rewards for each seed
        eval_seed: Seed used for evaluation
        save_path: Path to save the plot
    """
    # Calculate statistics
    mean_rewards = np.mean(all_training_rewards, axis=0)
    std_rewards = np.std(all_training_rewards, axis=0)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot training curve
    episodes = np.arange(1, len(mean_rewards) + 1)
    plt.plot(episodes, mean_rewards, label='Training Mean', color='#2E86AB', linewidth=2)
    plt.fill_between(
        episodes,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.3,
        color='#2E86AB',
        label='Training Std'
    )
    
    # Add evaluation results
    eval_mean = np.mean(eval_rewards)
    eval_std = np.std(eval_rewards)
    
    plt.axhline(
        y=eval_mean,
        color='#A23B72',
        linestyle='--',
        linewidth=2,
        label=f'Evaluation Mean (seed={eval_seed})'
    )
    plt.fill_between(
        [0, len(mean_rewards)],
        eval_mean - eval_std,
        eval_mean + eval_std,
        alpha=0.2,
        color='#A23B72'
    )
    
    # Styling
    plt.xlabel('Episode', fontsize=13, fontweight='bold')
    plt.ylabel('Reward', fontsize=13, fontweight='bold')
    plt.title('SAC Learning Curve - Cart-Pole Balance Task', 
              fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Learning curve saved to {save_path}")
    plt.close()


def save_training_log(
    all_training_rewards: np.ndarray,
    eval_rewards: List[float],
    train_seeds: List[int],
    eval_seed: int,
    num_episodes: int,
    hyperparameters: dict,
    save_path: str = 'results/training_log.txt'
):
    """
    Save detailed training log to text file
    
    Args:
        all_training_rewards: Training rewards for all seeds
        eval_rewards: Evaluation rewards
        train_seeds: Seeds used for training
        eval_seed: Seed used for evaluation
        num_episodes: Number of training episodes
        hyperparameters: Dictionary of hyperparameters
        save_path: Path to save log file
    """
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SAC TRAINING LOG - CART-POLE BALANCE\n")
        f.write("="*70 + "\n\n")
        
        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Environment: dm_control/cartpole-balance-v0\n")
        f.write(f"Training Seeds: {train_seeds}\n")
        f.write(f"Evaluation Seed: {eval_seed}\n")
        f.write(f"Number of Episodes: {num_episodes}\n\n")
        
        # Hyperparameters
        f.write("HYPERPARAMETERS\n")
        f.write("-" * 70 + "\n")
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Training Results
        f.write("TRAINING RESULTS\n")
        f.write("-" * 70 + "\n")
        for i, seed in enumerate(train_seeds):
            final_10_mean = np.mean(all_training_rewards[i, -10:])
            final_10_std = np.std(all_training_rewards[i, -10:])
            f.write(f"Seed {seed}:\n")
            f.write(f"  Final 10 episodes: {final_10_mean:.2f} ± {final_10_std:.2f}\n")
            f.write(f"  Best episode: {np.max(all_training_rewards[i]):.2f}\n")
        
        overall_final_mean = np.mean(all_training_rewards[:, -10:])
        overall_final_std = np.std(all_training_rewards[:, -10:])
        f.write(f"\nOverall (last 10 episodes):\n")
        f.write(f"  Mean: {overall_final_mean:.2f}\n")
        f.write(f"  Std: {overall_final_std:.2f}\n\n")
        
        # Evaluation Results
        f.write("EVALUATION RESULTS (seed={})\n".format(eval_seed))
        f.write("-" * 70 + "\n")
        for i, seed in enumerate(train_seeds):
            f.write(f"Agent trained with seed {seed}: {eval_rewards[i]:.2f}\n")
        
        eval_mean = np.mean(eval_rewards)
        eval_std = np.std(eval_rewards)
        f.write(f"\nOverall Evaluation:\n")
        f.write(f"  Mean: {eval_mean:.2f}\n")
        f.write(f"  Std: {eval_std:.2f}\n")
        f.write(f"  Best: {np.max(eval_rewards):.2f}\n")
        f.write(f"  Worst: {np.min(eval_rewards):.2f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("End of Log\n")
        f.write("="*70 + "\n")
    
    print(f"Training log saved to {save_path}")


def print_progress(episode: int, total_episodes: int, episode_reward: float, 
                   running_avg: float, update_steps: int = 0):
    """
    Print training progress
    
    Args:
        episode: Current episode number
        total_episodes: Total number of episodes
        episode_reward: Reward for current episode
        running_avg: Running average of recent rewards
        update_steps: Number of update steps performed
    """
    progress = (episode + 1) / total_episodes * 100
    print(f"Episode {episode + 1:4d}/{total_episodes} ({progress:5.1f}%) | "
          f"Reward: {episode_reward:7.2f} | "
          f"Avg(10): {running_avg:7.2f} | "
          f"Updates: {update_steps:6d}")


def smooth_rewards(rewards: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Smooth reward curve using moving average
    
    Args:
        rewards: Array of rewards
        window: Window size for moving average
        
    Returns:
        Smoothed rewards
    """
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    # Pad the beginning to maintain same length
    padding = np.full(window - 1, smoothed[0])
    return np.concatenate([padding, smoothed])


def flatten_observation(obs: Union[np.ndarray, Dict]) -> np.ndarray:
    """
    Flatten observation from Dict or array format to 1D array
    
    Args:
        obs: Observation (can be dict or array)
        
    Returns:
        Flattened observation as 1D numpy array
    """
    if isinstance(obs, dict):
        # For Dict observations (like dm_control), concatenate all values
        return np.concatenate([np.array(obs[key]).flatten() for key in sorted(obs.keys())])
    else:
        # Already an array, just flatten it
        return np.array(obs).flatten()


def get_obs_dim(observation_space) -> int:
    """
    Get observation dimension from observation space
    
    Args:
        observation_space: Gymnasium observation space
        
    Returns:
        Dimension of flattened observation
    """
    # Handle Dict observation spaces (like dm_control)
    if hasattr(observation_space, 'spaces') and isinstance(observation_space.spaces, dict):
        # Sum up dimensions from all subspaces
        total_dim = 0
        for key in sorted(observation_space.spaces.keys()):
            space = observation_space.spaces[key]
            if hasattr(space, 'shape'):
                total_dim += np.prod(space.shape)
            else:
                # Fallback: sample and flatten
                sample = space.sample()
                total_dim += np.array(sample).flatten().shape[0]
        return int(total_dim)
    else:
        # Regular Box space
        if hasattr(observation_space, 'shape') and observation_space.shape is not None:
            return int(np.prod(observation_space.shape))
        else:
            # Fallback: sample and check
            sample = observation_space.sample()
            return int(np.array(sample).flatten().shape[0])
