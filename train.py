"""
Main training script for SAC on Cart-Pole Balance task
Run this file to train and evaluate the agent
"""

import argparse
import numpy as np
import torch
import gymnasium as gym
import shimmy  # Required to register dm_control environments

from sac_agent import SACAgent
from utils import (
    set_seed, 
    create_results_dir, 
    plot_learning_curves,
    save_training_log,
    print_progress,
    flatten_observation,
    get_obs_dim
)


def train_agent(env_name: str, seed: int, num_episodes: int, 
                agent_config: dict, verbose: bool = True) -> tuple:
    """
    Train SAC agent on specified environment
    
    Args:
        env_name: Gymnasium environment name
        seed: Random seed
        num_episodes: Number of training episodes
        agent_config: Configuration dictionary for SAC agent
        verbose: Whether to print progress
        
    Returns:
        Tuple of (trained_agent, episode_rewards)
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Create environment
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    
    # Get environment dimensions
    state_dim = get_obs_dim(env.observation_space)
    action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
    
    # Initialize agent
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **agent_config
    )
    
    # Training loop
    episode_rewards = []
    update_steps = 0
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training with seed {seed}")
        print(f"{'='*70}")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = flatten_observation(obs)
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Select action
            action = agent.select_action(state, evaluate=False)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = flatten_observation(next_obs)
            done = terminated or truncated
            
            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update agent
            if len(agent.replay_buffer) >= agent.batch_size:
                agent.update()
                update_steps += 1
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if verbose and (episode + 1) % 10 == 0:
            running_avg = np.mean(episode_rewards[-10:])
            print_progress(episode, num_episodes, episode_reward, running_avg, update_steps)
    
    env.close()
    
    if verbose:
        print(f"\nTraining completed! Total updates: {update_steps}")
        print(f"Final 10 episodes avg: {np.mean(episode_rewards[-10:]):.2f}")
    
    return agent, episode_rewards


def evaluate_agent(agent: SACAgent, env_name: str, seed: int, 
                   num_episodes: int = 10, verbose: bool = True) -> list:
    """
    Evaluate trained agent
    
    Args:
        agent: Trained SAC agent
        env_name: Gymnasium environment name
        seed: Random seed for evaluation
        num_episodes: Number of evaluation episodes
        verbose: Whether to print progress
        
    Returns:
        List of episode rewards
    """
    env = gym.make(env_name)
    env.reset(seed=seed)
    
    episode_rewards = []
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Evaluating with seed {seed}")
        print(f"{'='*70}")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = flatten_observation(obs)
        episode_reward = 0
        done = False
        
        while not done:
            # Use deterministic policy (mean action)
            action = agent.select_action(state, evaluate=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = flatten_observation(next_obs)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        
        if verbose:
            print(f"Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.2f}")
    
    env.close()
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    if verbose:
        print(f"\nEvaluation Results:")
        print(f"  Mean: {mean_reward:.2f}")
        print(f"  Std: {std_reward:.2f}")
        print(f"  Min: {np.min(episode_rewards):.2f}")
        print(f"  Max: {np.max(episode_rewards):.2f}")
    
    return episode_rewards


def main():
    """Main function to run training and evaluation"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train SAC on Cart-Pole Balance')
    parser.add_argument('--episodes', type=int, default=300, 
                        help='Number of training episodes (default: 300)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden layer dimension (default: 256)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Entropy temperature (default: 0.2)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (must be cuda for GPU-only)')
    args = parser.parse_args()
    
    # Enforce GPU-only execution
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ERROR: CUDA is not available. This script requires GPU execution.\n"
            "Please check:\n"
            "  1. GPU drivers are installed: nvidia-smi\n"
            "  2. PyTorch with CUDA support is installed\n"
            "  3. CUDA is properly configured"
        )
    
    if args.device != 'cuda':
        raise ValueError(f"Device must be 'cuda' for GPU-only execution. Got: {args.device}")
    
    # Configuration
    ENV_NAME = "dm_control/cartpole-balance-v0"
    TRAIN_SEEDS = [0, 1, 2]
    EVAL_SEED = 10
    NUM_EPISODES = args.episodes
    EVAL_EPISODES = args.eval_episodes
    
    # Agent configuration (force GPU)
    agent_config = {
        'hidden_dim': args.hidden_dim,
        'lr': args.lr,
        'gamma': args.gamma,
        'tau': args.tau,
        'alpha': args.alpha,
        'batch_size': args.batch_size,
        'device': 'cuda'  # Force GPU
    }
    
    # Create results directory
    results_dir = create_results_dir('results')
    
    # Print configuration
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    print(f"Environment: {ENV_NAME}")
    print(f"Training Seeds: {TRAIN_SEEDS}")
    print(f"Evaluation Seed: {EVAL_SEED}")
    print(f"Training Episodes: {NUM_EPISODES}")
    print(f"Evaluation Episodes: {EVAL_EPISODES}")
    print(f"\nHyperparameters:")
    for key, value in agent_config.items():
        print(f"  {key}: {value}")
    print(f"\nDevice: GPU (CUDA)")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print("="*70)
    
    # Storage for results
    all_training_rewards = []
    all_agents = []
    all_eval_rewards = []
    
    # Train with multiple seeds
    for seed in TRAIN_SEEDS:
        agent, rewards = train_agent(
            ENV_NAME, 
            seed, 
            NUM_EPISODES, 
            agent_config,
            verbose=True
        )
        all_training_rewards.append(rewards)
        all_agents.append(agent)
    
    # Convert to numpy array
    all_training_rewards = np.array(all_training_rewards)
    
    # Evaluate all agents
    for i, agent in enumerate(all_agents):
        print(f"\nEvaluating agent trained with seed {TRAIN_SEEDS[i]}...")
        eval_rewards = evaluate_agent(
            agent, 
            ENV_NAME, 
            EVAL_SEED, 
            num_episodes=EVAL_EPISODES,
            verbose=True
        )
        all_eval_rewards.append(np.mean(eval_rewards))
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\nTraining Performance (last 10 episodes):")
    for i, seed in enumerate(TRAIN_SEEDS):
        final_mean = np.mean(all_training_rewards[i, -10:])
        print(f"  Seed {seed}: {final_mean:.2f}")
    
    print(f"\nEvaluation Performance (seed={EVAL_SEED}):")
    for i, seed in enumerate(TRAIN_SEEDS):
        print(f"  Agent (seed {seed}): {all_eval_rewards[i]:.2f}")
    
    overall_eval_mean = np.mean(all_eval_rewards)
    overall_eval_std = np.std(all_eval_rewards)
    print(f"\nOverall Evaluation: {overall_eval_mean:.2f} ± {overall_eval_std:.2f}")
    print("="*70)
    
    # Save results
    plot_learning_curves(
        all_training_rewards,
        all_eval_rewards,
        EVAL_SEED,
        save_path=f'{results_dir}/learning_curve.png'
    )
    
    save_training_log(
        all_training_rewards,
        all_eval_rewards,
        TRAIN_SEEDS,
        EVAL_SEED,
        NUM_EPISODES,
        agent_config,
        save_path=f'{results_dir}/training_log.txt'
    )
    
    # Save best model
    best_idx = np.argmax(all_eval_rewards)
    best_agent = all_agents[best_idx]
    best_seed = TRAIN_SEEDS[best_idx]
    
    best_agent.save(f'{results_dir}/best_model.pt')
    print(f"\nBest agent (trained with seed {best_seed}) saved!")
    print(f"Evaluation reward: {all_eval_rewards[best_idx]:.2f}")
    
    print(f"\nAll results saved to '{results_dir}/' directory")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
