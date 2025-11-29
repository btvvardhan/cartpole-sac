"""
Visualize a trained agent
Run this after training to see the agent in action
"""

import argparse
import torch
import gymnasium as gym
import numpy as np
import shimmy  # Required to register dm_control environments
from sac_agent import SACAgent
from utils import flatten_observation, get_obs_dim


def visualize_agent(model_path: str, episodes: int = 5, render: bool = False):
    """
    Visualize trained agent performance
    
    Args:
        model_path: Path to trained model
        episodes: Number of episodes to run
        render: Whether to render (requires display)
    """
    # Environment
    env_name = "dm_control/cartpole-balance-v0"
    
    if render:
        try:
            env = gym.make(env_name, render_mode="human")
        except:
            print("Rendering not available, running without visualization")
            env = gym.make(env_name)
    else:
        env = gym.make(env_name)
    
    # Get dimensions
    state_dim = get_obs_dim(env.observation_space)
    action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
    
    # Create agent
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device='cuda'
    )
    
    # Load trained model
    print(f"Loading model from {model_path}...")
    try:
        checkpoint = torch.load(model_path)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Run episodes
    episode_rewards = []
    episode_lengths = []
    
    print(f"\nRunning {episodes} episodes...")
    print("-" * 50)
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = flatten_observation(obs)
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Select action (deterministic)
            action = agent.select_action(state, evaluate=True)
            
            # Take step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = flatten_observation(next_obs)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if render and episode < 3:  # Only render first 3 episodes
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    env.close()
    
    # Statistics
    print("-" * 50)
    print(f"\nStatistics over {episodes} episodes:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Std Reward: {np.std(episode_rewards):.2f}")
    print(f"  Min Reward: {np.min(episode_rewards):.2f}")
    print(f"  Max Reward: {np.max(episode_rewards):.2f}")
    print(f"  Mean Length: {np.mean(episode_lengths):.1f} steps")


def main():
    parser = argparse.ArgumentParser(description='Visualize trained SAC agent')
    parser.add_argument('--model', type=str, default='results/best_model.pt',
                        help='Path to trained model (default: results/best_model.pt)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run (default: 5)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment (requires display)')
    args = parser.parse_args()
    
    print("="*50)
    print("SAC Agent Visualization")
    print("="*50)
    
    visualize_agent(args.model, args.episodes, args.render)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
