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
from no_time_limit_wrapper import NoTimeLimitWrapper


def visualize_agent(model_path: str, episodes: int = 1, render: bool = False, realtime: bool = True, max_steps: int = None):
    """
    Visualize trained agent performance in real-time
    
    Args:
        model_path: Path to trained model
        episodes: Number of episodes to run (default: 1 for continuous viewing)
        render: Whether to render (requires display)
        realtime: If True, adds small delay for better real-time viewing
        max_steps: Maximum steps per episode (None = no limit, runs until failure)
    """
    import time
    import signal
    import sys
    
    # Flag for graceful shutdown
    interrupted = [False]
    
    def signal_handler(sig, frame):
        print("\n\n⚠ Interrupted by user. Stopping gracefully...")
        interrupted[0] = True
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Environment
    env_name = "dm_control/cartpole-balance-v0"
    
    if render:
        try:
            env = gym.make(env_name, render_mode="human")
            print("✓ Real-time rendering enabled")
        except Exception as e:
            print(f"⚠ Rendering not available: {e}")
            print("  Running without visualization")
            env = gym.make(env_name)
            render = False
    else:
        env = gym.make(env_name)
    
    # Wrap environment to remove time limits
    print("Removing time limit - episode will run indefinitely until pole falls...")
    env = NoTimeLimitWrapper(env)
    print("✓ Time limit removed - continuous balancing enabled")
    
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
        env.close()
        return
    
    # Run episodes
    episode_rewards = []
    episode_lengths = []
    
    print(f"\nRunning {episodes} episode(s) (Press Ctrl+C to stop)...")
    print("-" * 70)
    
    for episode in range(episodes):
        if interrupted[0]:
            break
            
        obs, _ = env.reset()
        state = flatten_observation(obs)
        episode_reward = 0
        total_steps = 0  # Total steps in episode
        
        print(f"\n🎯 Episode {episode + 1}: Starting continuous balancing...")
        if render:
            print("   Watch the cart-pole balance in real-time!")
            print("   (Press Ctrl+C to stop)")
        else:
            print("   Running... (Use --render to see visualization)")
        
        start_time = time.time()
        episode_ended = False
        
        while not interrupted[0] and not episode_ended:
            # Check step limit (user-defined)
            if max_steps is not None and total_steps >= max_steps:
                print(f"\n   ✓ Reached step limit ({max_steps} steps)")
                break
            
            # Select action (deterministic)
            action = agent.select_action(state, evaluate=True)
            
            # Take step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = flatten_observation(next_obs)
            
            # Check if pole actually fell (terminated = real failure)
            # Time limit (truncation) is ignored by NoTimeLimitWrapper
            if terminated:
                episode_ended = True
                print(f"\n   ⚠ Pole fell! Episode ended naturally at step {total_steps}")
                break
            
            # Episode continues indefinitely until pole falls or user interrupts
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # Render every step for real-time viewing
            if render:
                env.render()
                # Small delay for better visualization (optional)
                if realtime:
                    time.sleep(0.001)  # 1ms delay - environment runs very fast
            
            # Print progress every 100 steps
            if total_steps % 100 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
                sim_time = total_steps * 0.0025  # Approximate simulation time
                print(f"   Step {total_steps:,} | Reward: {episode_reward:.1f} | Real: {elapsed:.1f}s | Sim: {sim_time:.1f}s", 
                      end='\r', flush=True)
        
        elapsed_time = time.time() - start_time
        
        if interrupted[0]:
            print("\n\n⚠ Stopped by user (Ctrl+C)")
        else:
            episode_rewards.append(episode_reward)
            episode_lengths.append(total_steps)
            
            print(f"\n   ✓ Episode {episode + 1} completed!")
            print(f"   Total Steps: {total_steps:,}")
            print(f"   Final Reward: {episode_reward:.2f}")
            print(f"   Real Time: {elapsed_time:.1f} seconds")
            print(f"   Simulated Time: {total_steps * 0.0025:.1f} seconds")
            
            if episode_ended:
                print(f"   Episode ended: Pole fell naturally")
            else:
                print(f"   Episode stopped: User limit or interruption")
    
    env.close()
    
    # Statistics
    if episode_rewards:
        print("\n" + "-" * 70)
        print(f"\n📊 Statistics:")
        if len(episode_rewards) == 1:
            print(f"   Reward: {episode_rewards[0]:.2f}")
            print(f"   Steps: {episode_lengths[0]:,}")
            print(f"   Duration: {episode_lengths[0] * 0.0025:.1f} seconds (simulated)")
        else:
            print(f"   Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            print(f"   Mean Steps: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
            print(f"   Best: {np.max(episode_rewards):.2f} | Worst: {np.min(episode_rewards):.2f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize trained SAC agent')
    parser.add_argument('--model', type=str, default='results/best_model.pt',
                        help='Path to trained model (default: results/best_model.pt)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to run (default: 1 for continuous viewing)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment in real-time (requires display)')
    parser.add_argument('--no-delay', action='store_true',
                        help='Disable delay for maximum speed (faster than real-time)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum steps per episode (None = no limit, default: None)')
    args = parser.parse_args()
    
    print("="*70)
    print("SAC Agent Real-Time Visualization")
    print("="*70)
    print("\nTo see the agent balancing in real-time:")
    print("  python visualize.py --model results/best_model.pt --render")
    print("\n" + "="*70)
    
    visualize_agent(
        args.model, 
        args.episodes, 
        args.render, 
        realtime=not args.no_delay,
        max_steps=args.max_steps
    )
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
