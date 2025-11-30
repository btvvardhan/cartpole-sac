#!/usr/bin/env python3
"""
Generate comprehensive metrics and visualizations for GitHub README
Run this after training to create additional statistics and plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def load_model_stats(model_path='results/best_model.pt'):
    """Extract model statistics"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Count parameters
    actor_params = sum(p.numel() for p in checkpoint['actor_state_dict'].values())
    critic_params = sum(p.numel() for p in checkpoint['critic_state_dict'].values())
    total_params = actor_params + critic_params
    
    return {
        'actor_parameters': actor_params,
        'critic_parameters': critic_params,
        'total_parameters': total_params,
        'model_size_mb': Path(model_path).stat().st_size / (1024 * 1024)
    }

def parse_training_log(log_path='results/training_log.txt'):
    """Parse training log and extract metrics"""
    with open(log_path, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Extract training results
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'Seed' in line and 'Final 10 episodes:' in lines[i+1]:
            seed = line.split('Seed')[1].split(':')[0].strip()
            metrics[f'seed_{seed}'] = {}
            # Extract final mean and std
            if 'Final 10 episodes:' in lines[i+1]:
                parts = lines[i+1].split('±')
                mean = float(parts[0].split(':')[1].strip())
                std = float(parts[1].strip())
                metrics[f'seed_{seed}']['final_mean'] = mean
                metrics[f'seed_{seed}']['final_std'] = std
            # Extract best episode
            if 'Best episode:' in lines[i+2]:
                best = float(lines[i+2].split(':')[1].strip())
                metrics[f'seed_{seed}']['best'] = best
    
    # Extract evaluation results
    eval_section = content.split('EVALUATION RESULTS')[1] if 'EVALUATION RESULTS' in content else ""
    eval_rewards = []
    for line in eval_section.split('\n'):
        if 'Agent trained with seed' in line:
            reward = float(line.split(':')[1].strip())
            eval_rewards.append(reward)
    
    metrics['evaluation_rewards'] = eval_rewards
    metrics['best_eval_reward'] = max(eval_rewards) if eval_rewards else None
    
    return metrics

def create_performance_summary():
    """Create a comprehensive performance summary"""
    model_stats = load_model_stats()
    training_metrics = parse_training_log()
    
    summary = {
        'model': model_stats,
        'training': training_metrics,
        'summary': {
            'best_evaluation_reward': training_metrics.get('best_eval_reward', 'N/A'),
            'average_evaluation_reward': np.mean(training_metrics.get('evaluation_rewards', [])),
            'model_parameters': model_stats['total_parameters'],
            'model_size_mb': round(model_stats['model_size_mb'], 2)
        }
    }
    
    # Save to JSON
    with open('results/metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"\nModel Statistics:")
    print(f"  Total Parameters: {model_stats['total_parameters']:,}")
    print(f"  Actor Parameters: {model_stats['actor_parameters']:,}")
    print(f"  Critic Parameters: {model_stats['critic_parameters']:,}")
    print(f"  Model Size: {model_stats['model_size_mb']:.2f} MB")
    
    print(f"\nEvaluation Results:")
    if training_metrics.get('evaluation_rewards'):
        rewards = training_metrics['evaluation_rewards']
        print(f"  Best Reward: {max(rewards):.2f}")
        print(f"  Average Reward: {np.mean(rewards):.2f}")
        print(f"  Std Reward: {np.std(rewards):.2f}")
    
    print(f"\nMetrics saved to: results/metrics_summary.json")
    print("="*60)
    
    return summary

def create_model_comparison_chart():
    """Create a comparison chart of all three models"""
    training_metrics = parse_training_log()
    
    if not training_metrics.get('evaluation_rewards'):
        print("No evaluation data found")
        return
    
    rewards = training_metrics['evaluation_rewards']
    seeds = ['Seed 0', 'Seed 1', 'Seed 2']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(seeds, rewards, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, reward) in enumerate(zip(bars, rewards)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add mean line
    mean_reward = np.mean(rewards)
    plt.axhline(y=mean_reward, color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_reward:.2f}')
    
    plt.ylabel('Evaluation Reward', fontsize=13, fontweight='bold')
    plt.title('Model Comparison - Evaluation Performance (Seed 10)', 
              fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([min(rewards) - 20, max(rewards) + 20])
    plt.tight_layout()
    
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Model comparison chart saved to: results/model_comparison.png")
    plt.close()

if __name__ == "__main__":
    print("\nGenerating comprehensive metrics and visualizations...")
    print("="*60)
    
    # Create results directory if needed
    Path('results').mkdir(exist_ok=True)
    
    # Generate summary
    summary = create_performance_summary()
    
    # Create comparison chart
    create_model_comparison_chart()
    
    print("\n✓ All metrics generated successfully!")
    print("\nFiles created:")
    print("  - results/metrics_summary.json")
    print("  - results/model_comparison.png")

