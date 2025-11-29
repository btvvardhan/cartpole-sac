"""
Soft Actor-Critic (SAC) Agent
Implements the SAC algorithm for continuous control
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple

from networks import Actor, Critic
from replay_buffer import ReplayBuffer


class SACAgent:
    """
    Soft Actor-Critic Agent
    
    Features:
    - Twin Q-networks (double Q-learning)
    - Stochastic policy with entropy regularization
    - Off-policy learning with replay buffer
    - Soft target updates
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        device: str = 'cuda'
    ):
        """
        Initialize SAC agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer size
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Entropy temperature
            buffer_size: Replay buffer capacity
            batch_size: Mini-batch size
            device: Device to run on ('cuda' or 'cpu')
        """
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Enforce GPU-only execution
        if device != 'cuda':
            raise ValueError(f"Device must be 'cuda' for GPU-only execution. Got: {device}")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ERROR: CUDA is not available. This agent requires GPU execution.\n"
                "Please ensure CUDA is properly configured and PyTorch has CUDA support."
            )
        
        self.device = torch.device('cuda')
        
        print(f"SAC Agent initialized on device: {self.device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy parameters to target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.critic_losses = []
        self.actor_losses = []
        
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Select action given state
        
        Args:
            state: Current state
            evaluate: If True, use deterministic policy (mean)
            
        Returns:
            Action to take
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                # Use mean action for evaluation
                _, _, action = self.actor.sample(state)
            else:
                # Sample from policy for exploration
                action, _, _ = self.actor.sample(state)
        
        return action.cpu().numpy()[0]
    
    def update(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Update actor and critic networks
        
        Returns:
            Tuple of (critic_loss, actor_loss) or (None, None) if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # ========== Update Critic ==========
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            
            # Compute target Q-values using target critic
            q1_next_target, q2_next_target = self.critic_target(next_states, next_actions)
            q_next_target = torch.min(q1_next_target, q2_next_target)
            
            # Add entropy term
            q_next_target = q_next_target - self.alpha * next_log_probs
            
            # Compute target Q value
            q_target = rewards + (1 - dones) * self.gamma * q_next_target
        
        # Get current Q estimates
        q1_current, q2_current = self.critic(states, actions)
        
        # Compute critic loss (MSE)
        critic_loss = F.mse_loss(q1_current, q_target) + F.mse_loss(q2_current, q_target)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ========== Update Actor ==========
        # Sample new actions from current policy
        new_actions, log_probs, _ = self.actor.sample(states)
        
        # Compute Q-values for new actions
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        # Compute actor loss (maximize Q - alpha * entropy)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ========== Soft Update Target Network ==========
        self._soft_update(self.critic, self.critic_target)
        
        # Store losses for logging
        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module):
        """
        Soft update target network parameters
        θ_target = τ * θ_source + (1 - τ) * θ_target
        
        Args:
            source: Source network
            target: Target network
        """
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, filepath: str):
        """
        Save agent's networks
        
        Args:
            filepath: Path to save checkpoint
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load agent's networks
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {filepath}")
