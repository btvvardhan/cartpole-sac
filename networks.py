"""
Neural Network Architectures for SAC
Includes Actor (policy) and Critic (Q-function) networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    """
    Stochastic policy network with Gaussian distribution
    Outputs mean and log_std for continuous actions
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 log_std_min: float = -20, log_std_max: float = 2):
        """
        Initialize Actor network
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of hidden units
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        # Action rescaling (for bounded action spaces)
        self.action_scale = 1.0
        self.action_bias = 0.0
        
    def forward(self, state: torch.Tensor) -> tuple:
        """
        Forward pass through network
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (mean, log_std)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> tuple:
        """
        Sample action from policy distribution
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action, log_prob, mean)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Create normal distribution
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        
        # Apply tanh squashing
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Deterministic action (mean after tanh)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean


class Critic(nn.Module):
    """
    Twin Q-networks (double Q-learning)
    Outputs Q-values for state-action pairs
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize Critic network with twin Q-functions
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of hidden units
        """
        super(Critic, self).__init__()
        
        # Q1 network
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2 network
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple:
        """
        Forward pass through both Q-networks
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Tuple of (Q1_value, Q2_value)
        """
        sa = torch.cat([state, action], 1)
        
        # Q1
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        # Q2
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2
    
    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q1 network only
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Q1 value
        """
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1
