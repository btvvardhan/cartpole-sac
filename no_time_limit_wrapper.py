"""
Gymnasium wrapper to ignore time limits (truncation) and only stop on actual termination
This allows episodes to run indefinitely until the task actually fails
"""

import gymnasium as gym
from gymnasium import Wrapper


class NoTimeLimitWrapper(Wrapper):
    """
    Wrapper that ignores truncation (time limits) and only stops on termination (actual failure)
    Allows episodes to run indefinitely until the task naturally fails
    """
    
    def __init__(self, env):
        super().__init__(env)
        # Disable max_episode_steps in spec
        if hasattr(self.env, 'spec') and self.env.spec:
            self.env.spec.max_episode_steps = None
    
    def step(self, action):
        """Step environment but ignore truncation"""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Always set truncated to False - ignore time limits
        # Only stop if actually terminated (task failed)
        truncated = False
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment"""
        return self.env.reset(**kwargs)

