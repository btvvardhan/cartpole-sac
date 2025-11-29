"""
Test script to verify installation and GPU availability
Run this before training to check everything is set up correctly
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import torch
        print("✓ PyTorch")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib")
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        return False
    
    try:
        import gymnasium
        print("✓ Gymnasium")
    except ImportError as e:
        print(f"✗ Gymnasium: {e}")
        return False
    
    try:
        import dm_control
        print("✓ DeepMind Control Suite")
    except ImportError as e:
        print(f"✗ DeepMind Control Suite: {e}")
        return False
    
    try:
        import shimmy
        print("✓ Shimmy")
    except ImportError as e:
        print(f"✗ Shimmy: {e}")
        return False
    
    return True


def test_cuda():
    """Test CUDA availability"""
    import torch
    
    print("\nTesting CUDA...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        
        # Test tensor creation on GPU
        try:
            x = torch.randn(3, 3).cuda()
            print(f"✓ GPU tensor creation successful")
            return True
        except Exception as e:
            print(f"✗ GPU tensor creation failed: {e}")
            return False
    else:
        print("✗ CUDA is not available")
        print("  Training will run on CPU (slower)")
        return False


def test_environment():
    """Test if environment can be created"""
    print("\nTesting environment creation...")
    
    try:
        import gymnasium as gym
        import shimmy  # Required to register dm_control environments
        from utils import get_obs_dim
        
        env = gym.make("dm_control/cartpole-balance-v0")
        env.reset()
        print("✓ Environment created successfully")
        
        state_dim = get_obs_dim(env.observation_space)
        action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False


def test_modules():
    """Test if custom modules can be imported"""
    print("\nTesting custom modules...")
    
    try:
        from networks import Actor, Critic
        print("✓ networks.py")
    except ImportError as e:
        print(f"✗ networks.py: {e}")
        return False
    
    try:
        from replay_buffer import ReplayBuffer
        print("✓ replay_buffer.py")
    except ImportError as e:
        print(f"✗ replay_buffer.py: {e}")
        return False
    
    try:
        from sac_agent import SACAgent
        print("✓ sac_agent.py")
    except ImportError as e:
        print(f"✗ sac_agent.py: {e}")
        return False
    
    try:
        from utils import set_seed, create_results_dir
        print("✓ utils.py")
    except ImportError as e:
        print(f"✗ utils.py: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("="*70)
    print("CART-POLE SAC - INSTALLATION TEST")
    print("="*70)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n⚠ Some packages are missing. Run: pip install -r requirements.txt")
    
    # Test CUDA
    cuda_available = test_cuda()
    if not cuda_available:
        print("\n⚠ CUDA not available. Training will be slower on CPU.")
    
    # Test environment
    if not test_environment():
        all_passed = False
        print("\n⚠ Environment creation failed. Check dm_control installation.")
    
    # Test custom modules
    if not test_modules():
        all_passed = False
        print("\n⚠ Custom modules import failed. Check file structure.")
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nYou're ready to train! Run:")
        print("  python train.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("="*70)
        print("\nPlease fix the errors above before training.")
        sys.exit(1)
    
    print("")


if __name__ == "__main__":
    main()
