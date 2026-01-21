"""
Test script for SlimeVolley integration with prettyNEAT

This script tests:
1. Environment creation
2. Random agent performance
3. Observation and action spaces
4. Episode execution
"""

import numpy as np
import sys
from domain import make_env
from domain.config import games

def test_environment():
    """Test basic environment functionality"""
    print("=" * 60)
    print("Testing SlimeVolley Environment Integration")
    print("=" * 60)
    
    # Test 1: Check if game configuration exists
    print("\n1. Checking game configuration...")
    if 'slimevolley' in games:
        game = games['slimevolley']
        print(f"   ✓ Game configuration found")
        print(f"   - Input size: {game.input_size}")
        print(f"   - Output size: {game.output_size}")
        print(f"   - Max episode length: {game.max_episode_length}")
        print(f"   - Action selection: {game.actionSelect}")
    else:
        print("   ✗ Game configuration not found!")
        return False
    
    # Test 2: Create environment
    print("\n2. Creating environment...")
    try:
        env = make_env('SlimeVolley-v0')
        print(f"   ✓ Environment created successfully")
        print(f"   - Observation space: {env.observation_space}")
        print(f"   - Action space: {env.action_space}")
    except Exception as e:
        print(f"   ✗ Failed to create environment: {e}")
        return False
    
    # Test 3: Reset environment
    print("\n3. Testing environment reset...")
    try:
        obs = env.reset()
        print(f"   ✓ Reset successful")
        print(f"   - Observation shape: {obs.shape}")
        print(f"   - First observation: {obs}")
    except Exception as e:
        print(f"   ✗ Failed to reset environment: {e}")
        return False
    
    # Test 4: Run random agent
    print("\n4. Running random agent for one episode...")
    try:
        total_reward = 0
        done = False
        steps = 0
        max_steps = 100  # Just test first 100 steps
        
        while not done and steps < max_steps:
            # Random action (3 continuous values)
            action = np.random.randn(3)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if steps % 20 == 0:
                print(f"   Step {steps}: reward={reward:.2f}, total={total_reward:.2f}")
        
        print(f"   ✓ Episode completed")
        print(f"   - Total steps: {steps}")
        print(f"   - Total reward: {total_reward:.2f}")
        print(f"   - Episode finished: {done}")
    except Exception as e:
        print(f"   ✗ Failed during episode execution: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test different action formats
    print("\n5. Testing different action formats...")
    try:
        obs = env.reset()
        
        # Test scalar action
        obs, reward, done, info = env.step(0.5)
        print(f"   ✓ Scalar action works")
        
        # Test 1-element array
        obs, reward, done, info = env.step(np.array([0.5]))
        print(f"   ✓ Single-element array action works")
        
        # Test 3-element array
        obs, reward, done, info = env.step(np.array([0.5, -0.5, 0.8]))
        print(f"   ✓ Three-element array action works")
        
    except Exception as e:
        print(f"   ✗ Action format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Run a full episode
    print("\n6. Running complete episode...")
    try:
        obs = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            action = np.random.randn(3)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        print(f"   ✓ Full episode completed")
        print(f"   - Total steps: {steps}")
        print(f"   - Total reward: {total_reward:.2f}")
        
    except Exception as e:
        print(f"   ✗ Full episode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nYou can now train NEAT agents on SlimeVolley using:")
    print("  python neat_train.py -p p/slimevolley_quick.json -n 4")
    print("\nOr with the full configuration:")
    print("  python neat_train.py -p p/slimevolley.json -n 8")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    success = test_environment()
    sys.exit(0 if success else 1)
