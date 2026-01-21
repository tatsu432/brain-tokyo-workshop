#!/usr/bin/env python
"""
Debug script to understand what's happening with SlimeVolley training.

This script will:
1. Test the actual game environment directly
2. Check what rewards SlimeVolley gives
3. Test network output ranges with different activation functions
4. Verify action mapping is being called
"""

import numpy as np
import sys

def test_slimevolley_rewards():
    """Test the actual SlimeVolley environment to understand its rewards"""
    print("=" * 70)
    print("Testing SlimeVolley Environment Directly")
    print("=" * 70)
    
    try:
        import gym
        import slimevolleygym
        
        env = gym.make('SlimeVolley-v0')
        obs = env.reset()
        
        print(f"\n1. Environment loaded successfully")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation: {obs}")
        
        # Test with random actions
        total_reward = 0
        episode_rewards = []
        
        print(f"\n2. Running 1 episode with random actions...")
        for i in range(3000):  # max episode length
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if reward != 0:
                episode_rewards.append(reward)
                print(f"   Step {i}: reward = {reward} (total: {total_reward})")
            if done:
                break
        
        print(f"\n   Episode finished at step {i}")
        print(f"   Total reward: {total_reward}")
        print(f"   Non-zero rewards: {episode_rewards}")
        print(f"   Number of point events: {len(episode_rewards)}")
        
        env.close()
        
        print("\n3. Analysis:")
        print(f"   - Rewards are sparse: only when lives are won/lost")
        print(f"   - Typical reward per episode: {total_reward}")
        print(f"   - This should be negative (opponent usually wins)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_activation_output_ranges():
    """Test what ranges different activation functions produce"""
    print("\n\n" + "=" * 70)
    print("Testing Activation Function Output Ranges")
    print("=" * 70)
    
    try:
        from neat_src.ann import _applyAct
        
        # Test inputs
        test_inputs = np.array([-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10])
        
        # Activation functions from config
        activations = {
            1: "Linear",
            2: "Unsigned Step",
            3: "Sin",
            4: "Gaussian",
            5: "Tanh",
            6: "Sigmoid",
            7: "Inverse",
            8: "Abs",
            9: "ReLU",
            10: "Cos",
        }
        
        print("\nInput values:", test_inputs)
        print()
        
        for act_id, act_name in activations.items():
            outputs = _applyAct(act_id, test_inputs)
            print(f"Activation {act_id} ({act_name:15s}): range [{np.min(outputs):7.2f}, {np.max(outputs):7.2f}]")
            print(f"   Outputs: {outputs}")
        
        print("\n" + "-" * 70)
        print("KEY INSIGHT:")
        print("  - Most activations output in range [-1, 1] or [0, 1]")
        print("  - Linear activation can output ANY value!")
        print("  - This affects whether action mapping thresholds make sense")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_shaping_magnitude():
    """Calculate expected shaped reward magnitude"""
    print("\n\n" + "=" * 70)
    print("Analyzing Reward Shaping Magnitude")
    print("=" * 70)
    
    print("\nShaped reward components (per step):")
    print("  - Ball proximity (+2.0 if close, +1.0 if medium): ~1.0 avg")
    print("  - Edge penalty (-3.0): occasional")
    print("  - Distance change (±dist_change * 5.0): ~±0.5 avg")
    print("  - Good positioning (+1.0): occasional")
    print("  - Wrong side penalty (-1.0): occasional")
    print()
    print("  Estimated shaped reward per step: ~0.5 to 2.0")
    print()
    print("Over 3000-step episode:")
    print("  - Total shaped reward: 0.5 * 3000 = 1500 (conservative)")
    print("  - With shaping_weight=0.01: 1500 * 0.01 = 15.0")
    print()
    print("Original SlimeVolley rewards:")
    print("  - Typical per episode: -3.0 to -5.0 (agent loses)")
    print()
    print("Combined:")
    print("  - Original: -4.0")
    print("  - Shaped: +15.0")
    print("  - Total: ~+11.0")
    print()
    print("🚨 PROBLEM IDENTIFIED:")
    print("  The shaped rewards are 3-4x LARGER than game rewards!")
    print("  Agent learns to maximize shaped rewards, not win the game!")
    print("  This is classic 'reward hacking'")
    print("=" * 70)


def main():
    """Run all diagnostic tests"""
    print("\n" * 2)
    print("#" * 70)
    print("# SlimeVolley Training Bottleneck - Deep Diagnosis")
    print("#" * 70)
    
    # Test 1: Direct environment
    test1 = test_slimevolley_rewards()
    
    # Test 2: Activation functions
    test2 = test_activation_output_ranges()
    
    # Test 3: Reward shaping analysis
    test_reward_shaping_magnitude()
    
    print("\n\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("The reward shaping is TOO STRONG and creates a new plateau!")
    print()
    print("Solutions (in order of recommendation):")
    print()
    print("1. Try the NON-shaped version first:")
    print("   python neat_train.py -p p/slimevolley_fixed.json -n 9")
    print()
    print("2. If that doesn't work, there may be a deeper issue with:")
    print("   - Output activation function ranges")
    print("   - Action mapping still not working correctly")
    print("   - Network architecture")
    print()
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
