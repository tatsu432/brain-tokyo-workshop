#!/usr/bin/env python
"""
Test random baseline: What fitness does a completely random policy get?
This will tell us if -3.80 is good or bad.
"""

import numpy as np
import gym

print("=" * 80)
print("RANDOM BASELINE TEST")
print("=" * 80)

# Try to import slimevolleygym to register the environment
try:
    import slimevolleygym
    HAS_SLIMEVOLLEY = True
except:
    print("WARNING: slimevolleygym module not found, but env might still be registered")
    HAS_SLIMEVOLLEY = False

# Try to create environment
try:
    test_env = gym.make('SlimeVolley-v0')
    test_env.close()
    print("\n✓ SlimeVolley-v0 environment is available")
except Exception as e:
    print(f"\n✗ ERROR: Cannot create SlimeVolley-v0 environment: {e}")
    print("Install with: pip install slimevolleygym")
    exit(1)

print("\nTesting completely random policy...")
print("This tells us the worst-case baseline performance")

# Test random policy
num_episodes = 20
rewards = []

for episode in range(num_episodes):
    env = gym.make('SlimeVolley-v0')
    obs = env.reset()
    total_reward = 0
    done = False
    steps = 0
    max_steps = 3000
    
    while not done and steps < max_steps:
        # Completely random action
        random_action = np.random.randint(0, 2, size=3)  # [forward, backward, jump]
        obs, reward, done, info = env.step(random_action)
        total_reward += reward
        steps += 1
    
    rewards.append(total_reward)
    env.close()
    if (episode + 1) % 5 == 0:
        print(f"  Episode {episode+1}/20: reward = {total_reward:.2f}")

rewards = np.array(rewards)

print("\n## Results")
print("-" * 80)
print(f"Random policy over {num_episodes} games:")
print(f"  Mean reward: {np.mean(rewards):.2f}")
print(f"  Std reward:  {np.std(rewards):.2f}")
print(f"  Min reward:  {np.min(rewards):.2f}")
print(f"  Max reward:  {np.max(rewards):.2f}")
print()
print(f"NEAT best after 1024 gens: -3.80")
print(f"Random baseline:           {np.mean(rewards):.2f}")
print()
if np.mean(rewards) < -4.5:
    print("✓ Random is much worse than NEAT")
    print("  This means NEAT learned SOMETHING, but not enough")
elif np.mean(rewards) < -3.5:
    print("⚠ Random is similar to NEAT")
    print("  This means NEAT barely learned anything!")
else:
    print("✗ Random is better than NEAT")
    print("  This means something is VERY WRONG")

print("\n## Analysis")
print("-" * 80)
improvement = np.mean(rewards) - (-3.80)
if improvement < -0.5:
    print(f"NEAT improved by {abs(improvement):.2f} over random")
    print("This is a small improvement over 1024 generations")
    print()
    print("Conclusion: Task is extremely difficult")
    print("The built-in opponent is probably very strong")
    print("Evolution from random is nearly impossible")
else:
    print(f"NEAT is only {abs(improvement):.2f} better than random")
    print("This is almost no learning!")
    print()
    print("Conclusion: Either the task is impossibly hard,")
    print("or there's still a bug in the implementation")

print("\n## Recommendation")
print("-" * 80)
if np.mean(rewards) < -4.0:
    print("The task is confirmed to be VERY HARD.")
    print()
    print("SlimeVolley with the built-in opponent is too difficult")
    print("for basic NEAT to learn from scratch.")
    print()
    print("Options:")
    print("  1. Accept this is a hard task (most likely)")
    print("  2. Try self-play instead of fixed opponent")
    print("  3. Use curriculum learning (easier opponents first)")
    print("  4. Switch to RL algorithms with memory (PPO, SAC)")
    print("  5. Dramatically increase network size and training time")
else:
    print("Something may still be wrong with the implementation.")
    print("Consider debugging further or checking if other tasks work.")

print("=" * 80)
