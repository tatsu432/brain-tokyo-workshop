#!/usr/bin/env python
"""
CRITICAL CHECK: Is V4 actually being used?

Since V4 is still stuck at -3.80, we need to verify:
1. Is linear activation actually loaded?
2. Is clipping actually happening?
3. Are the new thresholds actually being used?
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domain import config

print("=" * 80)
print("VERIFY V4 IS ACTUALLY RUNNING")
print("=" * 80)

# Check config
games = config.games
slime = games['slimevolley']

print("\n## SlimeVolley Configuration")
print("-" * 80)
print(f"Output size: {slime.output_size}")
print(f"O_act: {slime.o_act}")
print()

if slime.output_size == 2:
    print("✓ Output size is 2 (correct)")
else:
    print(f"✗ Output size is {slime.output_size} (should be 2)")

if slime.o_act[0] == 1:
    print("✓ Output activation is LINEAR (1) - V4 is loaded!")
elif slime.o_act[0] == 5:
    print("✗ Output activation is TANH (5) - V3 is still loaded!")
    print()
    print("ERROR: V4 changes not applied!")
    print("Need to restart Python or reload config")
else:
    print(f"? Output activation is {slime.o_act[0]} (unexpected)")

print("\n## Check if clipping is in slimevolley.py")
print("-" * 80)

with open('domain/slimevolley.py', 'r') as f:
    code = f.read()

if 'np.clip(action, -1.0, 1.0)' in code:
    print("✓ Found np.clip in slimevolley.py")
else:
    print("✗ No np.clip found - V4 not applied!")

if 'action[0] > 0.3' in code:
    print("✓ Found threshold 0.3 (V4)")
elif 'action[0] > 0.05' in code:
    print("✗ Found threshold 0.05 (V3) - V4 not applied!")
elif 'action[0] > 0.2' in code:
    print("✗ Found threshold 0.2 (V2) - old version!")
else:
    print("? Could not find threshold")

print("\n## Conclusion")
print("=" * 80)

if slime.o_act[0] == 1 and 'np.clip(action, -1.0, 1.0)' in code and 'action[0] > 0.3' in code:
    print()
    print("✓ V4 IS CORRECTLY APPLIED")
    print()
    print("Since V4 is running but still stuck at -3.80,")
    print("this confirms the problem is NOT the activation function.")
    print()
    print("The problem is TASK DIFFICULTY.")
    print()
    print("SlimeVolley is too hard for basic feedforward NEAT to learn.")
    print()
    print("Recommendations:")
    print("  1. Dramatically increase network size: [30, 30] or [50, 50]")
    print("  2. Increase evaluation trials: 20 instead of 5")
    print("  3. Increase population: 256 or 512")
    print("  4. Accept that feedforward NEAT might not be suitable")
    print("  5. Try different approach: self-play, curriculum, or RL")
else:
    print()
    print("✗ V4 NOT APPLIED CORRECTLY")
    print()
    print("The config file has not been reloaded or changes were not saved.")
    print("This might be why it's still stuck at -3.80.")
    print()
    print("Try:")
    print("  1. Stop the current training")
    print("  2. Check that domain/config.py has o_act=np.full(2,1)")
    print("  3. Check that domain/slimevolley.py has np.clip")
    print("  4. Restart training")

print("=" * 80)
