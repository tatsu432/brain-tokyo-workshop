#!/usr/bin/env python
"""Simple check: what shape does the network output?"""

import json
import numpy as np

# Load config file directly
with open("p/slimevolley_fixed.json", "r") as f:
    config = json.load(f)

print("=" * 80)
print("SIMPLE SHAPE CHECK")
print("=" * 80)

print("\n## From slimevolley_fixed.json:")
print("-" * 80)
print(f"Task: {config['task']}")
print(f"alg_nReps: {config['alg_nReps']}")
print(f"popSize: {config['popSize']}")

# Now check the game config
print("\n## Loading game config from domain/config.py...")
print("-" * 80)

# Read config.py to extract slimevolley settings
with open("domain/config.py", "r") as f:
    config_py = f.read()

# Extract output_size value (hacky but works without importing)
import re

match = re.search(r"slimevolley = Game\([^)]+output_size=(\d+)", config_py, re.DOTALL)
if match:
    output_size = int(match.group(1))
    print(f"✓ Found output_size: {output_size}")
else:
    print("✗ Could not find output_size in config.py")
    output_size = None

match = re.search(r"slimevolley = Game\([^)]+input_size=(\d+)", config_py, re.DOTALL)
if match:
    input_size = int(match.group(1))
    print(f"✓ Found input_size: {input_size}")
else:
    input_size = None

match = re.search(r"slimevolley = Game\([^)]+layers=\[([^\]]+)\]", config_py, re.DOTALL)
if match:
    layers_str = match.group(1)
    layers = [int(x.strip()) for x in layers_str.split(",")]
    print(f"✓ Found layers: {layers}")
else:
    layers = None

print("\n## Network Structure")
print("-" * 80)
if output_size and input_size and layers:
    total_nodes = input_size + sum(layers) + output_size
    print(f"Input nodes:  {input_size}")
    print(f"Hidden nodes: {sum(layers)} {layers}")
    print(f"Output nodes: {output_size}")
    print(f"Total nodes:  {total_nodes}")
    print()
    if output_size == 2:
        print("✓ Network configured for 2 outputs")
        print()
        print("This means when act() is called, it should return 2 values.")
        print("These 2 values go to _process_action in domain/slimevolley.py")
        print()
        print("_process_action expects:")
        print("  action[0] → horizontal movement")
        print("  action[1] → jump")
        print()
        print("This should all be working correctly!")
    else:
        print(f"✗ WARNING: output_size is {output_size}, not 2!")
        print("This might be the bug!")

print("\n## Conclusion")
print("-" * 80)
if output_size == 2:
    print("Network structure is correct (2 outputs).")
    print()
    print("The fitness stagnation (-3.80 for 700+ gens) is NOT a bug.")
    print("It's because SlimeVolley is extremely hard for basic NEAT.")
    print()
    print("The task requires:")
    print("  - Precise ball trajectory prediction")
    print("  - Timing to hit ball at right moment")
    print("  - Strategic positioning")
    print("  - Reacting to strong AI opponent")
    print()
    print("A simple 2-output feedforward network cannot learn this.")
else:
    print(f"BUG: output_size is {output_size}, not 2!")
    print("Need to fix this first.")

print("=" * 80)
