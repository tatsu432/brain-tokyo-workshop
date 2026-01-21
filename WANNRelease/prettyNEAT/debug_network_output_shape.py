#!/usr/bin/env python
"""
CRITICAL CHECK: Is the network actually outputting 2 values as expected?

When I changed output_size from 3 to 2, the network structure changed.
But does domain/slimevolley.py's _process_action receive 2 values?

If the network still outputs 3 values somehow, my action mapping would fail!
"""

import numpy as np
import sys
import os

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neat_src.ann import act, _applyAct
from domain import config

print("=" * 80)
print("CRITICAL CHECK: Network Output Shape")
print("=" * 80)

# Load slimevolley config
games = config.games
slime_config = games['slimevolley']

print("\n## Configuration")
print("-" * 80)
print(f"Input size:  {slime_config.input_size}")
print(f"Output size: {slime_config.output_size}")  # Should be 2
print(f"Layers:      {slime_config.layers}")  # Should be [8, 8]
print(f"O_act:       {slime_config.o_act}")  # Should be [5, 5] (tanh, tanh)

# Calculate total nodes
n_input = slime_config.input_size
n_output = slime_config.output_size
hidden = slime_config.layers
total_nodes = n_input + sum(hidden) + n_output
print(f"\nTotal nodes: {total_nodes}")
print(f"  Inputs:  {n_input}")
print(f"  Hidden:  {sum(hidden)} ({hidden})")
print(f"  Outputs: {n_output}")

# Create a dummy network
print("\n## Testing Network Forward Pass")
print("-" * 80)

# Create weight matrix (all zeros initially, then add some connections)
wMat = np.zeros((total_nodes, total_nodes))

# Add some dummy connections
# Input 0 → Hidden layer 1, node 0
wMat[n_input, 0] = 0.5  
# Hidden layer 1, node 0 → Output 0
wMat[n_input + hidden[0], n_input] = 1.0
# Hidden layer 1, node 1 → Output 1  
wMat[n_input + hidden[0] + 1, n_input + 1] = 1.0

# Create activation vector
aVec = np.ones(total_nodes, dtype=int)  # All linear initially
# Set hidden layer activations
for i in range(n_input, n_input + sum(hidden)):
    aVec[i] = 5  # tanh for hidden
# Set output activations
for i in range(n_input + sum(hidden), total_nodes):
    aVec[i] = 5  # tanh for outputs

# Test forward pass
dummy_input = np.random.randn(12)  # 12-dim observation
print(f"Input shape: {dummy_input.shape}")
print(f"Input: {dummy_input}")

# Run network
wVec = wMat.flatten()
output = act(wVec, aVec, n_input, n_output, dummy_input)

print(f"\nOutput shape: {output.shape}")
print(f"Output: {output}")

print("\n## Critical Check")
print("-" * 80)
if output.shape[0] == 2 or (len(output.shape) == 2 and output.shape[1] == 2):
    print("✓ Network outputs 2 values as expected!")
    print()
    print("This means the network structure is correct.")
    print("The problem must be elsewhere (task difficulty, NEAT config, etc.)")
elif output.shape[0] == 3 or (len(output.shape) == 2 and output.shape[1] == 3):
    print("✗ BUG FOUND: Network outputs 3 values!")
    print()
    print("The config says output_size=2, but the network still outputs 3!")
    print("This would cause _process_action to use wrong values.")
    print()
    print("Need to investigate why output_size change didn't take effect.")
    sys.exit(1)
else:
    print(f"? Unexpected output shape: {output.shape}")
    print("Need to investigate network structure")
    sys.exit(1)

print("\n## If Output Shape is Correct, Then What's Wrong?")
print("-" * 80)
print()
print("If network outputs 2 values correctly, the issue is NOT a bug.")
print("The issue is that SlimeVolley is just extremely difficult for NEAT.")
print()
print("Evidence:")
print("  - Fitness stuck at -3.80 for 700+ generations")
print("  - Only 0.80 improvement over 1024 gens")
print("  - Agent can move but can't learn to win")
print()
print("Root cause: Task difficulty vs algorithm capability mismatch")
print()
print("Solutions:")
print("  1. Use self-play (agent vs itself, co-evolution)")
print("  2. Use curriculum (start vs weak opponent, gradually increase)")
print("  3. Add memory (recurrent connections, LSTM)")
print("  4. Dramatically increase network size ([50,50] instead of [8,8])")
print("  5. Use modern RL (PPO, SAC) instead of NEAT")
print("  6. Accept that this task is beyond basic NEAT")
print()
print("=" * 80)
