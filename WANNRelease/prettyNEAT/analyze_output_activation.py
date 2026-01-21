#!/usr/bin/env python
"""
Analyze the output activation function issue.

This is likely the REAL root cause of the problem.
"""

import numpy as np

print("=" * 80)
print("OUTPUT ACTIVATION FUNCTION ANALYSIS")
print("=" * 80)

print("\n## Current Configuration")
print("-" * 80)
print("From domain/config.py:")
print("  o_act = np.full(3, 1)  # All output nodes use activation #1 = LINEAR")
print()
print("Linear activation: f(x) = x")
print("  Output range: [-∞, +∞] (UNBOUNDED!)")
print()

print("\n## The Problem with Linear Outputs")
print("-" * 80)
print()
print("With unbounded outputs, the network can produce values like:")
print("  Network output: [50.0, 30.0, 20.0]")
print()
print("Your action mapping checks:")
print("  if action[0] > 0.2:   forward = 1    ← 50.0 > 0.2 ✓")
print("  if action[1] > 0.3:   jump = 1       ← 30.0 > 0.3 ✓")
print()
print("But wait! With the current code:")
print("  if action[0] > 0.2:   forward = 1, backward = 0")
print("  elif action[0] < -0.2: forward = 0, backward = 1")
print()
print("So action[0]=50.0 → forward only (no backward). That's actually OK!")
print()
print("Unless... let me check the activation functions being used...")
print()

print("\n## Testing Different Activation Scenarios")
print("-" * 80)

# Simulate what NEAT might output with different activations
scenarios = [
    ("All Linear (current)", 1, [-10, 0, 10], "Unbounded, but action mapping handles it?"),
    ("All Tanh", 5, [-10, 0, 10], "Bounded to [-1, 1], action mapping works well"),
    ("All Sigmoid", 6, [-10, 0, 10], "Bounded to [0, 1], need different thresholds"),
    ("All ReLU", 9, [-10, 0, 10], "Bounded to [0, ∞], problematic for negative actions"),
]

def apply_act(act_id, x):
    """Simplified version of _applyAct"""
    if act_id == 1:  # Linear
        return x
    elif act_id == 5:  # Tanh
        return np.tanh(x)
    elif act_id == 6:  # Sigmoid
        return (np.tanh(x/2.0) + 1.0) / 2.0
    elif act_id == 9:  # ReLU
        return np.maximum(0, x)
    else:
        return x

for scenario_name, act_id, test_inputs, note in scenarios:
    outputs = [apply_act(act_id, x) for x in test_inputs]
    print(f"\n{scenario_name}:")
    print(f"  Raw network: {test_inputs}")
    print(f"  After activation: {[f'{x:.2f}' for x in outputs]}")
    print(f"  Range: [{min(outputs):.2f}, {max(outputs):.2f}]")
    print(f"  Note: {note}")

print("\n\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print()
print("The o_act=1 (linear) is actually used by SwingUp too, so it's not")
print("inherently wrong. BUT there might be subtle issues:")
print()
print("1. With alg_act=0 (all activations for hidden nodes), the")
print("   network topology might create very large or small values")
print()
print("2. The output nodes with linear activation don't bound these values")
print()
print("3. BUT... your action mapping should still work because it uses")
print("   if/elif with thresholds, not both actions simultaneously")
print()
print("WAIT - Let me check if there's an issue with how the network")
print("output is being passed to _process_action()...")
print()
print("=" * 80)
print()
print("Actually, I think the REAL issue might be different:")
print()
print("1. Reward shaping is definitely broken (fitness is positive)")
print("2. Need to test NON-shaped version to see if action mapping works")
print("3. If non-shaped version also stuck, might be network initialization")
print()
print("Did you test p/slimevolley_fixed.json (WITHOUT shaped)?")
print("Please share those results!")
print()
print("=" * 80)
