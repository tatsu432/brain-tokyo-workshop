#!/usr/bin/env python
"""
Diagnose what's actually happening with network outputs and actions.

The tanh fix made things WORSE (-4.20 vs -3.33), which suggests:
1. Network outputs with tanh are too small (near 0)
2. Agent stuck in deadband, not taking actions
3. Agent just stands still and loses badly
"""

import numpy as np
import sys

print("=" * 80)
print("DIAGNOSING WHY TANH MADE THINGS WORSE")
print("=" * 80)

print("\n## Problem: Fitness got WORSE")
print("-" * 80)
print("Before fixes: -3.33 (losing ~3.3 lives per game)")
print("With shaped rewards: +5.17 (reward hacking)")
print("With tanh output: -4.60 to -4.20 (losing ~4.2 to 4.6 lives per game)")
print()
print("The tanh fix made performance WORSE, not better!")

print("\n## Hypothesis: Tanh outputs are too small")
print("-" * 80)
print()
print("With random weights and tanh activation:")
print("- Initial network weights: typically small random values")
print("- Raw network output: small values (e.g., -0.5 to +0.5)")
print("- After tanh: even smaller (tanh(0.5) = 0.46)")
print()
print("Action mapping thresholds:")
print("  forward: output[0] > 0.2")
print("  backward: output[0] < -0.2")
print("  jump: output[1] > 0.3")
print()
print("If network outputs are ~0 to 0.15:")
print("  → All in deadband!")
print("  → No actions taken")
print("  → Agent stands still")
print("  → Loses all 5 lives = -5.0 fitness")
print()
print("Observed: -4.60, which is close to -5.0!")
print("This confirms agent is barely doing anything.")

print("\n## Testing: What do random tanh networks output?")
print("-" * 80)

# Simulate random network with tanh output
np.random.seed(42)
num_samples = 1000

outputs_forward = []
outputs_jump = []

for _ in range(num_samples):
    # Simulate random network calculation
    # Typical: weighted sum of ~10-20 values with small random weights
    raw_output_forward = np.random.randn() * 0.5  # Small random value
    raw_output_jump = np.random.randn() * 0.5

    # Apply tanh
    output_forward = np.tanh(raw_output_forward)
    output_jump = np.tanh(raw_output_jump)

    outputs_forward.append(output_forward)
    outputs_jump.append(output_jump)

outputs_forward = np.array(outputs_forward)
outputs_jump = np.array(outputs_jump)

print(f"\nRandom tanh network outputs (n={num_samples}):")
print(f"  Forward/backward channel:")
print(f"    Mean: {np.mean(outputs_forward):.3f}")
print(f"    Std: {np.std(outputs_forward):.3f}")
print(f"    Range: [{np.min(outputs_forward):.3f}, {np.max(outputs_forward):.3f}]")
print(f"    % > 0.2 (forward): {100 * np.mean(outputs_forward > 0.2):.1f}%")
print(f"    % < -0.2 (backward): {100 * np.mean(outputs_forward < -0.2):.1f}%")
print(
    f"    % in deadband [-0.2, 0.2]: {100 * np.mean(np.abs(outputs_forward) <= 0.2):.1f}%"
)
print()
print(f"  Jump channel:")
print(f"    Mean: {np.mean(outputs_jump):.3f}")
print(f"    Std: {np.std(outputs_jump):.3f}")
print(f"    Range: [{np.min(outputs_jump):.3f}, {np.max(outputs_jump):.3f}]")
print(f"    % > 0.3 (jump): {100 * np.mean(outputs_jump > 0.3):.1f}%")
print(f"    % no jump: {100 * np.mean(outputs_jump <= 0.3):.1f}%")

print("\n## Conclusion")
print("-" * 80)
print()
print("If most outputs are in the deadband, the agent rarely acts!")
print("This explains why fitness is -4.60 (almost worst case -5.0).")
print()
print("The problem: Thresholds (0.2, 0.3) are too HIGH for tanh outputs.")
print()

print("\n## Solution Options")
print("-" * 80)
print()
print("Option 1: Lower the thresholds")
print("  Change to: forward/backward at ±0.1, jump at 0.15")
print("  Pro: Allows tanh outputs to trigger actions")
print("  Con: May be too sensitive to noise")
print()
print("Option 2: Use sigmoid activation instead of tanh")
print("  Sigmoid outputs [0, 1] instead of [-1, 1]")
print("  Use thresholds: forward > 0.6, backward < 0.4, jump > 0.5")
print("  Pro: Asymmetric, may help with one-sided actions")
print("  Con: Less intuitive for forward/backward")
print()
print("Option 3: Increase weight initialization scale")
print("  Make initial weights larger")
print("  Pro: Pushes outputs away from zero")
print("  Con: May cause instability")
print()
print("Option 4: Go back to linear output activation BUT...")
print("  Keep it BUT add output clipping in action mapping")
print("  Clip outputs to [-2, 2] range before thresholding")
print("  Pro: Prevents extreme saturation, keeps exploration")
print("  Con: Arbitrary clipping range")
print()
print("RECOMMENDATION: Try Option 1 first (lower thresholds)")
print("  If that doesn't work, try Option 4 (linear + clipping)")
print()
print("=" * 80)
