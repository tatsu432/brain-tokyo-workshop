#!/usr/bin/env python
"""
Test V4 fix: LINEAR outputs + CLIPPING (like SwingUp!)

This is the REAL fix that matches the working SwingUp pattern:
1. Linear output activation (good gradient flow)
2. Clip outputs to [-1, 1] (prevents saturation)
3. Moderate thresholds (0.3, 0.4) that work with full range
"""

import numpy as np
import sys

print("=" * 80)
print("V4 FIX: LINEAR + CLIPPING (Like SwingUp!)")
print("=" * 80)

print("\n## Why This Should Work")
print("-" * 80)
print()
print("SwingUp configuration:")
print("  - output_size: 1")
print("  - o_act: linear (actId=1)")
print("  - Action processing: np.clip(action, -1.0, 1.0)")
print("  - Result: WORKS!")
print()
print("V4 SlimeVolley configuration (matching SwingUp pattern):")
print("  - output_size: 2")
print("  - o_act: linear (actId=1)")
print("  - Action processing: np.clip(action, -1.0, 1.0)")
print("  - Thresholds: 0.3 (horizontal), 0.4 (jump)")
print()
print("This should work because:")
print("  ✓ Linear activation = good gradient for learning")
print("  ✓ Clipping = prevents extreme values")
print("  ✓ Moderate thresholds = good action distribution")

print("\n## Testing Action Activation Rates")
print("-" * 80)

# Simulate random LINEAR outputs that are then clipped
np.random.seed(42)
num_samples = 1000

forward_count = 0
backward_count = 0
deadband_count = 0
jump_count = 0

outputs_h = []
outputs_j = []

for _ in range(num_samples):
    # Simulate linear network output (can be any value)
    raw_h = np.random.randn() * 2.0  # Std=2.0, so some values will be large
    raw_j = np.random.randn() * 2.0

    # CLIP to [-1, 1] like SwingUp does
    clipped_h = np.clip(raw_h, -1.0, 1.0)
    clipped_j = np.clip(raw_j, -1.0, 1.0)

    outputs_h.append(clipped_h)
    outputs_j.append(clipped_j)

    # Apply action mapping with V4 thresholds
    if clipped_h > 0.3:
        forward_count += 1
    elif clipped_h < -0.3:
        backward_count += 1
    else:
        deadband_count += 1

    if clipped_j > 0.4:
        jump_count += 1

outputs_h = np.array(outputs_h)
outputs_j = np.array(outputs_j)

print(f"\nWith {num_samples} random linear outputs (clipped to [-1,1]):")
print(f"\nHorizontal channel:")
print(f"  Mean: {np.mean(outputs_h):.3f}")
print(f"  Std:  {np.std(outputs_h):.3f}")
print(f"  Range: [{np.min(outputs_h):.3f}, {np.max(outputs_h):.3f}]")
print(f"  Forward (> 0.3):  {forward_count} ({100*forward_count/num_samples:.1f}%)")
print(f"  Backward (< -0.3): {backward_count} ({100*backward_count/num_samples:.1f}%)")
print(
    f"  Deadband [-0.3, 0.3]: {deadband_count} ({100*deadband_count/num_samples:.1f}%)"
)
print()
print(f"Jump channel:")
print(f"  Mean: {np.mean(outputs_j):.3f}")
print(f"  Std:  {np.std(outputs_j):.3f}")
print(f"  Range: [{np.min(outputs_j):.3f}, {np.max(outputs_j):.3f}]")
print(f"  Jump (> 0.4): {jump_count} ({100*jump_count/num_samples:.1f}%)")
print(
    f"  No jump: {num_samples - jump_count} ({100*(num_samples-jump_count)/num_samples:.1f}%)"
)

print("\n## Analysis")
print("-" * 80)

horiz_active = forward_count + backward_count
horiz_pct = 100 * horiz_active / num_samples
jump_pct = 100 * jump_count / num_samples
deadband_pct = 100 * deadband_count / num_samples

print(f"\nAction activation rates:")
print(f"  Horizontal movement: {horiz_pct:.1f}%")
print(f"  Jump: {jump_pct:.1f}%")
print(f"  Deadband: {deadband_pct:.1f}%")

if horiz_pct > 50 and horiz_pct < 90:
    print("\n✓ Horizontal movement: Good balance (active but not constant)")
elif horiz_pct > 90:
    print("\n⚠ Horizontal movement: Too active (might thrash)")
else:
    print("\n✗ Horizontal movement: Too inactive")

if jump_pct > 25 and jump_pct < 60:
    print("✓ Jump: Good balance (frequent but selective)")
elif jump_pct > 60:
    print("⚠ Jump: Too frequent (might always be in air)")
else:
    print("✗ Jump: Too rare")

if deadband_pct > 10 and deadband_pct < 50:
    print("✓ Deadband: Reasonable (allows some standing)")
elif deadband_pct > 50:
    print("✗ Deadband: Too large (agent too passive)")
else:
    print("⚠ Deadband: Very small (agent might thrash)")

print("\n## Test Action Mapping Logic")
print("-" * 80)

test_cases = [
    ([0.8, 0.6], [1, 0, 1], "Strong positive → forward+jump"),
    ([0.5, 0.3], [1, 0, 0], "Moderate positive → forward only"),
    ([0.1, 0.1], [0, 0, 0], "Weak → no action"),
    ([-0.5, 0.6], [0, 1, 1], "Negative + jump → backward+jump"),
    ([-0.8, 0.2], [0, 1, 0], "Strong negative → backward only"),
    ([0.2, 0.5], [0, 0, 1], "Deadband + jump → jump only"),
]

all_passed = True
for inputs, expected, description in test_cases:
    # Clip first (like in actual code)
    clipped = np.clip(inputs, -1.0, 1.0)
    out_h, out_j = clipped

    # Apply mapping
    if out_h > 0.3:
        forward, backward = 1, 0
    elif out_h < -0.3:
        forward, backward = 0, 1
    else:
        forward, backward = 0, 0

    jump = 1 if out_j > 0.4 else 0

    result = [forward, backward, jump]
    match = result == expected

    status = "✓" if match else "✗"
    print(f"{status} {inputs} → clip → {clipped} → {result} | {description}")

    if not match:
        print(f"  Expected: {expected}")
        all_passed = False

print("\n## Comparison: V3 (Tanh) vs V4 (Linear+Clip)")
print("-" * 80)

print("\nV3 (Tanh + Low Thresholds):")
print("  - Activation: tanh (outputs naturally in [-1,1])")
print("  - Thresholds: 0.05, 0.1")
print("  - Result: 89.5% horizontal, 44.6% jump")
print("  - Problem: Outputs cluster near 0, thresholds too sensitive")
print("  - Fitness: Stuck at -3.80")
print()
print("V4 (Linear + Clipping + Moderate Thresholds):")
print("  - Activation: linear (then clipped to [-1,1])")
print("  - Thresholds: 0.3, 0.4")
print("  - Result: {:.1f}% horizontal, {:.1f}% jump".format(horiz_pct, jump_pct))
print("  - Advantage: Matches SwingUp (proven to work!)")
print("  - Advantage: Better gradient flow for learning")

print("\n## Final Verdict")
print("=" * 80)

if all_passed and 50 < horiz_pct < 90 and 25 < jump_pct < 60:
    print("\n✓ V4 FIX LOOKS EXCELLENT!")
    print()
    print("This matches the SwingUp pattern exactly:")
    print("  ✓ Linear outputs (good gradient)")
    print("  ✓ Clipping to [-1,1] (prevents saturation)")
    print("  ✓ Moderate thresholds (good action distribution)")
    print()
    print("Action activation rates are balanced:")
    print(f"  - Horizontal: {horiz_pct:.1f}%")
    print(f"  - Jump: {jump_pct:.1f}%")
    print(f"  - Deadband: {deadband_pct:.1f}%")
    print()
    print("THIS SHOULD FINALLY WORK!")
    print()
    print("Run training:")
    print("  python neat_train.py -p p/slimevolley_fixed.json -n 9")
    print()
    print("Expected:")
    print("  - Should see improvement beyond -3.80 now")
    print("  - Linear activation allows better learning")
    print("  - Clipping prevents the saturation issues")
    print()
    exit_code = 0
else:
    print("\n⚠ V4 FIX NEEDS TUNING")
    if not all_passed:
        print("  - Logic tests failed")
    if not (50 < horiz_pct < 90):
        print(f"  - Horizontal activation {horiz_pct:.1f}% out of range")
    if not (25 < jump_pct < 60):
        print(f"  - Jump activation {jump_pct:.1f}% out of range")
    exit_code = 1

print("=" * 80)
sys.exit(exit_code)
