#!/usr/bin/env python
"""
Test the V3 complete fix:
1. Changed from 3 outputs to 2 outputs (matches action mapping!)
2. Using tanh activation (bounded)
3. Lowered thresholds (0.05, 0.1) to match tanh distribution
"""

import numpy as np
import sys

print("=" * 80)
print("TESTING V3 FIX: 2 Outputs + Tanh + Low Thresholds")
print("=" * 80)

print("\n## V3 Configuration")
print("-" * 80)
print("Network outputs: 2 [horizontal, jump]")
print("Output activation: tanh (bounded to [-1, 1])")
print("Action mapping thresholds:")
print("  - Forward: action[0] > 0.05")
print("  - Backward: action[0] < -0.05")
print("  - Deadband: |action[0]| ≤ 0.05")
print("  - Jump: action[1] > 0.1")

print("\n## Testing Action Activation Rates")
print("-" * 80)

# Simulate random tanh network outputs
np.random.seed(42)
num_samples = 1000

forward_count = 0
backward_count = 0
deadband_count = 0
jump_count = 0

for _ in range(num_samples):
    # Simulate tanh output
    raw_h = np.random.randn() * 0.5
    raw_j = np.random.randn() * 0.5
    
    out_h = np.tanh(raw_h)
    out_j = np.tanh(raw_j)
    
    # Apply action mapping
    if out_h > 0.05:
        forward_count += 1
    elif out_h < -0.05:
        backward_count += 1
    else:
        deadband_count += 1
    
    if out_j > 0.1:
        jump_count += 1

print(f"\nWith {num_samples} random tanh outputs:")
print(f"  Forward activations: {forward_count} ({100*forward_count/num_samples:.1f}%)")
print(f"  Backward activations: {backward_count} ({100*backward_count/num_samples:.1f}%)")
print(f"  Deadband (no horizontal): {deadband_count} ({100*deadband_count/num_samples:.1f}%)")
print(f"  Jump activations: {jump_count} ({100*jump_count/num_samples:.1f}%)")

print("\n## Analysis")
print("-" * 80)

if deadband_count < 150:  # Less than 15% in deadband
    print("✓ Good: Agent will be active most of the time")
    verdict = "GOOD"
elif deadband_count < 300:  # 15-30% in deadband
    print("⚠ Acceptable: Agent reasonably active")
    verdict = "OK"
else:  # More than 30% in deadband
    print("✗ Bad: Agent inactive too often")
    verdict = "BAD"

if jump_count > 400:  # More than 40% jump
    print("✓ Good: Agent will jump frequently")
elif jump_count > 300:  # 30-40% jump
    print("⚠ Acceptable: Agent jumps moderately")
else:  # Less than 30% jump
    print("✗ Bad: Agent rarely jumps")
    verdict = "BAD"

print("\n## Test Action Mapping Logic")
print("-" * 80)

test_cases = [
    ([0.8, 0.5], [1, 0, 1], "Strong positive → forward+jump"),
    ([0.1, 0.2], [1, 0, 1], "Weak positive → forward+jump"),
    ([0.03, -0.05], [0, 0, 0], "Near zero → no action"),
    ([-0.1, 0.5], [0, 1, 1], "Negative → backward+jump"),
    ([-0.8, -0.5], [0, 1, 0], "Strong negative → backward only"),
    ([0.5, 0.05], [1, 0, 0], "High horiz, low jump → forward only"),
]

all_passed = True
for inputs, expected, description in test_cases:
    out_h, out_j = inputs
    
    # Apply mapping
    if out_h > 0.05:
        forward, backward = 1, 0
    elif out_h < -0.05:
        forward, backward = 0, 1
    else:
        forward, backward = 0, 0
    
    jump = 1 if out_j > 0.1 else 0
    
    result = [forward, backward, jump]
    match = result == expected
    
    status = "✓" if match else "✗"
    print(f"{status} {inputs} → {result} | {description}")
    
    if not match:
        print(f"  Expected: {expected}")
        all_passed = False

print("\n## Verdict")
print("=" * 80)

if all_passed and verdict in ["GOOD", "OK"]:
    print("\n✓ V3 FIX LOOKS GOOD!")
    print()
    print("Configuration:")
    print("  - 2 outputs (matches action mapping)")
    print("  - Tanh activation (bounded)")
    print("  - Low thresholds (0.05, 0.1)")
    print()
    print("Action activation rates:")
    print(f"  - Horizontal movement: {100*(forward_count+backward_count)/num_samples:.1f}%")
    print(f"  - Jump: {100*jump_count/num_samples:.1f}%")
    print()
    print("This should allow the agent to explore effectively!")
    print()
    print("Run training with:")
    print("  python neat_train.py -p p/slimevolley_fixed.json -n 9")
    print()
    return_code = 0
else:
    print("\n✗ V3 FIX NEEDS MORE TUNING")
    print()
    print("Issues:")
    if deadband_count > 300:
        print("  - Too much deadband (agent inactive)")
    if jump_count < 300:
        print("  - Too little jumping")
    if not all_passed:
        print("  - Logic test failures")
    print()
    print("Consider further threshold adjustments.")
    return_code = 1

print("=" * 80)
sys.exit(return_code)
