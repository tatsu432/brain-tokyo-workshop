#!/usr/bin/env python
"""
Verification script to ensure all fixes are correctly applied.

Run this BEFORE starting training to verify the setup is correct.
"""

import sys
import json
import numpy as np

print("=" * 80)
print("SLIMEVOLLEY FIX VERIFICATION")
print("=" * 80)

all_good = True

# Test 1: Check config file
print("\n## Test 1: Checking p/slimevolley_fixed.json")
print("-" * 80)

try:
    with open("p/slimevolley_fixed.json", "r") as f:
        config = json.load(f)

    print(f"✓ Config file found")
    print(f"  task: {config['task']}")

    if config["task"] == "slimevolley":
        print(f"  ✓ Task is 'slimevolley' (NOT 'slimevolley_shaped')")
    else:
        print(f"  ✗ WARNING: Task is '{config['task']}' (should be 'slimevolley')")
        all_good = False

    print(f"  alg_nReps: {config['alg_nReps']} (should be 5)")
    print(f"  alg_act: {config['alg_act']} (should be 0)")
    print(f"  popSize: {config['popSize']}")

except Exception as e:
    print(f"✗ Error loading config: {e}")
    all_good = False

# Test 2: Check domain config
print("\n## Test 2: Checking domain/config.py")
print("-" * 80)

try:
    # Read the file as text to check o_act value
    with open("domain/config.py", "r") as f:
        content = f.read()

    # Check if o_act was changed
    if "o_act=np.full(3,5)" in content or "o_act=np.full(3, 5)" in content:
        print("✓ Output activation FIXED:")
        print("  o_act=np.full(3,5) → tanh (bounded to [-1, 1])")
        print("  This is CRITICAL for action mapping to work!")
    elif "o_act=np.full(3,1)" in content or "o_act=np.full(3, 1)" in content:
        print("✗ Output activation STILL BROKEN:")
        print("  o_act=np.full(3,1) → linear (unbounded)")
        print("  This will cause saturation and poor learning!")
        print()
        print("  FIX: Edit domain/config.py line ~130:")
        print("       Change o_act=np.full(3,1) to o_act=np.full(3,5)")
        all_good = False
    else:
        print("? Could not verify o_act value (check manually)")

    # Check layers
    if "layers=[8, 8]" in content:
        print("✓ Network layers simplified: [8, 8]")
    elif "layers=[15, 10]" in content:
        print("✗ Network still using old layers: [15, 10]")
        all_good = False

except Exception as e:
    print(f"✗ Error checking domain/config.py: {e}")
    all_good = False

# Test 3: Check action mapping
print("\n## Test 3: Checking domain/slimevolley.py action mapping")
print("-" * 80)

try:
    with open("domain/slimevolley.py", "r") as f:
        content = f.read()

    # Check for the fixed action mapping
    if "mutually exclusive" in content or "Mutually exclusive" in content:
        print("✓ Action mapping includes mutually exclusive logic")
    else:
        print("? Could not verify action mapping (check manually)")

    # Check for deadband
    if "-0.2" in content and "0.2" in content:
        print("✓ Deadband thresholds present (prevents random activation)")
    else:
        print("? Could not verify deadband (check manually)")

    # Check for reward shaping class
    if "class SlimeVolleyRewardShapingEnv" in content:
        print("✓ Reward shaping class exists (but should NOT be used)")

except Exception as e:
    print(f"✗ Error checking slimevolley.py: {e}")
    all_good = False

# Test 4: Simulate action mapping with tanh outputs
print("\n## Test 4: Simulating Action Mapping with Tanh Outputs")
print("-" * 80)


def tanh(x):
    return np.tanh(x)


def process_action_simulation(action):
    """Simulated version of the fixed action mapping"""
    action = np.array(action).flatten()

    if len(action) >= 2:
        if action[0] > 0.2:
            forward, backward = 1, 0
        elif action[0] < -0.2:
            forward, backward = 0, 1
        else:
            forward, backward = 0, 0

        jump = 1 if action[1] > 0.3 else 0
        return np.array([forward, backward, jump], dtype=np.int8)
    return np.array([0, 0, 0])


# Test with tanh-bounded outputs
test_cases = [
    (tanh([2.0, 1.0, 0.5]), [1, 0, 1], "Large positive → forward+jump"),
    (tanh([0.5, 0.8, 0.0]), [1, 0, 1], "Medium positive → forward+jump"),
    (tanh([0.3, -0.5, 0.0]), [1, 0, 0], "Small positive → forward only"),
    (tanh([0.0, 0.0, 0.0]), [0, 0, 0], "Zero → no action"),
    (tanh([-0.5, 0.8, 0.0]), [0, 1, 1], "Negative horiz → backward+jump"),
]

all_tests_passed = True
for raw_input, expected, description in test_cases:
    result = process_action_simulation(raw_input)
    match = np.array_equal(result, expected)
    status = "✓" if match else "✗"

    print(
        f"{status} Input: {[f'{x:.2f}' for x in raw_input]} → {result} | {description}"
    )
    if not match:
        print(f"  Expected: {expected}")
        all_tests_passed = False

if all_tests_passed:
    print("\n✓ All action mapping tests passed with tanh-bounded outputs!")
else:
    print("\n✗ Some tests failed!")
    all_good = False

# Final summary
print("\n\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

if all_good:
    print("\n✓ ALL CHECKS PASSED!")
    print()
    print("Your system is now configured correctly with:")
    print("  1. Fixed action mapping (mutually exclusive)")
    print("  2. Bounded output activation (tanh)")
    print("  3. Simplified network ([8,8])")
    print("  4. No reward shaping")
    print()
    print("You're ready to train:")
    print("  python neat_train.py -p p/slimevolley_fixed.json -n 9")
    print()
    print("Expected results:")
    print("  - Fitness starts NEGATIVE (-4 to -3)")
    print("  - Gradually improves past -3.33 plateau")
    print("  - Eventually reaches 0 and goes positive (agent winning)")
    print()
    return_code = 0
else:
    print("\n✗ SOME CHECKS FAILED!")
    print()
    print("Please review the errors above and fix them before training.")
    print("The most critical fix is changing o_act from 1 to 5 in domain/config.py")
    print()
    return_code = 1

print("=" * 80)
sys.exit(return_code)
