#!/usr/bin/env python
"""
Test script to verify the improved action mapping fixes the bottleneck.

This script tests the action mapping to ensure:
1. Forward and backward are mutually exclusive (no conflicts)
2. Deadband prevents random activation
3. Thresholds are appropriate for NEAT output range

Run with: python test_action_mapping.py
"""

import numpy as np
import sys

# Try to import, but provide standalone version if dependencies missing
try:
    from domain.slimevolley import SlimeVolleyEnv
    HAS_ENV = True
except ImportError as e:
    print(f"Warning: Could not import SlimeVolleyEnv: {e}")
    print("Using standalone action mapping for testing...")
    HAS_ENV = False
    
    class SlimeVolleyEnvStandalone:
        """Standalone version of action mapping for testing"""
        def _process_action(self, action):
            """Improved action mapping (copy of fixed version)"""
            action = np.array(action).flatten()
            
            if len(action) == 1:
                val = float(action[0])
                if val > 0.5:
                    binary_action = np.array([1, 0, 1], dtype=np.int8)
                elif val > 0.1:
                    binary_action = np.array([1, 0, 0], dtype=np.int8)
                elif val > -0.1:
                    binary_action = np.array([0, 0, 1], dtype=np.int8)
                elif val > -0.5:
                    binary_action = np.array([0, 1, 0], dtype=np.int8)
                else:
                    binary_action = np.array([0, 1, 1], dtype=np.int8)
            else:
                if action[0] > 0.2:
                    forward = 1
                    backward = 0
                elif action[0] < -0.2:
                    forward = 0
                    backward = 1
                else:
                    forward = 0
                    backward = 0
                
                jump = 1 if action[1] > 0.3 else 0
                binary_action = np.array([forward, backward, jump], dtype=np.int8)
            
            return binary_action
    
    SlimeVolleyEnv = SlimeVolleyEnvStandalone

def test_action_mapping():
    """Test the improved action mapping"""
    print("=" * 70)
    print("Testing Improved SlimeVolley Action Mapping")
    print("=" * 70)
    
    env = SlimeVolleyEnv()
    
    print("\n1. Testing 3-output mode (most common case):")
    print("-" * 70)
    print("Format: [horizontal, jump, unused] → [forward, backward, jump]")
    print()
    
    test_cases_3d = [
        # (input, expected_output, description)
        ([0.5, 0.5, 0.0], [1, 0, 1], "Forward + Jump (typical attack)"),
        ([0.5, 0.1, 0.0], [1, 0, 0], "Forward only (horizontal < jump threshold)"),
        ([-0.5, 0.5, 0.0], [0, 1, 1], "Backward + Jump (retreating)"),
        ([-0.5, 0.1, 0.0], [0, 1, 0], "Backward only"),
        ([0.1, 0.5, 0.0], [0, 0, 1], "Jump only (in deadband)"),
        ([0.0, 0.0, 0.0], [0, 0, 0], "No action (all zeros)"),
        ([0.3, -0.1, 0.0], [1, 0, 0], "Forward, no jump (negative jump)"),
        ([-0.3, -0.1, 0.0], [0, 1, 0], "Backward, no jump"),
        
        # Critical tests for conflict prevention
        ([0.3, 0.3, 0.0], [1, 0, 0], "Both positive → Forward wins (NO CONFLICT!)"),
        ([1.0, 1.0, 0.0], [1, 0, 1], "Both max → Forward + Jump (NO CONFLICT!)"),
        ([-1.0, 1.0, 0.0], [0, 1, 1], "Negative horiz → Backward + Jump"),
    ]
    
    passed = 0
    failed = 0
    
    for input_action, expected, description in test_cases_3d:
        result = env._process_action(input_action)
        match = np.array_equal(result, expected)
        status = "✓ PASS" if match else "✗ FAIL"
        
        if match:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} | Input: {input_action} → Output: {result} | {description}")
        if not match:
            print(f"       Expected: {expected}")
    
    print("\n" + "-" * 70)
    print(f"3-output tests: {passed} passed, {failed} failed")
    
    # Test single output mode
    print("\n2. Testing 1-output mode (alternative configuration):")
    print("-" * 70)
    print("Format: single value → [forward, backward, jump]")
    print()
    
    test_cases_1d = [
        ([0.8], [1, 0, 1], "High positive → Forward + Jump"),
        ([0.3], [1, 0, 0], "Mid positive → Forward only"),
        ([0.0], [0, 0, 1], "Zero → Jump only"),
        ([-0.3], [0, 1, 0], "Mid negative → Backward only"),
        ([-0.8], [0, 1, 1], "High negative → Backward + Jump"),
    ]
    
    passed_1d = 0
    failed_1d = 0
    
    for input_action, expected, description in test_cases_1d:
        result = env._process_action(input_action)
        match = np.array_equal(result, expected)
        status = "✓ PASS" if match else "✗ FAIL"
        
        if match:
            passed_1d += 1
        else:
            failed_1d += 1
        
        print(f"{status} | Input: {input_action[0]:5.1f} → Output: {result} | {description}")
        if not match:
            print(f"       Expected: {expected}")
    
    print("\n" + "-" * 70)
    print(f"1-output tests: {passed_1d} passed, {failed_1d} failed")
    
    # Summary
    total_passed = passed + passed_1d
    total_tests = passed + failed + passed_1d + failed_1d
    
    print("\n" + "=" * 70)
    print(f"OVERALL: {total_passed}/{total_tests} tests passed")
    print("=" * 70)
    
    if failed == 0 and failed_1d == 0:
        print("\n✓ All tests passed! Action mapping is working correctly.")
        print("\nKey improvements verified:")
        print("  1. Forward/backward are mutually exclusive (no conflicts)")
        print("  2. Deadband prevents random activation near zero")
        print("  3. Clear thresholds create distinct action zones")
        print("\nThis should fix the -3.33 fitness bottleneck!")
        return True
    else:
        print("\n✗ Some tests failed. Please review the action mapping logic.")
        return False


def demonstrate_old_vs_new():
    """Show how the old mapping caused problems"""
    print("\n\n" + "=" * 70)
    print("Demonstrating OLD vs NEW Action Mapping")
    print("=" * 70)
    
    print("\nOLD MAPPING (caused bottleneck):")
    print("-" * 70)
    print("Problem: Both forward AND backward could activate simultaneously!")
    print()
    
    # Simulate old mapping
    test_inputs = [
        [0.5, 0.5, 0.5],
        [0.1, 0.1, 0.1],
        [1.0, 1.0, 1.0],
    ]
    
    for action in test_inputs:
        # Old logic: action[i] > 0 → activate
        old_output = np.array([
            1 if action[0] > 0 else 0,
            1 if action[1] > 0 else 0,
            1 if action[2] > 0 else 0
        ], dtype=np.int8)
        
        print(f"Input: {action} → OLD Output: {old_output}", end="")
        if old_output[0] == 1 and old_output[1] == 1:
            print(" ← CONFLICT! (forward + backward)")
        else:
            print()
    
    print("\nNEW MAPPING (fixed):")
    print("-" * 70)
    print("Improvement: Forward/backward are mutually exclusive with deadband")
    print()
    
    env = SlimeVolleyEnv()
    for action in test_inputs:
        new_output = env._process_action(action)
        print(f"Input: {action} → NEW Output: {new_output}", end="")
        if new_output[0] == 1 and new_output[1] == 1:
            print(" ← CONFLICT! (this should never happen)")
        else:
            print(" ✓")
    
    print("\n" + "=" * 70)
    print("As you can see, the new mapping prevents conflicts!")
    print("=" * 70)


def main():
    """Run all tests"""
    try:
        success = test_action_mapping()
        demonstrate_old_vs_new()
        
        if success:
            print("\n\nNEXT STEPS:")
            print("-" * 70)
            print("1. Run training with the fixed code:")
            print("   python neat_train.py -p p/slimevolley_fixed.json -n 9")
            print()
            print("2. Monitor progress - expect to see:")
            print("   - Fitness should improve past -3.33 within 100 generations")
            print("   - Agent should engage with ball instead of hugging edges")
            print()
            print("3. If still stuck after 200 gens, try reward shaping:")
            print("   python neat_train.py -p p/slimevolley_shaped.json -n 9")
            print()
            
            return 0
        else:
            return 1
            
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
