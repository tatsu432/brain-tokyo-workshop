#!/usr/bin/env python
"""
Deep diagnostic to find the REAL issue with SlimeVolley NEAT training.

Run this to understand what's actually happening.
"""

import numpy as np
import sys
import os

print("=" * 80)
print("SLIMEVOLLEY NEAT TRAINING - DEEP DIAGNOSTIC")
print("=" * 80)

print("\n## ISSUE #1: Reward Shaping is Reward Hacking")
print("-" * 80)
print("Your shaped version shows FITNESS = +3 to +5")
print("But SlimeVolley only gives ±1 per life won/lost (max ±5 per episode)")
print()
print("This means:")
print("  - Original game reward: ~-3 (agent is losing)")
print("  - Shaped rewards: ~+8 to +10 (overwhelming the signal!)")
print("  - Agent learned to maximize shaped rewards, NOT win the game")
print()
print("  🚨 This is 'reward hacking' - agent exploits the shaped rewards")
print()

print("\n## ISSUE #2: Need to Test WITHOUT Reward Shaping")
print("-" * 80)
print("Did you test p/slimevolley_fixed.json (without 'shaped' in name)?")
print()
print("The shaped version (slimevolley_shaped.json) has broken rewards.")
print("You MUST test the fixed version WITHOUT shaping:")
print()
print("  python neat_train.py -p p/slimevolley_fixed.json -n 9")
print()

print("\n## ISSUE #3: Possible Deeper Problems")
print("-" * 80)
print("If the non-shaped version ALSO doesn't work, then we need to look at:")
print()
print("1. Network output ranges with different activation functions")
print("2. Whether o_act (output activation) is set correctly")
print("3. Whether the action mapping thresholds match activation ranges")
print()

# Check config
try:
    import json

    print("\n## Checking Configurations")
    print("-" * 80)

    configs_to_check = [
        "p/slimevolley.json",
        "p/slimevolley_fixed.json",
        "p/slimevolley_shaped.json",
    ]

    for config_path in configs_to_check:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
            print(f"\n{config_path}:")
            print(f"  task: {cfg.get('task', 'N/A')}")
            print(
                f"  alg_act: {cfg.get('alg_act', 'N/A')} (0=all activations, 5=tanh only)"
            )
            print(f"  alg_nReps: {cfg.get('alg_nReps', 'N/A')}")
            print(f"  popSize: {cfg.get('popSize', 'N/A')}")
        else:
            print(f"\n{config_path}: NOT FOUND")

    # Check task config
    print("\n## Checking Game Configuration")
    print("-" * 80)

    sys.path.insert(
        0, "/Users/a81808/code/self_study/brain-tokyo-workshop/WANNRelease/prettyNEAT"
    )
    from domain.config import games

    if "slimevolley" in games:
        game = games["slimevolley"]
        print("\nslimevolley game config:")
        print(f"  env_name: {game.env_name}")
        print(f"  output_size: {game.output_size}")
        print(f"  o_act: {game.o_act}")
        print(f"  layers: {game.layers}")
        print(f"  h_act: {game.h_act}")
        print()
        print(f"  🔍 output activation (o_act): {game.o_act}")
        print(f"     This means output nodes use activation #{game.o_act[0]}")

        # Show what that activation does
        from neat_src.ann import _applyAct

        test_vals = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
        outputs = _applyAct(int(game.o_act[0]), test_vals)
        print(f"     Test: input {test_vals}")
        print(f"           output {outputs}")
        print(f"     Range: [{np.min(outputs):.2f}, {np.max(outputs):.2f}]")

    if "slimevolley_shaped" in games:
        game_shaped = games["slimevolley_shaped"]
        print("\nslimevolley_shaped game config:")
        print(f"  env_name: {game_shaped.env_name}")
        print(f"  (other params same as slimevolley)")

except Exception as e:
    print(f"\n✗ Error checking configs: {e}")
    import traceback

    traceback.print_exc()

print("\n\n" + "=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)
print()
print("Based on your training output showing POSITIVE fitness:")
print()
print("1. ✓ Reward shaping IS working (fitness changed sign)")
print("2. ✗ Reward shaping is TOO STRONG (reward hacking)")
print("3. ? Need to test non-shaped version to see if action mapping fix works")
print()
print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("1. Test the non-shaped version (this is critical!):")
print("   python neat_train.py -p p/slimevolley_fixed.json -n 9")
print()
print("2. Watch the fitness values:")
print("   - Should be NEGATIVE (-4 to -3 range initially)")
print("   - Should gradually improve toward 0")
print("   - Positive values mean reward hacking, not real progress")
print()
print("3. If fixed version also stuck:")
print("   - Run this script with your test best individual")
print("   - Check actual game rewards vs total rewards")
print("   - May need to adjust output activation function")
print()
print("=" * 80)

print("\n")
