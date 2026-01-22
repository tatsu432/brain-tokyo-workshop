#!/usr/bin/env python3
"""
Deep Diagnostic for SlimeVolley NEAT

This tests the ACTUAL evaluation pipeline that NEAT uses,
including the correct activation functions from config.
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)

print("=" * 70)
print("SLIMEVOLLEY DEEP DIAGNOSTIC")
print("=" * 70)

# ============================================================================
# TEST 1: Verify config is correct
# ============================================================================
print("\n" + "=" * 70)
print("TEST 1: Config Verification")
print("=" * 70)

try:
    from domain.config import games

    game = games["slimevolley"]

    print(f"SlimeVolley config:")
    print(f"  input_size:  {game.input_size}")
    print(f"  output_size: {game.output_size}")
    print(f"  o_act:       {game.o_act}  (1=linear, 5=tanh)")
    print(f"  layers:      {game.layers}")
    print(f"  weightCap:   {game.weightCap}")
    print(f"  actSelect:   {game.actionSelect}")

    if np.all(game.o_act == 5):
        print("\n  ✓ Output activation is TANH")
    else:
        print(f"\n  ⚠ Output activation is {game.o_act}, not tanh!")

except Exception as e:
    print(f"✗ Config check failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# TEST 2: Test the ACTUAL ann.act() function with tanh
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: ANN Output with TANH Activation")
print("=" * 70)

try:
    from neat_src.ann import act, _applyAct
    from domain.config import games

    game = games["slimevolley"]
    n_input = game.input_size
    n_output = game.output_size
    n_nodes = 1 + n_input + n_output  # 15 nodes for minimal network

    # Create activation vector like GymTask does
    # This should be: [1(bias), 1,1,1,1,1,1,1,1,1,1,1,1(inputs), 5,5(outputs)]
    aVec = np.r_[np.full(1, 1), game.i_act, game.o_act]
    print(f"Activation vector (aVec): {aVec}")
    print(f"  Last 2 elements (outputs): {aVec[-2:]} (should be 5 for tanh)")

    # Create random weight matrix
    wMat = np.zeros((n_nodes, n_nodes))
    for out_idx in range(n_nodes - n_output, n_nodes):
        for in_idx in range(n_input + 1):
            wMat[in_idx, out_idx] = np.random.uniform(-2, 2)

    wVec = wMat.flatten()

    # Test with sample observations
    test_obs = np.array([[0.0, 0.1, 0.0, 0.0, 0.2, 0.5, -0.1, 0.3, 0.5, 0.1, 0.0, 0.0]])

    output = act(wVec, aVec, n_input, n_output, test_obs)
    print(f"\nSample observation: {test_obs[0][:6]}...")
    print(f"Network output: {output.flatten()}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    if np.all(np.abs(output) <= 1.0):
        print("✓ Output is bounded to [-1, 1] - tanh is working!")
    else:
        print("⚠ Output exceeds [-1, 1] - tanh may not be applied!")

    # Test many observations
    print("\nTesting 100 random observations:")
    all_outputs = []
    for _ in range(100):
        obs = np.random.randn(1, 12) * 0.5
        out = act(wVec, aVec, n_input, n_output, obs)
        all_outputs.append(out.flatten())

    all_outputs = np.array(all_outputs)
    print(
        f"  Output 0 range: [{all_outputs[:,0].min():.3f}, {all_outputs[:,0].max():.3f}]"
    )
    print(
        f"  Output 1 range: [{all_outputs[:,1].min():.3f}, {all_outputs[:,1].max():.3f}]"
    )

    # Check saturation with tanh
    sat_rate = np.mean(np.abs(all_outputs) > 0.99)
    print(f"  Saturation rate (|output| > 0.99): {100*sat_rate:.1f}%")

except Exception as e:
    print(f"✗ ANN test failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# TEST 3: Actual GymTask Fitness with Correct aVec
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: GymTask Fitness Variance (CRITICAL TEST)")
print("=" * 70)

try:
    from domain import GymTask, games

    game = games["slimevolley"]
    task = GymTask(game, nReps=5)

    n_input = game.input_size
    n_output = game.output_size
    n_nodes = 1 + n_input + n_output

    # Create activation vector correctly
    aVec = np.r_[np.full(1, 1), game.i_act, game.o_act]

    print(f"Testing with aVec[-2:] = {aVec[-2:]} (output activations)")

    fitnesses = []
    print("\nEvaluating 20 random networks:")
    for i in range(20):
        # Create new random weights each time
        wMat = np.zeros((n_nodes, n_nodes))
        for out_idx in range(n_nodes - n_output, n_nodes):
            for in_idx in range(n_input + 1):
                wMat[in_idx, out_idx] = np.random.uniform(-2, 2)

        wVec = wMat.flatten()
        fitness = task.getFitness(wVec, aVec)
        fitnesses.append(fitness)
        print(f"  Network {i+1:2d}: fitness = {fitness:6.2f}")

    print(f"\nStatistics:")
    print(f"  Mean:     {np.mean(fitnesses):.2f}")
    print(f"  Std:      {np.std(fitnesses):.3f}")
    print(f"  Min:      {min(fitnesses):.2f}")
    print(f"  Max:      {max(fitnesses):.2f}")

    if np.std(fitnesses) < 0.1:
        print(f"\n⚠️  CRITICAL: FITNESS VARIANCE IS NEARLY ZERO!")
        print(f"   This means selection CANNOT distinguish networks.")
        print(f"   Problem is NOT in the activation function.")
    elif np.std(fitnesses) < 0.5:
        print(f"\n⚠️  Low fitness variance ({np.std(fitnesses):.3f})")
        print(f"   Selection has weak signal to work with.")
    else:
        print(f"\n✓ Good fitness variance ({np.std(fitnesses):.3f})")
        print(f"   Selection should be able to distinguish networks.")

except Exception as e:
    print(f"✗ GymTask test failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# TEST 4: What Actions Are Actually Being Taken?
# ============================================================================
print("\n" + "=" * 70)
print("TEST 4: Action Analysis During Actual Gameplay")
print("=" * 70)

try:
    import slimevolleygym
    import gym as old_gym
    from domain.slimevolley import SlimeVolleyEnv
    from neat_src.ann import act, selectAct
    from domain.config import games

    game = games["slimevolley"]

    # Create a random network
    n_input = game.input_size
    n_output = game.output_size
    n_nodes = 1 + n_input + n_output

    wMat = np.zeros((n_nodes, n_nodes))
    for out_idx in range(n_nodes - n_output, n_nodes):
        for in_idx in range(n_input + 1):
            wMat[in_idx, out_idx] = np.random.uniform(-2, 2)

    wVec = wMat.flatten()
    aVec = np.r_[np.full(1, 1), game.i_act, game.o_act]

    # Create environment
    env = SlimeVolleyEnv()
    obs = env.reset()

    action_counts = {}
    raw_outputs = []

    print("Playing 500 steps with random network...")

    for step in range(500):
        # Get network output (like GymTask does)
        annOut = act(wVec, aVec, n_input, n_output, obs.reshape(1, -1))
        action_continuous = selectAct(annOut, game.actionSelect)
        raw_outputs.append(action_continuous.copy())

        # Environment converts to binary
        obs, reward, done, info = env.step(action_continuous)

        # Track what binary action was taken
        binary = env._process_action(action_continuous)
        key = tuple(binary)
        action_counts[key] = action_counts.get(key, 0) + 1

        if done:
            obs = env.reset()

    raw_outputs = np.array(raw_outputs)

    print(f"\nRaw network outputs (continuous, AFTER tanh if applied):")
    print(
        f"  Output 0: range=[{raw_outputs[:,0].min():.3f}, {raw_outputs[:,0].max():.3f}], "
        f"mean={raw_outputs[:,0].mean():.3f}"
    )
    print(
        f"  Output 1: range=[{raw_outputs[:,1].min():.3f}, {raw_outputs[:,1].max():.3f}], "
        f"mean={raw_outputs[:,1].mean():.3f}"
    )

    print(f"\nBinary action distribution:")
    action_names = {
        (0, 0, 0): "NOOP",
        (1, 0, 0): "Forward",
        (0, 1, 0): "Backward",
        (0, 0, 1): "Jump",
        (1, 0, 1): "Forward+Jump",
        (0, 1, 1): "Backward+Jump",
    }
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        name = action_names.get(action, str(action))
        print(f"  {name:<15}: {count/5:.1f}%")

    env.close()

except Exception as e:
    print(f"✗ Action analysis failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# TEST 5: Compare Networks Directly
# ============================================================================
print("\n" + "=" * 70)
print("TEST 5: Do Different Networks Behave Differently?")
print("=" * 70)

try:
    from domain.slimevolley import SlimeVolleyEnv
    from neat_src.ann import act, selectAct
    from domain.config import games

    game = games["slimevolley"]
    n_input = game.input_size
    n_output = game.output_size
    n_nodes = 1 + n_input + n_output

    # Create 5 different networks
    networks = []
    for _ in range(5):
        wMat = np.zeros((n_nodes, n_nodes))
        for out_idx in range(n_nodes - n_output, n_nodes):
            for in_idx in range(n_input + 1):
                wMat[in_idx, out_idx] = np.random.uniform(-2, 2)
        networks.append(wMat.flatten())

    aVec = np.r_[np.full(1, 1), game.i_act, game.o_act]

    # Test all networks on the same observation
    test_obs = np.array([[0.0, 0.1, 0.0, 0.0, 0.2, 0.5, -0.1, 0.3, 0.5, 0.1, 0.0, 0.0]])

    print(f"Testing 5 networks on same observation: {test_obs[0][:4]}...")
    print("-" * 60)

    outputs = []
    for i, wVec in enumerate(networks):
        out = act(wVec, aVec, n_input, n_output, test_obs)
        outputs.append(out.flatten())
        print(f"  Network {i+1}: output = {out.flatten()}")

    outputs = np.array(outputs)
    output_variance = np.std(outputs, axis=0)
    print(f"\nOutput variance across networks:")
    print(f"  Output 0 variance: {output_variance[0]:.4f}")
    print(f"  Output 1 variance: {output_variance[1]:.4f}")

    if np.all(output_variance < 0.01):
        print("\n⚠️  All networks produce IDENTICAL outputs!")
        print("   This should NOT happen with random weights.")
        print("   Check if weights are being used correctly.")
    elif np.all(output_variance < 0.1):
        print("\n⚠️  Very low output variance between networks.")
    else:
        print("\n✓ Networks produce different outputs - good!")

except Exception as e:
    print(f"✗ Network comparison failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# TEST 6: Simple Policy Baseline
# ============================================================================
print("\n" + "=" * 70)
print("TEST 6: Simple Policy Baseline")
print("=" * 70)

try:
    import slimevolleygym
    import gym as old_gym

    env = old_gym.make("SlimeVolley-v0", disable_env_checker=True)

    def simple_policy(obs):
        """Follow ball, jump when close and high"""
        agent_x = obs[0]
        ball_x = obs[4]
        ball_y = obs[5]

        if ball_x > agent_x + 0.1:
            forward, backward = 1, 0
        elif ball_x < agent_x - 0.1:
            forward, backward = 0, 1
        else:
            forward, backward = 0, 0

        jump = 1 if abs(ball_x - agent_x) < 0.3 and ball_y > 0.5 else 0
        return np.array([forward, backward, jump])

    rewards = []
    for _ in range(10):
        obs = env.reset()
        total = 0
        done = False
        while not done:
            action = simple_policy(obs)
            obs, reward, done, _ = env.step(action)
            total += reward
        rewards.append(total)

    print(f"Simple policy over 10 games:")
    print(f"  Mean: {np.mean(rewards):.2f}")
    print(f"  Std:  {np.std(rewards):.2f}")
    print(f"  Range: [{min(rewards)}, {max(rewards)}]")

    if np.mean(rewards) > -3:
        print(f"\n✓ Simple policy achieves {np.mean(rewards):.1f}")
        print(f"  NEAT should be able to find something at least this good!")

    env.close()

except Exception as e:
    print(f"✗ Simple policy test failed: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print(
    """
Key questions answered:

1. Is tanh being applied to outputs?
   → Check if output range is bounded to [-1, 1]

2. Do different random networks produce different outputs?
   → Check output variance in Test 5

3. Do different networks get different fitness?
   → Check fitness variance in Test 3

4. What actions are networks actually taking?
   → Check action distribution in Test 4

If tanh is working but fitness variance is still low, the problem
may be that the BASELINE OPPONENT is so strong that even with 
different actions, the outcome is the same (always lose 5-0).

NEXT STEPS based on results:
- If output not bounded: tanh not being applied correctly
- If networks output same thing: weights not being used
- If fitness variance ~0: need reward shaping or easier opponent
"""
)
