#!/usr/bin/env python3
"""
Quick diagnostic to verify the saturation hypothesis.
Run this from your prettyNEAT directory.
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)

# Import your actual NEAT code
try:
    from neat_src.ann import act
    from domain import GymTask, games

    HAS_DOMAIN = True
except ImportError:
    HAS_DOMAIN = False
    print("Could not import domain module - running simplified test")

# Create minimal network like NEAT does
n_input = 12
n_output = 2
n_nodes = 1 + n_input + n_output  # 15 nodes

# Random weights in [-2, 2]
wMat = np.zeros((n_nodes, n_nodes))
for out_idx in range(n_nodes - n_output, n_nodes):
    for in_idx in range(n_input + 1):  # bias + inputs
        wMat[in_idx, out_idx] = np.random.uniform(-2, 2)

# Simulate typical SlimeVolley observations
# (approximate ranges from actual environment)
test_observations = [
    np.array([0.0, 0.1, 0.0, 0.0, 0.2, 0.5, -0.1, 0.3, 0.5, 0.1, 0.0, 0.0]),  # typical
    np.array(
        [0.5, 0.1, 0.2, 0.0, 0.0, 0.8, 0.0, -0.5, -0.5, 0.1, 0.0, 0.0]
    ),  # ball high
    np.array(
        [-0.3, 0.1, -0.1, 0.0, -0.2, 0.3, 0.1, 0.0, 0.5, 0.1, 0.0, 0.0]
    ),  # moving left
]

print("=" * 60)
print("SATURATION TEST")
print("=" * 60)
print(f"\nNetwork: {n_input} inputs + 1 bias -> {n_output} outputs")
print(f"Total connections per output: {n_input + 1}")
print(f"Weight range: [-2, 2]")

print("\n" + "-" * 60)
print("Testing network outputs with typical observations:")
print("-" * 60)


def forward(obs, wMat):
    """Simple forward pass"""
    n_nodes = wMat.shape[0]
    nodeAct = np.zeros(n_nodes)
    nodeAct[0] = 1.0  # bias
    nodeAct[1:13] = obs

    # Propagate to outputs (last 2 nodes)
    for iNode in range(13, 15):
        nodeAct[iNode] = np.dot(nodeAct, wMat[:, iNode])

    return nodeAct[-2:]


saturation_count = 0
total_tests = 100

print(f"\n{'Observation':<50} {'Raw Output':<25} {'Clipped':<20} {'Saturated?'}")
print("-" * 110)

for i, obs in enumerate(test_observations):
    raw_out = forward(obs, wMat)
    clipped = np.clip(raw_out, -1, 1)
    is_saturated = (np.abs(clipped) == 1.0).any()
    sat_str = "YES ⚠️" if is_saturated else "no"
    print(f"{str(obs[:6])+'...':<50} {str(raw_out):<25} {str(clipped):<20} {sat_str}")

# Test many random observations
print("\n" + "-" * 60)
print(f"Testing {total_tests} random observations:")
print("-" * 60)

sat_both = 0
sat_one = 0
sat_none = 0

for _ in range(total_tests):
    obs = np.random.randn(12) * 0.5  # Typical observation range
    raw_out = forward(obs, wMat)
    clipped = np.clip(raw_out, -1, 1)
    n_sat = np.sum(np.abs(clipped) == 1.0)
    if n_sat == 2:
        sat_both += 1
    elif n_sat == 1:
        sat_one += 1
    else:
        sat_none += 1

print(
    f"Both outputs saturated: {sat_both}/{total_tests} ({100*sat_both/total_tests:.0f}%)"
)
print(
    f"One output saturated:   {sat_one}/{total_tests} ({100*sat_one/total_tests:.0f}%)"
)
print(
    f"No saturation:          {sat_none}/{total_tests} ({100*sat_none/total_tests:.0f}%)"
)

if sat_both + sat_one > 80:
    print("\n" + "=" * 60)
    print("⚠️  HIGH SATURATION DETECTED!")
    print("=" * 60)
    print(
        """
The network outputs are almost always saturated at ±1.
This means:
  - Network has NO fine control over actions
  - All random networks produce similar actions
  - Selection has no fitness gradient to work with

SOLUTIONS (try in order):

1. USE TANH OUTPUT ACTIVATION (recommended):
   In config.py, change:
     o_act=np.full(2,1)  # linear
   To:
     o_act=np.full(2,5)  # tanh (bounded to [-1,1])

2. REDUCE WEIGHT CAP:
   In your hyperparameters, change:
     ann_absWCap: 2.0
   To:
     ann_absWCap: 0.5

3. NORMALIZE OBSERVATIONS:
   In GymTask, scale observations to [-1, 1]
"""
    )
else:
    print("\n✓ Saturation is not severe - problem is elsewhere")

# Now test the action mapping
print("\n" + "=" * 60)
print("ACTION MAPPING TEST")
print("=" * 60)


def process_action(raw_output):
    action = np.clip(raw_output, -1, 1)
    if action[0] > 0.3:
        forward, backward = 1, 0
    elif action[0] < -0.3:
        forward, backward = 0, 1
    else:
        forward, backward = 0, 0
    jump = 1 if action[1] > 0.4 else 0
    return (forward, backward, jump)


# What actions do we get?
action_counts = {}
for _ in range(1000):
    obs = np.random.randn(12) * 0.5
    raw_out = forward(obs, wMat)
    action = process_action(raw_out)
    action_counts[action] = action_counts.get(action, 0) + 1

print("\nAction distribution from this random network:")
print("-" * 60)
action_names = {
    (0, 0, 0): "NOOP (stand)",
    (1, 0, 0): "Forward",
    (0, 1, 0): "Backward",
    (0, 0, 1): "Jump",
    (1, 0, 1): "Forward+Jump",
    (0, 1, 1): "Backward+Jump",
}
for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
    name = action_names.get(action, str(action))
    print(f"  {name:<20}: {count/10:.1f}%")

# Check if all actions are being used
if len(action_counts) < 3:
    print("\n⚠️  Network is only producing 1-2 different actions!")
    print("   This confirms saturation is killing action diversity.")

# Test with GymTask if available
if HAS_DOMAIN:
    print("\n" + "=" * 60)
    print("ACTUAL GYMTASK FITNESS TEST")
    print("=" * 60)

    game = games["slimevolley"]
    task = GymTask(game, nReps=5)

    wVec = wMat.flatten()
    aVec = np.ones(n_nodes)

    fitnesses = []
    for i in range(10):
        # Create new random network each time
        wMat_new = np.zeros((n_nodes, n_nodes))
        for out_idx in range(n_nodes - n_output, n_nodes):
            for in_idx in range(n_input + 1):
                wMat_new[in_idx, out_idx] = np.random.uniform(-2, 2)
        wVec = wMat_new.flatten()

        fitness = task.getFitness(wVec, aVec)
        fitnesses.append(fitness)
        print(f"  Random network {i+1}: fitness = {fitness:.2f}")

    print(f"\nFitness variance: {np.std(fitnesses):.3f}")
    if np.std(fitnesses) < 0.5:
        print("⚠️  Very low fitness variance - selection cannot distinguish networks!")
