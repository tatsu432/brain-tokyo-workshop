#!/usr/bin/env python3
"""
Diagnose NEAT's Actual Population

This checks what networks NEAT actually creates and evaluates,
not what we think it creates.
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)

print("=" * 70)
print("NEAT POPULATION DIAGNOSTIC")
print("=" * 70)

# ============================================================================
# TEST 1: Check Initial Population
# ============================================================================
print("\n" + "=" * 70)
print("TEST 1: NEAT Initial Population Structure")
print("=" * 70)

try:
    from neat_src import Neat, loadHyp
    from domain.config import games

    # Load hyperparameters
    hyp = loadHyp(pFileName="p/default_neat.json")

    # Override for slimevolley
    hyp["task"] = "slimevolley"

    print(f"Hyperparameters:")
    print(f"  task: {hyp.get('task')}")
    print(f"  popSize: {hyp.get('popSize')}")
    print(f"  ann_nInput: {hyp.get('ann_nInput')}")
    print(f"  ann_nOutput: {hyp.get('ann_nOutput')}")
    print(f"  ann_initAct: {hyp.get('ann_initAct')}")
    print(f"  ann_absWCap: {hyp.get('ann_absWCap')}")

    # Create NEAT instance
    neat = Neat(hyp)

    # Get initial population
    pop = neat.ask()

    print(f"\nInitial population size: {len(pop)}")

    # Analyze first few individuals
    print(f"\nAnalyzing first 5 individuals:")
    for i, ind in enumerate(pop[:5]):
        wMat = ind.wMat
        aVec = ind.aVec

        # Count connections
        n_connections = np.count_nonzero(~np.isnan(wMat) & (wMat != 0))
        n_nodes = len(aVec)

        # Check weight range
        weights = wMat[~np.isnan(wMat) & (wMat != 0)]
        if len(weights) > 0:
            w_min, w_max = weights.min(), weights.max()
        else:
            w_min, w_max = 0, 0

        print(f"\n  Individual {i+1}:")
        print(f"    Nodes: {n_nodes}")
        print(f"    Connections: {n_connections}")
        print(f"    Weight range: [{w_min:.3f}, {w_max:.3f}]")
        print(f"    aVec: {aVec}")
        print(f"    aVec output nodes: {aVec[-2:]} (should be 5 for tanh)")

except Exception as e:
    print(f"✗ Population check failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# TEST 2: Evaluate Initial Population
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: Initial Population Fitness Distribution")
print("=" * 70)

try:
    from domain import GymTask, games
    from neat_src import Neat, loadHyp

    # Load config
    hyp = loadHyp(pFileName="p/default_neat.json")
    hyp["task"] = "slimevolley"

    game = games["slimevolley"]
    task = GymTask(game, nReps=3)

    # Create NEAT and get population
    neat = Neat(hyp)
    pop = neat.ask()

    print(f"Evaluating first 20 individuals from NEAT population...")

    fitnesses = []
    for i, ind in enumerate(pop[:20]):
        wVec = ind.wMat.flatten()
        aVec = ind.aVec

        fitness = task.getFitness(wVec, aVec)
        fitnesses.append(fitness)
        print(
            f"  Ind {i+1:2d}: fitness = {fitness:6.2f}, "
            f"nodes = {len(aVec)}, "
            f"output_act = {aVec[-2:]}"
        )

    print(f"\nFitness statistics:")
    print(f"  Mean:  {np.mean(fitnesses):.2f}")
    print(f"  Std:   {np.std(fitnesses):.3f}")
    print(f"  Range: [{min(fitnesses):.2f}, {max(fitnesses):.2f}]")

    unique_fitnesses = len(set([round(f, 2) for f in fitnesses]))
    print(f"  Unique fitness values: {unique_fitnesses}/20")

    if np.std(fitnesses) < 0.1:
        print(f"\n⚠️  CRITICAL: Almost no fitness variance in initial population!")
    elif unique_fitnesses < 5:
        print(f"\n⚠️  Very few unique fitness values - selection will struggle")
    else:
        print(f"\n✓ Reasonable fitness variance")

except Exception as e:
    print(f"✗ Initial evaluation failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# TEST 3: Check What's Different About CartPole SwingUp
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: Compare with CartPole SwingUp (working case)")
print("=" * 70)

try:
    from domain import GymTask, games

    if "swingup" in games:
        game_swingup = games["swingup"]
        print(f"CartPole SwingUp config:")
        print(f"  input_size:  {game_swingup.input_size}")
        print(f"  output_size: {game_swingup.output_size}")
        print(f"  o_act:       {game_swingup.o_act}")
        print(f"  layers:      {game_swingup.layers}")

        # Quick fitness test
        task_swingup = GymTask(game_swingup, nReps=3)

        n_input = game_swingup.input_size
        n_output = game_swingup.output_size
        n_nodes = 1 + n_input + n_output

        fitnesses_swingup = []
        print(f"\nTesting 10 random SwingUp networks:")
        for i in range(10):
            wMat = np.zeros((n_nodes, n_nodes))
            for out_idx in range(n_nodes - n_output, n_nodes):
                for in_idx in range(n_input + 1):
                    wMat[in_idx, out_idx] = np.random.uniform(-2, 2)

            wVec = wMat.flatten()
            aVec = np.r_[np.full(1, 1), game_swingup.i_act, game_swingup.o_act]

            fitness = task_swingup.getFitness(wVec, aVec)
            fitnesses_swingup.append(fitness)
            print(f"  Network {i+1}: fitness = {fitness:.2f}")

        print(f"\nSwingUp fitness variance: {np.std(fitnesses_swingup):.2f}")
        print(f"SlimeVolley fitness variance: ~0 (from earlier test)")

        print(f"\nKEY DIFFERENCE:")
        print(f"  SwingUp: Dense reward every step based on pole angle")
        print(f"  SlimeVolley: Sparse reward only when points scored")
    else:
        print("SwingUp not found in games")

except Exception as e:
    print(f"✗ SwingUp comparison failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print(
    """
The core issue is likely REWARD SPARSITY, not network architecture:

CartPole SwingUp:
  - Reward every step: r = cos(theta) - position_penalty
  - Random policy gets varied rewards (some balance briefly by luck)
  - High fitness variance → selection can work

SlimeVolley:
  - Reward only on point scored: +1 or -1
  - Random policy loses 5-0 or 5-1 every time (very consistent)
  - Near-zero fitness variance → selection has no signal

SOLUTIONS:

1. ADD REWARD SHAPING (recommended):
   Use SlimeVolley-Shaped-v0 that adds small rewards for:
   - Moving toward ball
   - Being close to ball  
   - Hitting the ball
   
2. USE EASIER OPPONENT:
   Train against a weaker or stationary opponent first
   
3. INCREASE POPULATION + GENERATIONS:
   With sparse rewards, you need more random exploration
   Population: 500+, Generations: 5000+
"""
)
