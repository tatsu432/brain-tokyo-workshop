import numpy as np
import argparse

np.set_printoptions(precision=2)
np.set_printoptions(linewidth=160)

from neat_src import *  # NEAT
from domain import *  # Task environments


def main(argv):
    """Tests network on task N times and returns mean fitness."""
    infile = args.infile
    outPref = args.outPref
    hyp_default = args.default
    hyp_adjust = args.hyperparam
    nRep = args.nReps
    view = args.view
    raw_reward = args.raw_reward

    # Load task and parameters
    hyp = loadHyp(pFileName=hyp_default)
    updateHyp(hyp, hyp_adjust)
    task = GymTask(games[hyp["task"]], nReps=hyp["alg_nReps"])

    # Bullet needs some extra help getting started
    if hyp["task"].startswith("bullet"):
        # Call render() without mode parameter (render_mode should be set at init if needed)
        task.env.render()

    # Import and Test network
    wVec, aVec, wKey = importNet(infile)

    # Check if this is a SlimeVolley task and we want raw rewards
    is_slimevolley = hyp["task"].startswith("slimevolley")

    if is_slimevolley and raw_reward:
        # Test with raw game reward (actual score -5 to +5)
        raw_rewards = testSlimeVolleyRaw(task, wVec, aVec, view=view, nRep=nRep)
        print("[***]\tRaw Game Score (per episode):", raw_rewards)
        print("[***]\tMean Raw Game Score:", np.mean(raw_rewards))
        lsave(outPref + "rawScore.out", raw_rewards)
    else:
        fitness = np.empty(1)
        fitness[:] = task.getFitness(wVec, aVec, view=view, nRep=nRep)
        print("[***]\tFitness (shaped):", fitness)
        lsave(outPref + "fitDist.out", fitness)


def testSlimeVolleyRaw(task, wVec, aVec, view=False, nRep=1, seed=-1):
    """
    Test SlimeVolley against the BASELINE opponent (built-in RNN policy).
    Returns raw game reward (actual score -5 to +5).

    NOTE: This tests against the internal baseline opponent, NOT self-play.
    The baseline is a tiny RNN policy with ~120 parameters trained by the author.

    Args:
      task: GymTask instance
      wVec: Weight vector
      aVec: Activation vector
      view: Whether to render
      nRep: Number of repetitions
      seed: Random seed

    Returns:
      raw_rewards: Array of raw game scores for each episode
    """
    import random
    import time

    # Use the original SlimeVolley environment for testing (against baseline opponent)
    # This ensures we test against the exact built-in RNN baseline, not any modified opponent
    try:
        from slimevolleygym.slimevolley import SlimeVolleyEnv

        test_env = SlimeVolleyEnv()
        print("[***]\tTesting against BASELINE opponent (built-in RNN policy)")
    except ImportError:
        # Fallback to task's environment
        test_env = task.env
        print("[***]\tTesting against baseline (using task environment)")

    wVec = np.copy(wVec)
    wVec[np.isnan(wVec)] = 0

    raw_rewards = []

    for iRep in range(nRep):
        if seed >= 0:
            random.seed(seed + iRep)
            np.random.seed(seed + iRep)
            test_env.seed(seed + iRep)

        state = test_env.reset()

        done = False
        rallies_won = 0
        rallies_lost = 0

        for tStep in range(task.maxEpisodeLength):
            # Get action from our trained network
            annOut = act(wVec, aVec, task.nInput, task.nOutput, state)
            action = selectAct(annOut, task.actSelect)

            # Process action for SlimeVolley (convert to binary)
            from domain.slimevolley_actions import SlimeVolleyActionProcessor
            processor = SlimeVolleyActionProcessor(clip_actions=True)
            binary_action = processor.process(action)

            # Step environment - opponent uses built-in baseline policy (otherAction=None)
            state, reward, done, info = test_env.step(binary_action)

            # Track raw game reward (+1 for win, -1 for loss)
            if reward > 0:
                rallies_won += 1
            elif reward < 0:
                rallies_lost += 1

            if view:
                # Call render() without mode parameter
                test_env.render()
                time.sleep(0.02)

            if done:
                break

        final_score = rallies_won - rallies_lost
        raw_rewards.append(final_score)

        if view:
            print(
                f"  Episode {iRep+1}: Score={final_score}, Won={rallies_won}, Lost={rallies_lost}"
            )

    test_env.close()
    return np.array(raw_rewards)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    """Parse input and launch"""
    parser = argparse.ArgumentParser(description=("Test ANNs on Task"))

    parser.add_argument(
        "-i",
        "--infile",
        type=str,
        help="file name for genome input",
        default="log/test_best.out",
    )

    parser.add_argument(
        "-o",
        "--outPref",
        type=str,
        help="file name prefix for result input",
        default="log/result_",
    )

    parser.add_argument(
        "-d",
        "--default",
        type=str,
        help="default hyperparameter file",
        default="p/default_neat.json",
    )

    parser.add_argument(
        "-p", "--hyperparam", type=str, help="hyperparameter file", default=None
    )

    parser.add_argument(
        "-r", "--nReps", type=int, help="Number of repetitions", default=1
    )

    parser.add_argument(
        "-v",
        "--view",
        type=str2bool,
        help="Visualize (True) or Save (False)",
        default=True,
    )

    parser.add_argument(
        "--raw_reward",
        type=str2bool,
        help="Show raw game reward instead of shaped (SlimeVolley only)",
        default=True,
    )

    args = parser.parse_args()
    main(args)
