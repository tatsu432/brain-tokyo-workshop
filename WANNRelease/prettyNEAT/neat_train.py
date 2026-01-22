import os
import sys
import time
import math
import argparse
import subprocess
import warnings
import numpy as np

# Suppress gym step API deprecation warning
# Filter by message pattern and by module to catch all variations
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*Initializing environment in old step API.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.wrappers.step_api_compatibility")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

np.set_printoptions(precision=2, linewidth=160)

from neat_src.ind import Ind

# MPI
from mpi4py import MPI

comm = MPI.COMM_WORLD  # A communicator defines who can talk to whom.
rank = comm.Get_rank()
# Convention:
# rank == 0 → master
# rank > 0 → workers

# prettyNeat
from neat_src import *  # NEAT
from domain import *  # Task environments

# Non-blocking rendering (only import in master process to avoid MPI issues)
render_manager = None

# 1 master + many workers
# master: evolution logic
# workers: evaluate fitness


# -- Self-Play Configuration ------------------------------------------------ -- #


class SelfPlayConfig:
    """Configuration for self-play training."""

    def __init__(self):
        # Whether to enable self-play
        self.enabled = False

        # Evaluation mode: 'baseline', 'archive', 'mixed'
        self.eval_mode = "mixed"

        # Weight for baseline vs archive evaluation
        self.baseline_weight = 0.6
        self.archive_weight = 0.4

        # Number of archive opponents to sample
        self.n_archive_opponents = 3

        # Archive settings
        self.max_archive_size = 50
        self.archive_add_frequency = 5  # Add best every N generations
        self.archive_fitness_threshold = -3.0  # Min fitness to add to archive

        # Curriculum settings
        self.enable_curriculum = True
        self.touch_threshold = 5.0  # Avg touches to advance from 'touch' stage
        self.rally_threshold = 0.0  # Avg rally diff to advance from 'rally' stage


# Global self-play state
selfplay_config = SelfPlayConfig()
opponent_archive = []  # List of (wVec, aVec, fitness, generation) tuples


# -- Run NEAT ------------------------------------------------------------ -- #
def master():
    """Main NEAT optimization script"""
    global fileName, hyp, selfplay_config, opponent_archive, render_manager
    data = DataGatherer(fileName, hyp)

    # Check for verbose mode via environment variable
    verbose = os.environ.get("NEAT_VERBOSE", "0") == "1"
    if verbose:
        print("[Master] Verbose mode ENABLED", flush=True)

    # Set discrete action flag based on task config
    task_config = games[hyp["task"]]
    data.is_discrete_action = task_config.actionSelect in ["prob", "hard"]

    # Configure display settings from hyperparameters
    display_config = hyp.get("display_config", {})
    if display_config:
        data.set_display_config(**display_config)

    # Initialize rendering manager (non-blocking visualization)
    render_enabled = hyp.get("render_enabled", False)
    if render_enabled:
        from domain.render_selfplay import RenderManager

        render_interval = hyp.get("render_interval", 10)
        render_fps = hyp.get("render_fps", 50)
        render_max_steps = hyp.get("render_max_steps", 3000)

        render_manager = RenderManager(
            render_interval=render_interval,
            render_fps=render_fps,
            max_steps=render_max_steps,
            enabled=True,
        )
        print(
            f"Rendering ENABLED: every {render_interval} generations @ {render_fps} FPS"
        )
    else:
        render_manager = None

    # Check if self-play is enabled
    selfplay_config.enabled = hyp.get("selfplay_enabled", False)
    enable_curriculum = hyp.get("enable_curriculum", False)

    # Initialize curriculum tracking (for both self-play and non-self-play)
    curriculum_stage = "touch" if enable_curriculum else None

    if selfplay_config.enabled:
        print("Self-play mode ENABLED")
        selfplay_config.eval_mode = hyp.get("selfplay_eval_mode", "mixed")
        selfplay_config.baseline_weight = hyp.get("selfplay_baseline_weight", 0.6)
        selfplay_config.archive_weight = hyp.get("selfplay_archive_weight", 0.4)
        selfplay_config.n_archive_opponents = hyp.get("selfplay_n_archive_opponents", 3)
        selfplay_config.archive_add_frequency = hyp.get("selfplay_archive_add_freq", 5)
        selfplay_config.enable_curriculum = hyp.get("selfplay_enable_curriculum", True)

        # Override with global enable_curriculum if set
        if enable_curriculum is not None:
            selfplay_config.enable_curriculum = enable_curriculum

        # Initialize curriculum tracking
        if hasattr(data, "selfplay_stats"):
            pass
        else:
            data.selfplay_stats = {
                "avg_touches": [],
                "avg_rallies_won": [],
                "avg_rallies_lost": [],
                "curriculum_stage": [],
                "archive_size": [],
            }
    elif enable_curriculum:
        print("Curriculum learning ENABLED (non-self-play)")
        # Initialize curriculum stats for non-self-play
        if not hasattr(data, "curriculum_stats"):
            data.curriculum_stats = {
                "avg_touches": [],
                "avg_rallies_won": [],
                "avg_rallies_lost": [],
                "curriculum_stage": [],
            }

    neat = Neat(hyp)

    for gen in range(hyp["maxGen"]):
        if verbose:
            print(f"\n[Master] ===== Generation {gen} =====", flush=True)

        pop = neat.ask()  # Get newly evolved individuals from NEAT

        # Evaluate population (with self-play if enabled)
        if selfplay_config.enabled:
            reward, action_dist, pop_stats, raw_reward = batchMpiEvalSelfPlay(
                pop, track_actions=True, verbose=verbose
            )

            # Update curriculum based on population statistics
            if selfplay_config.enable_curriculum:
                curriculum_stage = update_curriculum(pop_stats, curriculum_stage)
                broadcast_curriculum_stage(curriculum_stage)

            # Add best individual to archive periodically
            if gen % selfplay_config.archive_add_frequency == 0 and gen > 0:
                best_idx = np.argmax(reward)
                best_fitness = reward[best_idx]

                if verbose:
                    print(
                        f"[Master] Gen {gen}: Checking archive add (best_fitness={best_fitness:.2f}, threshold={selfplay_config.archive_fitness_threshold})",
                        flush=True,
                    )

                if best_fitness > selfplay_config.archive_fitness_threshold:
                    best_ind = pop[best_idx]
                    wVec_to_add = best_ind.wMat.flatten()
                    aVec_to_add = best_ind.aVec.flatten()

                    if verbose:
                        print(
                            f"[Master] Adding to archive: wVec shape={wVec_to_add.shape}, aVec shape={aVec_to_add.shape}",
                            flush=True,
                        )

                    add_to_archive(wVec_to_add, aVec_to_add, best_fitness, gen)
                    # Broadcast updated archive to workers
                    broadcast_archive()

            # Track self-play stats
            data.selfplay_stats["avg_touches"].append(pop_stats.get("avg_touches", 0))
            data.selfplay_stats["avg_rallies_won"].append(
                pop_stats.get("avg_rallies_won", 0)
            )
            data.selfplay_stats["avg_rallies_lost"].append(
                pop_stats.get("avg_rallies_lost", 0)
            )
            data.selfplay_stats["curriculum_stage"].append(curriculum_stage)
            data.selfplay_stats["archive_size"].append(len(opponent_archive))

        else:
            # Evaluate population for non-self-play
            reward, action_dist, pop_stats, raw_reward = batchMpiEval(
                pop, track_actions=True
            )

            # Update curriculum for non-self-play if enabled
            if enable_curriculum:
                if "curriculum_stage" not in locals():
                    curriculum_stage = "touch"  # Initialize if not set
                curriculum_stage = update_curriculum(pop_stats, curriculum_stage)
                broadcast_curriculum_stage(curriculum_stage)

            # Track curriculum stats for non-self-play
            if enable_curriculum and hasattr(data, "curriculum_stats"):
                data.curriculum_stats["avg_touches"].append(
                    pop_stats.get("avg_touches", 0)
                )
                data.curriculum_stats["avg_rallies_won"].append(
                    pop_stats.get("avg_rallies_won", 0)
                )
                data.curriculum_stats["avg_rallies_lost"].append(
                    pop_stats.get("avg_rallies_lost", 0)
                )
                if "curriculum_stage" in locals():
                    data.curriculum_stats["curriculum_stage"].append(curriculum_stage)
                else:
                    data.curriculum_stats["curriculum_stage"].append("touch")

        neat.tell(reward)  # Send fitness to NEAT

        data = gatherData(
            data, neat, gen, hyp, action_dist=action_dist, raw_fitness=raw_reward,
            shaped_stats=pop_stats
        )

        # Display with curriculum/self-play info
        enable_curriculum = hyp.get("enable_curriculum", False)
        if selfplay_config.enabled:
            print(
                f"Gen {gen}: {data.display()} | Stage: {curriculum_stage}, Archive: {len(opponent_archive)}"
            )
        elif enable_curriculum:
            # Show curriculum stage for non-self-play
            if "curriculum_stage" in locals():
                print(f"Gen {gen}: {data.display()} | Stage: {curriculum_stage}")
            else:
                print(f"Gen {gen}: {data.display()}")
        else:
            print(f"Gen {gen}: {data.display()}")

        # Trigger non-blocking rendering of best individual (for both self-play and non-self-play)
        if render_manager is not None and render_manager.should_render(gen):
            # Get best individual from population
            best_idx = np.argmax(reward)
            best_ind = pop[best_idx]

            # Update archive in render manager for self-play visualization
            if selfplay_config.enabled and opponent_archive:
                render_manager.update_archive(opponent_archive)

            # Start non-blocking render
            render_manager.render_best(
                wVec=best_ind.wMat.flatten(),
                aVec=best_ind.aVec.flatten(),
                nInput=task_config.input_size,
                nOutput=task_config.output_size,
                actSelect=task_config.actionSelect,
                generation=gen,
                use_random_archive_opponent=hyp.get("render_use_archive_opponent", True)
                if selfplay_config.enabled
                else False,
            )

    # Clean up and data gathering at run end
    data = gatherData(data, neat, gen, hyp, savePop=True)
    data.save()
    data.savePop(neat.pop, fileName)  # Save population as 2D numpy arrays

    # Save archive if self-play was used
    if selfplay_config.enabled and opponent_archive:
        save_archive(fileName)

    # Clean up render manager
    if render_manager is not None:
        print("Waiting for render to finish...")
        render_manager.wait_for_render(timeout=10.0)
        render_manager.stop()

    stopAllWorkers()


def update_curriculum(pop_stats, current_stage):
    """Update curriculum stage based on population statistics."""
    global selfplay_config, hyp

    avg_touches = pop_stats.get("avg_touches", 0)
    avg_rally_diff = pop_stats.get("avg_rallies_won", 0) - pop_stats.get(
        "avg_rallies_lost", 0
    )

    # Get thresholds (from selfplay config if available, otherwise from hyp)
    touch_threshold = (
        selfplay_config.touch_threshold
        if selfplay_config.enabled
        else hyp.get("touch_threshold", 5.0)
    )
    rally_threshold = (
        selfplay_config.rally_threshold
        if selfplay_config.enabled
        else hyp.get("rally_threshold", 0.0)
    )

    if current_stage == "touch":
        if avg_touches >= touch_threshold:
            print(
                f"Curriculum: Advancing from 'touch' to 'rally' (avg_touches={avg_touches:.1f})"
            )
            return "rally"

    elif current_stage == "rally":
        if avg_rally_diff >= rally_threshold:
            print(
                f"Curriculum: Advancing from 'rally' to 'win' (rally_diff={avg_rally_diff:.1f})"
            )
            return "win"

    return current_stage


def add_to_archive(wVec, aVec, fitness, generation):
    """Add an individual to the opponent archive."""
    global opponent_archive, selfplay_config

    if len(opponent_archive) < selfplay_config.max_archive_size:
        opponent_archive.append((wVec.copy(), aVec.copy(), fitness, generation))
    else:
        # Replace worst if new individual is better
        worst_idx = min(
            range(len(opponent_archive)), key=lambda i: opponent_archive[i][2]
        )
        if fitness > opponent_archive[worst_idx][2]:
            opponent_archive[worst_idx] = (
                wVec.copy(),
                aVec.copy(),
                fitness,
                generation,
            )


# Global storage to keep non-blocking send requests and data alive
_pending_broadcast_data = []
_pending_broadcast_requests = []


def broadcast_archive():
    """Broadcast archive to all workers using non-blocking sends.

    NOTE: We use isend() (non-blocking) because workers are blocked waiting for
    work on tag=1, not actively receiving tag=100. Blocking send could deadlock.

    IMPORTANT: We store the data and requests globally to prevent garbage collection
    before the sends complete. Workers receive these messages asynchronously.
    """
    global nWorker, opponent_archive, _pending_broadcast_data, _pending_broadcast_requests
    nSlave = nWorker - 1

    verbose = os.environ.get("NEAT_VERBOSE", "0") == "1"
    if verbose:
        print(
            f"[Master] Broadcasting archive (size={len(opponent_archive)}) to {nSlave} workers...",
            flush=True,
        )

    # Clear any completed previous broadcasts
    _pending_broadcast_data = []
    _pending_broadcast_requests = []

    # Create a serializable snapshot of the archive
    archive_snapshot = [(w.copy(), a.copy(), f, g) for w, a, f, g in opponent_archive]

    # Send to each worker with its own copy of data (to prevent buffer issues)
    for iWork in range(1, nSlave + 1):
        # Each worker gets its own copy to ensure buffer validity
        import copy

        msg = ("archive_update", copy.deepcopy(archive_snapshot))
        _pending_broadcast_data.append(msg)  # Keep reference alive

        req = comm.isend(msg, dest=iWork, tag=100)
        _pending_broadcast_requests.append(req)  # Keep request alive

        if verbose:
            print(f"[Master] Queued archive send to worker {iWork}", flush=True)

    if verbose:
        print(f"[Master] Archive broadcast queued (non-blocking)", flush=True)


# Global storage for curriculum broadcasts
_pending_curriculum_data = []
_pending_curriculum_requests = []


def broadcast_curriculum_stage(stage):
    """Broadcast curriculum stage to all workers using non-blocking sends."""
    global nWorker, _pending_curriculum_data, _pending_curriculum_requests
    nSlave = nWorker - 1

    # Clear previous
    _pending_curriculum_data = []
    _pending_curriculum_requests = []

    # Use non-blocking sends with persistent references
    for iWork in range(1, nSlave + 1):
        msg = ("curriculum_update", stage)
        _pending_curriculum_data.append(msg)
        req = comm.isend(msg, dest=iWork, tag=101)
        _pending_curriculum_requests.append(req)


def save_archive(fileName):
    """Save opponent archive to file."""
    global opponent_archive
    import pickle

    archive_path = f"log/{fileName}_archive.pkl"
    with open(archive_path, "wb") as f:
        pickle.dump(opponent_archive, f)
    print(f"Saved archive with {len(opponent_archive)} opponents to {archive_path}")


def load_archive(fileName):
    """Load opponent archive from file."""
    global opponent_archive
    import pickle

    archive_path = f"log/{fileName}_archive.pkl"
    if os.path.exists(archive_path):
        with open(archive_path, "rb") as f:
            opponent_archive = pickle.load(f)
        print(
            f"Loaded archive with {len(opponent_archive)} opponents from {archive_path}"
        )


def gatherData(data, neat, gen, hyp, savePop=False, action_dist=None, raw_fitness=None, shaped_stats=None):
    """Collects run data, saves it to disk, and exports pickled population

    Args:
      data       - (DataGatherer)  - collected run data
      neat       - (Neat)          - neat algorithm container
        .pop     - [Ind]           - list of individuals in population
        .species - (Species)       - current species
      gen        - (ind)           - current generation
      hyp        - (dict)          - algorithm hyperparameters
      savePop    - (bool)          - save current population to disk?
      action_dist - (np_array)     - aggregated action distribution [nOutput x n_bins]
      raw_fitness - (np_array)     - raw fitness values (actual game reward) for each individual
      shaped_stats - (dict)        - aggregated shaped reward component statistics

    Return:
      data - (DataGatherer) - updated run data
    """
    data.gatherData(
        neat.pop, neat.species, action_dist=action_dist, raw_fitness=raw_fitness,
        shaped_stats=shaped_stats
    )
    if (gen % hyp["save_mod"]) == 0:
        data = checkBest(data)
        data.save(gen)

    if savePop is True:  # Get a sample pop to play with in notebooks
        global fileName
        pref = "log/" + fileName
        import pickle

        with open(pref + "_pop.obj", "wb") as fp:
            pickle.dump(neat.pop, fp)

    return data


def checkBest(data):
    """Checks better performing individual if it performs over many trials.
    Test a new 'best' individual with many different seeds to see if it really
    outperforms the current best.

    Args:
      data - (DataGatherer) - collected run data

    Return:
      data - (DataGatherer) - collected run data with best individual updated


    * This is a bit hacky, but is only for data gathering, and not optimization
    """
    global filename, hyp
    if data.newBest is True:
        bestReps = max(hyp["bestReps"], (nWorker - 1))
        rep = np.tile(data.best[-1], bestReps)
        fitVector, _, _, _ = batchMpiEval(
            rep, sameSeedForEachIndividual=False, track_actions=False
        )
        trueFit = np.mean(fitVector)
        if trueFit > data.best[-2].fitness:  # Actually better!
            data.best[-1].fitness = trueFit
            data.fit_top[-1] = trueFit
            data.bestFitVec = fitVector
        else:  # Just lucky! Revert to previous best
            data.best[-1] = data.best[-2]
            data.fit_top[-1] = data.fit_top[-2]
            data.newBest = False
    return data


# -- Parallelization ----------------------------------------------------- -- #
def batchMpiEval(
    pop: list[Ind], sameSeedForEachIndividual: bool = True, track_actions: bool = False
) -> tuple:
    """Sends population to workers for evaluation one batch at a time.

    Args:
      pop - [Ind] - list of individuals
        .wMat - (np_array) - weight matrix of network
                [N X N]
        .aVec - (np_array) - activation function of each node
                [N X 1]
      track_actions - (bool) - whether to track action distributions

    Return:
      reward  - (np_array) - total fitness value of each individual (shaped)
                [N X 1]
      action_dist - (np_array) - aggregated action distribution (if track_actions=True)
                [nOutput X n_bins] or None
      raw_reward - (np_array) - raw fitness value of each individual (actual game reward)
                [N X 1]
      pop_stats - (dict) - aggregated population statistics for curriculum learning
                {'avg_touches': float, 'avg_rallies_won': float, 'avg_rallies_lost': float}

    Todo:
      * Asynchronous evaluation instead of batches
    """
    global nWorker, hyp
    nSlave = nWorker - 1
    nJobs = len(pop)
    nBatch = math.ceil(nJobs / nSlave)  # First worker is master

    # Set same seed for each individual
    if sameSeedForEachIndividual is False:
        seed = np.random.randint(1000, size=nJobs)
    else:
        seed = np.random.randint(1000)

    reward = np.empty(nJobs, dtype=np.float64)
    raw_reward = np.empty(nJobs, dtype=np.float64)  # Track raw fitness

    # Initialize action distribution aggregator
    action_dist_agg = None

    # Aggregate episode statistics for curriculum learning and shaped reward tracking
    total_touches = 0
    total_rallies_won = 0
    total_rallies_lost = 0
    total_ball_time_opponent_side = 0
    total_tracking_reward = 0.0
    total_episodes = 0

    i = 0  # Index of fitness we are filling
    for iBatch in range(nBatch):  # Send one batch of individuals
        for iWork in range(nSlave):  # (one to each worker if there)
            if i < nJobs:
                wVec = pop[i].wMat.flatten()
                n_wVec = np.shape(wVec)[0]
                aVec = pop[i].aVec.flatten()
                n_aVec = np.shape(aVec)[0]

                # Why this structure?
                # Because MPI needs to know array size first.
                comm.send(n_wVec, dest=(iWork) + 1, tag=1)
                comm.Send(wVec, dest=(iWork) + 1, tag=2)
                comm.send(n_aVec, dest=(iWork) + 1, tag=3)
                comm.Send(aVec, dest=(iWork) + 1, tag=4)
                if sameSeedForEachIndividual is False:
                    comm.send(seed.item(i), dest=(iWork) + 1, tag=5)
                else:
                    comm.send(seed, dest=(iWork) + 1, tag=5)
                # Send track_actions flag
                comm.send(track_actions, dest=(iWork) + 1, tag=6)

            else:  # message size of 0 is signal to skip (no work for this worker in batch)
                n_wVec = 0
                comm.send(
                    n_wVec, dest=(iWork) + 1, tag=1
                )  # Must use tag=1 to match worker's recv
            i = i + 1

        # Get fitness values back for that batch
        i -= nSlave
        for iWork in range(1, nSlave + 1):
            if i < nJobs:
                workResult = np.empty(1, dtype="d")
                comm.Recv(workResult, source=iWork)
                reward[i] = workResult[0]

                # Receive action distribution if tracking
                if track_actions:
                    # First receive the shape
                    action_dist_shape = comm.recv(source=iWork, tag=7)
                    action_dist = np.empty(action_dist_shape, dtype="d")
                    comm.Recv(action_dist, source=iWork, tag=8)

                    # Aggregate action distributions
                    if action_dist_agg is None:
                        action_dist_agg = action_dist.copy()
                    else:
                        action_dist_agg += action_dist

                # Receive episode statistics (always sent by worker if available)
                try:
                    ep_stats = comm.recv(source=iWork, tag=9)
                    total_touches += ep_stats.get("ball_touches", 0)
                    total_rallies_won += ep_stats.get("rallies_won", 0)
                    total_rallies_lost += ep_stats.get("rallies_lost", 0)
                    total_ball_time_opponent_side += ep_stats.get("ball_time_opponent_side", 0)
                    total_tracking_reward += ep_stats.get("tracking_reward", 0.0)
                    total_episodes += 1
                except:
                    # Worker may not send stats if environment doesn't support it
                    pass

                # Receive raw fitness
                raw_result = np.empty(1, dtype="d")
                comm.Recv(raw_result, source=iWork, tag=10)
                raw_reward[i] = raw_result[0]
            i += 1

    # Normalize aggregated action distribution
    if track_actions and action_dist_agg is not None:
        # Handle both discrete (1D) and continuous (2D) action distributions
        if action_dist_agg.ndim == 1:
            # Discrete actions: 1D array [nActions]
            total = np.sum(action_dist_agg)
            if total == 0:
                total = 1
            action_dist_agg = action_dist_agg / total
        else:
            # Continuous actions: 2D array [nOutput x n_bins]
            total = np.sum(action_dist_agg, axis=1, keepdims=True)
            total[total == 0] = 1
            action_dist_agg = action_dist_agg / total

    # Compute population statistics for curriculum learning and shaped reward tracking
    pop_stats = {
        "avg_touches": total_touches / max(total_episodes, 1),
        "avg_rallies_won": total_rallies_won / max(total_episodes, 1),
        "avg_rallies_lost": total_rallies_lost / max(total_episodes, 1),
        "avg_ball_time_opponent_side": total_ball_time_opponent_side / max(total_episodes, 1),
        "avg_tracking_reward": total_tracking_reward / max(total_episodes, 1),
    }

    if track_actions:
        return reward, action_dist_agg, pop_stats, raw_reward

    return reward, None, pop_stats, raw_reward


def batchMpiEvalSelfPlay(
    pop: list[Ind],
    sameSeedForEachIndividual: bool = True,
    track_actions: bool = False,
    verbose: bool = False,
) -> tuple:
    """
    Evaluate population with self-play support.

    Same as batchMpiEval but also collects episode statistics for curriculum learning.

    Returns:
      reward - total fitness values (shaped)
      action_dist - action distribution (if track_actions)
      pop_stats - aggregated population statistics
      raw_reward - raw fitness values (actual game reward)
    """
    global nWorker, hyp, opponent_archive, selfplay_config
    nSlave = nWorker - 1
    nJobs = len(pop)
    nBatch = math.ceil(nJobs / nSlave)

    if sameSeedForEachIndividual is False:
        seed = np.random.randint(1000, size=nJobs)
    else:
        seed = np.random.randint(1000)

    reward = np.empty(nJobs, dtype=np.float64)
    raw_reward = np.empty(nJobs, dtype=np.float64)  # Track raw fitness
    action_dist_agg = None

    # Aggregate episode statistics for curriculum learning and shaped reward tracking
    total_touches = 0
    total_rallies_won = 0
    total_rallies_lost = 0
    total_ball_time_opponent_side = 0
    total_tracking_reward = 0.0
    total_episodes = 0

    if verbose:
        print(
            f"[Master] Starting eval: {nJobs} jobs, {nSlave} workers, {nBatch} batches",
            flush=True,
        )

    i = 0
    for iBatch in range(nBatch):
        if verbose:
            print(
                f"[Master] Sending batch {iBatch+1}/{nBatch} (jobs {i}-{min(i+nSlave-1, nJobs-1)})",
                flush=True,
            )

        for iWork in range(nSlave):
            if i < nJobs:
                wVec = pop[i].wMat.flatten()
                n_wVec = np.shape(wVec)[0]
                aVec = pop[i].aVec.flatten()
                n_aVec = np.shape(aVec)[0]

                comm.send(n_wVec, dest=(iWork) + 1, tag=1)
                comm.Send(wVec, dest=(iWork) + 1, tag=2)
                comm.send(n_aVec, dest=(iWork) + 1, tag=3)
                comm.Send(aVec, dest=(iWork) + 1, tag=4)
                if sameSeedForEachIndividual is False:
                    comm.send(seed.item(i), dest=(iWork) + 1, tag=5)
                else:
                    comm.send(seed, dest=(iWork) + 1, tag=5)
                comm.send(track_actions, dest=(iWork) + 1, tag=6)
            else:
                n_wVec = 0
                comm.send(
                    n_wVec, dest=(iWork) + 1, tag=1
                )  # Fixed: added tag=1 for consistency
            i = i + 1

        # Get results back
        if verbose:
            print(
                f"[Master] Waiting for batch {iBatch+1}/{nBatch} results...", flush=True
            )

        i -= nSlave
        for iWork in range(1, nSlave + 1):
            if i < nJobs:
                if verbose:
                    print(
                        f"[Master] Waiting for worker {iWork} (job {i})...", flush=True
                    )

                workResult = np.empty(1, dtype="d")
                comm.Recv(workResult, source=iWork)
                reward[i] = workResult[0]

                if verbose:
                    print(
                        f"[Master] Got result from worker {iWork}: fitness={workResult[0]:.2f}",
                        flush=True,
                    )

                if track_actions:
                    action_dist_shape = comm.recv(source=iWork, tag=7)
                    action_dist = np.empty(action_dist_shape, dtype="d")
                    comm.Recv(action_dist, source=iWork, tag=8)

                    if action_dist_agg is None:
                        action_dist_agg = action_dist.copy()
                    else:
                        action_dist_agg += action_dist

                # Receive episode statistics
                ep_stats = comm.recv(source=iWork, tag=9)
                total_touches += ep_stats.get("ball_touches", 0)
                total_rallies_won += ep_stats.get("rallies_won", 0)
                total_rallies_lost += ep_stats.get("rallies_lost", 0)
                total_ball_time_opponent_side += ep_stats.get("ball_time_opponent_side", 0)
                total_tracking_reward += ep_stats.get("tracking_reward", 0.0)
                total_episodes += 1

                # Receive raw fitness
                raw_result = np.empty(1, dtype="d")
                comm.Recv(raw_result, source=iWork, tag=10)
                raw_reward[i] = raw_result[0]

            i += 1

    if verbose:
        print(f"[Master] All batches complete", flush=True)

    # Normalize action distribution
    if track_actions and action_dist_agg is not None:
        if action_dist_agg.ndim == 1:
            total = np.sum(action_dist_agg)
            action_dist_agg = action_dist_agg / max(total, 1)
        else:
            total = np.sum(action_dist_agg, axis=1, keepdims=True)
            total[total == 0] = 1
            action_dist_agg = action_dist_agg / total

    # Compute population statistics for curriculum learning and shaped reward tracking
    pop_stats = {
        "avg_touches": total_touches / max(total_episodes, 1),
        "avg_rallies_won": total_rallies_won / max(total_episodes, 1),
        "avg_rallies_lost": total_rallies_lost / max(total_episodes, 1),
        "avg_ball_time_opponent_side": total_ball_time_opponent_side / max(total_episodes, 1),
        "avg_tracking_reward": total_tracking_reward / max(total_episodes, 1),
    }

    return reward, action_dist_agg, pop_stats, raw_reward


def slave():
    """Evaluation process: evaluates networks sent from master process.

    PseudoArgs (recieved from master):
      wVec   - (np_array) - weight matrix as a flattened vector
               [1 X N**2]
      n_wVec - (int)      - length of weight vector (N**2)
      aVec   - (np_array) - activation function of each node
               [1 X N]    - stored as ints, see applyAct in ann.py
      n_aVec - (int)      - length of activation vector (N)
      seed   - (int)      - random seed (for consistency across workers)
      track_actions - (bool) - whether to track action distributions

    PseudoReturn (sent to master):
      result - (float)    - total fitness value of network (shaped)
      action_dist - (np_array) - action distribution (if track_actions=True)
      raw_result - (float)    - raw fitness value (actual game reward)
    """
    global hyp, selfplay_config, opponent_archive

    # Check for verbose mode (needed early for curriculum logging)
    verbose = os.environ.get("NEAT_VERBOSE", "0") == "1"

    # Check if self-play is enabled
    selfplay_enabled = hyp.get("selfplay_enabled", False)

    # Get task config and apply curriculum override if specified
    # This allows switching curriculum on/off via enable_curriculum config parameter
    game_config = games[hyp["task"]]
    enable_curriculum = hyp.get("enable_curriculum", None)
    if enable_curriculum is not None:
        if game_config.env_name == "SlimeVolley-Shaped-v0":
            if enable_curriculum:
                # Switch to curriculum version
                game_config = game_config._replace(
                    env_name="SlimeVolley-Shaped-Curriculum-v0"
                )
                if verbose:
                    print(
                        f"  [Worker] Curriculum ENABLED: Using SlimeVolley-Shaped-Curriculum-v0",
                        flush=True,
                    )
            # else: already using non-curriculum version
        elif game_config.env_name == "SlimeVolley-Shaped-Curriculum-v0":
            if not enable_curriculum:
                # Switch to non-curriculum version
                game_config = game_config._replace(env_name="SlimeVolley-Shaped-v0")
                if verbose:
                    print(
                        f"  [Worker] Curriculum DISABLED: Using SlimeVolley-Shaped-v0",
                        flush=True,
                    )

    if selfplay_enabled:
        from domain.task_gym_selfplay import SelfPlayGymTask

        # For self-play, curriculum is controlled by selfplay_enable_curriculum
        # but we also respect the global enable_curriculum if set
        selfplay_curriculum = hyp.get("selfplay_enable_curriculum", True)
        if enable_curriculum is not None:
            # Global enable_curriculum overrides selfplay setting
            selfplay_curriculum = enable_curriculum

        task = SelfPlayGymTask(
            game_config,
            nReps=hyp["alg_nReps"],
            eval_mode=hyp.get("selfplay_eval_mode", "mixed"),
            baseline_weight=hyp.get("selfplay_baseline_weight", 0.6),
            archive_weight=hyp.get("selfplay_archive_weight", 0.4),
            n_archive_opponents=hyp.get("selfplay_n_archive_opponents", 3),
            enable_curriculum=selfplay_curriculum,
        )
    else:
        task = GymTask(game_config, nReps=hyp["alg_nReps"])

    job_count = 0  # Debug counter

    # Evaluate any weight vectors sent this way
    while True:
        # Check for archive/curriculum updates (non-blocking)
        if selfplay_enabled:
            archive_updates = 0
            while comm.Iprobe(source=0, tag=100):
                msg_type, data = comm.recv(source=0, tag=100)
                if msg_type == "archive_update":
                    task.load_archive_snapshot(data)
                    archive_updates += 1
                    if verbose:
                        print(
                            f"[Worker {rank}] Received archive update #{archive_updates}, size={len(data)}",
                            flush=True,
                        )

        # Check for curriculum updates (for both self-play and non-self-play)
        curriculum_updates = 0
        while comm.Iprobe(source=0, tag=101):
            msg_type, data = comm.recv(source=0, tag=101)
            if msg_type == "curriculum_update":
                if selfplay_enabled:
                    task.set_curriculum_stage(data)
                elif hasattr(task, "env") and hasattr(task.env, "set_curriculum_stage"):
                    # Non-self-play: update environment curriculum directly
                    task.env.set_curriculum_stage(data)
                curriculum_updates += 1

        if curriculum_updates > 0 and verbose:
            if selfplay_enabled:
                print(
                    f"[Worker {rank}] Processed {curriculum_updates} curriculum update(s), stage: {task.current_stage}",
                    flush=True,
                )
            elif hasattr(task, "env") and hasattr(task.env, "current_stage"):
                print(
                    f"[Worker {rank}] Processed {curriculum_updates} curriculum update(s), stage: {task.env.current_stage}",
                    flush=True,
                )

        n_wVec = comm.recv(source=0, tag=1)
        if n_wVec > 0:
            job_count += 1
            wVec = np.empty(n_wVec, dtype="d")
            comm.Recv(wVec, source=0, tag=2)

            n_aVec = comm.recv(source=0, tag=3)
            aVec = np.empty(n_aVec, dtype="d")
            comm.Recv(aVec, source=0, tag=4)
            seed = comm.recv(source=0, tag=5)
            track_actions = comm.recv(source=0, tag=6)

            if track_actions:
                if verbose:
                    print(
                        f"[Worker {rank}] Starting job {job_count}, archive_size={len(task.archive) if selfplay_enabled else 0}",
                        flush=True,
                    )

                try:
                    fitness, action_dist, raw_fitness = task.getFitness(
                        wVec, aVec, track_actions=True
                    )
                except Exception as e:
                    print(
                        f"[Worker {rank}] ERROR in getFitness (job {job_count}): {e}",
                        flush=True,
                    )
                    import traceback

                    traceback.print_exc()
                    # Send dummy data to avoid deadlock
                    fitness = -999.0
                    raw_fitness = -999.0
                    action_dist = np.zeros(task.nOutput)

                if verbose:
                    print(
                        f"[Worker {rank}] Completed job {job_count}, fitness={fitness:.2f}, raw={raw_fitness:.2f}",
                        flush=True,
                    )

                result = np.array([fitness])
                comm.Send(result, dest=0)
                comm.send(action_dist.shape, dest=0, tag=7)
                comm.Send(action_dist, dest=0, tag=8)

                # Send episode stats (for both self-play and non-self-play curriculum)
                ep_stats = {}
                if hasattr(task, "env") and hasattr(task.env, "get_episode_stats"):
                    ep_stats = task.env.get_episode_stats()
                elif hasattr(task, "get_episode_stats"):
                    # Some task wrappers may have get_episode_stats
                    ep_stats = task.get_episode_stats()

                # Always send episode stats (empty dict if not available)
                comm.send(ep_stats, dest=0, tag=9)

                # Send raw fitness
                raw_result = np.array([raw_fitness])
                comm.Send(raw_result, dest=0, tag=10)
            else:
                # No action tracking
                try:
                    result = task.getFitness(wVec, aVec, track_actions=False)
                    if isinstance(result, tuple):
                        # Some tasks may return (fitness, raw_fitness) even without track_actions
                        fitness, raw_fitness = (
                            result[0],
                            result[1] if len(result) > 1 else result[0],
                        )
                    else:
                        fitness = result
                        # Try to get raw fitness from environment stats
                        raw_fitness = fitness  # Default: use fitness as raw
                        if hasattr(task, "env") and hasattr(
                            task.env, "get_episode_stats"
                        ):
                            ep_stats_temp = task.env.get_episode_stats()
                            raw_fitness = ep_stats_temp.get("raw_game_reward", fitness)
                except Exception as e:
                    print(
                        f"[Worker {rank}] ERROR in getFitness (job {job_count}): {e}",
                        flush=True,
                    )
                    import traceback

                    traceback.print_exc()
                    fitness = -999.0
                    raw_fitness = -999.0

                result = np.array([fitness])
                comm.Send(result, dest=0)

                # Send episode stats (for both self-play and non-self-play curriculum)
                ep_stats = {}
                if hasattr(task, "env") and hasattr(task.env, "get_episode_stats"):
                    ep_stats = task.env.get_episode_stats()
                elif hasattr(task, "get_episode_stats"):
                    ep_stats = task.get_episode_stats()

                # Always send episode stats (empty dict if not available)
                comm.send(ep_stats, dest=0, tag=9)

                # Send raw fitness
                raw_result = np.array([raw_fitness])
                comm.Send(raw_result, dest=0, tag=10)

                if verbose:
                    print(
                        f"[Worker {rank}] Completed job {job_count}, fitness={fitness:.2f}, raw={raw_fitness:.2f}",
                        flush=True,
                    )

        if n_wVec < 0:
            print("Worker # ", rank, " shutting down.")
            break


def stopAllWorkers():
    """Sends signal to all workers to shutdown."""
    global nWorker
    nSlave = nWorker - 1
    print("stopping workers")
    for iWork in range(nSlave):
        comm.send(-1, dest=(iWork) + 1, tag=1)


def mpi_fork(n):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    (from https://github.com/garymcintire/mpi_util/)
    """
    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", IN_MPI="1")
        print(["mpirun", "-np", str(n), sys.executable] + sys.argv)
        subprocess.check_call(
            ["mpirun", "-np", str(n), sys.executable] + ["-u"] + sys.argv, env=env
        )
        return "parent"
    else:
        global nWorker, rank
        nWorker = comm.Get_size()
        rank = comm.Get_rank()
        return "child"


# -- Input Parsing ------------------------------------------------------- -- #


def main(argv):
    """Handles command line input, launches optimization or evaluation script
    depending on MPI rank.
    """
    global fileName, hyp  # Used by both master and slave processes
    fileName = args.outPrefix
    hyp_default = args.default
    hyp_adjust = args.hyperparam

    hyp = loadHyp(pFileName=hyp_default)
    updateHyp(hyp, hyp_adjust)

    # Apply command-line rendering overrides (only affects master)
    if args.render:
        hyp["render_enabled"] = True
    if args.no_render:
        hyp["render_enabled"] = False
    if args.render_interval is not None:
        hyp["render_interval"] = args.render_interval
        if args.render_interval > 0:
            hyp["render_enabled"] = True
    if args.render_fps is not None:
        hyp["render_fps"] = args.render_fps

    # Launch main thread and workers
    if rank == 0:
        master()
    else:
        slave()


if __name__ == "__main__":
    """Parse input and launch"""
    parser = argparse.ArgumentParser(description=("Evolve NEAT networks"))

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
        "-o",
        "--outPrefix",
        type=str,
        help="file name for result output",
        default="test",
    )

    parser.add_argument(
        "-n", "--num_worker", type=int, help="number of cores to use", default=8
    )

    # Rendering options (override config file)
    parser.add_argument(
        "--render", action="store_true", help="enable rendering during training"
    )

    parser.add_argument(
        "--no-render", action="store_true", help="disable rendering during training"
    )

    parser.add_argument(
        "--render-interval",
        type=int,
        default=None,
        help="render every N generations (e.g., 10)",
    )

    parser.add_argument(
        "--render-fps", type=int, default=None, help="rendering frames per second"
    )

    args = parser.parse_args()

    # Use MPI if parallel
    if "parent" == mpi_fork(args.num_worker + 1):
        os._exit(0)

    main(args)
