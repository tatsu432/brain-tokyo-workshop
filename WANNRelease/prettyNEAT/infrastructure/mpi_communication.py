"""MPI communication for parallel evaluation."""
import math
import os
import numpy as np
from typing import Tuple, Optional, Dict, List
from mpi4py import MPI

from neat_src.ind import Ind
from core.setup import suppress_stderr

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def batch_mpi_eval(
    pop: List[Ind],
    n_worker: int,
    hyp: dict,
    same_seed_for_each_individual: bool = True,
    track_actions: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float], np.ndarray]:
    """Sends population to workers for evaluation one batch at a time.

    Args:
      pop - [Ind] - list of individuals
        .wMat - (np_array) - weight matrix of network [N X N]
        .aVec - (np_array) - activation function of each node [N X 1]
      n_worker - (int) - number of workers
      hyp - (dict) - hyperparameters
      same_seed_for_each_individual - (bool) - use same seed for all individuals
      track_actions - (bool) - whether to track action distributions

    Return:
      reward  - (np_array) - total fitness value of each individual (shaped) [N X 1]
      action_dist - (np_array) - aggregated action distribution (if track_actions=True)
                [nOutput X n_bins] or None
      raw_reward - (np_array) - raw fitness value of each individual (actual game reward) [N X 1]
      pop_stats - (dict) - aggregated population statistics for curriculum learning
                {'avg_touches': float, 'avg_rallies_won': float, 'avg_rallies_lost': float}
    """
    n_slave = n_worker - 1
    n_jobs = len(pop)
    n_batch = math.ceil(n_jobs / n_slave)  # First worker is master

    # Set same seed for each individual
    if same_seed_for_each_individual is False:
        seed = np.random.randint(1000, size=n_jobs)
    else:
        seed = np.random.randint(1000)

    reward = np.empty(n_jobs, dtype=np.float64)
    raw_reward = np.empty(n_jobs, dtype=np.float64)  # Track raw fitness

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
    for i_batch in range(n_batch):  # Send one batch of individuals
        for i_work in range(n_slave):  # (one to each worker if there)
            if i < n_jobs:
                w_vec = pop[i].wMat.flatten()
                n_w_vec = np.shape(w_vec)[0]
                a_vec = pop[i].aVec.flatten()
                n_a_vec = np.shape(a_vec)[0]

                # Why this structure?
                # Because MPI needs to know array size first.
                comm.send(n_w_vec, dest=(i_work) + 1, tag=1)
                comm.Send(w_vec, dest=(i_work) + 1, tag=2)
                comm.send(n_a_vec, dest=(i_work) + 1, tag=3)
                comm.Send(a_vec, dest=(i_work) + 1, tag=4)
                if same_seed_for_each_individual is False:
                    comm.send(seed.item(i), dest=(i_work) + 1, tag=5)
                else:
                    comm.send(seed, dest=(i_work) + 1, tag=5)
                # Send track_actions flag
                comm.send(track_actions, dest=(i_work) + 1, tag=6)

            else:  # message size of 0 is signal to skip (no work for this worker in batch)
                n_w_vec = 0
                comm.send(
                    n_w_vec, dest=(i_work) + 1, tag=1
                )  # Must use tag=1 to match worker's recv
            i = i + 1

        # Get fitness values back for that batch
        i -= n_slave
        for i_work in range(1, n_slave + 1):
            if i < n_jobs:
                work_result = np.empty(1, dtype="d")
                comm.Recv(work_result, source=i_work)
                reward[i] = work_result[0]

                # Receive action distribution if tracking
                if track_actions:
                    # First receive the shape
                    action_dist_shape = comm.recv(source=i_work, tag=7)
                    action_dist = np.empty(action_dist_shape, dtype="d")
                    comm.Recv(action_dist, source=i_work, tag=8)

                    # Aggregate action distributions
                    if action_dist_agg is None:
                        action_dist_agg = action_dist.copy()
                    else:
                        action_dist_agg += action_dist

                # Receive episode statistics (always sent by worker if available)
                try:
                    ep_stats = comm.recv(source=i_work, tag=9)
                    total_touches += ep_stats.get("ball_touches", 0)
                    total_rallies_won += ep_stats.get("rallies_won", 0)
                    total_rallies_lost += ep_stats.get("rallies_lost", 0)
                    total_ball_time_opponent_side += ep_stats.get(
                        "ball_time_opponent_side", 0
                    )
                    total_tracking_reward += ep_stats.get("tracking_reward", 0.0)
                    total_episodes += 1
                except:
                    # Worker may not send stats if environment doesn't support it
                    pass

                # Receive raw fitness
                raw_result = np.empty(1, dtype="d")
                comm.Recv(raw_result, source=i_work, tag=10)
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
        "avg_ball_time_opponent_side": total_ball_time_opponent_side
        / max(total_episodes, 1),
        "avg_tracking_reward": total_tracking_reward / max(total_episodes, 1),
    }

    if track_actions:
        return reward, action_dist_agg, pop_stats, raw_reward

    return reward, None, pop_stats, raw_reward


def batch_mpi_eval_selfplay(
    pop: List[Ind],
    n_worker: int,
    hyp: dict,
    same_seed_for_each_individual: bool = True,
    track_actions: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float], np.ndarray]:
    """
    Evaluate population with self-play support.

    Same as batch_mpi_eval but also collects episode statistics for curriculum learning.

    Returns:
      reward - total fitness values (shaped)
      action_dist - action distribution (if track_actions)
      pop_stats - aggregated population statistics
      raw_reward - raw fitness values (actual game reward)
    """
    n_slave = n_worker - 1
    n_jobs = len(pop)
    n_batch = math.ceil(n_jobs / n_slave)

    if same_seed_for_each_individual is False:
        seed = np.random.randint(1000, size=n_jobs)
    else:
        seed = np.random.randint(1000)

    reward = np.empty(n_jobs, dtype=np.float64)
    raw_reward = np.empty(n_jobs, dtype=np.float64)  # Track raw fitness
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
            f"[Master] Starting eval: {n_jobs} jobs, {n_slave} workers, {n_batch} batches",
            flush=True,
        )

    i = 0
    for i_batch in range(n_batch):
        if verbose:
            print(
                f"[Master] Sending batch {i_batch+1}/{n_batch} (jobs {i}-{min(i+n_slave-1, n_jobs-1)})",
                flush=True,
            )

        for i_work in range(n_slave):
            if i < n_jobs:
                w_vec = pop[i].wMat.flatten()
                n_w_vec = np.shape(w_vec)[0]
                a_vec = pop[i].aVec.flatten()
                n_a_vec = np.shape(a_vec)[0]

                comm.send(n_w_vec, dest=(i_work) + 1, tag=1)
                comm.Send(w_vec, dest=(i_work) + 1, tag=2)
                comm.send(n_a_vec, dest=(i_work) + 1, tag=3)
                comm.Send(a_vec, dest=(i_work) + 1, tag=4)
                if same_seed_for_each_individual is False:
                    comm.send(seed.item(i), dest=(i_work) + 1, tag=5)
                else:
                    comm.send(seed, dest=(i_work) + 1, tag=5)
                comm.send(track_actions, dest=(i_work) + 1, tag=6)
            else:
                n_w_vec = 0
                comm.send(
                    n_w_vec, dest=(i_work) + 1, tag=1
                )  # Fixed: added tag=1 for consistency
            i = i + 1

        # Get results back
        if verbose:
            print(
                f"[Master] Waiting for batch {i_batch+1}/{n_batch} results...", flush=True
            )

        i -= n_slave
        for i_work in range(1, n_slave + 1):
            if i < n_jobs:
                if verbose:
                    print(
                        f"[Master] Waiting for worker {i_work} (job {i})...", flush=True
                    )

                work_result = np.empty(1, dtype="d")
                comm.Recv(work_result, source=i_work)
                reward[i] = work_result[0]

                if verbose:
                    print(
                        f"[Master] Got result from worker {i_work}: fitness={work_result[0]:.2f}",
                        flush=True,
                    )

                if track_actions:
                    action_dist_shape = comm.recv(source=i_work, tag=7)
                    action_dist = np.empty(action_dist_shape, dtype="d")
                    comm.Recv(action_dist, source=i_work, tag=8)

                    if action_dist_agg is None:
                        action_dist_agg = action_dist.copy()
                    else:
                        action_dist_agg += action_dist

                # Receive episode statistics
                ep_stats = comm.recv(source=i_work, tag=9)
                total_touches += ep_stats.get("ball_touches", 0)
                total_rallies_won += ep_stats.get("rallies_won", 0)
                total_rallies_lost += ep_stats.get("rallies_lost", 0)
                total_ball_time_opponent_side += ep_stats.get("ball_time_opponent_side", 0)
                total_tracking_reward += ep_stats.get("tracking_reward", 0.0)
                total_episodes += 1

                # Receive raw fitness
                raw_result = np.empty(1, dtype="d")
                comm.Recv(raw_result, source=i_work, tag=10)
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
        "avg_ball_time_opponent_side": total_ball_time_opponent_side
        / max(total_episodes, 1),
        "avg_tracking_reward": total_tracking_reward / max(total_episodes, 1),
    }

    return reward, action_dist_agg, pop_stats, raw_reward


def run_worker(hyp: dict):
    """Evaluation process: evaluates networks sent from master process.

    PseudoArgs (received from master):
      wVec   - (np_array) - weight matrix as a flattened vector [1 X N**2]
      n_wVec - (int)      - length of weight vector (N**2)
      aVec   - (np_array) - activation function of each node [1 X N]
      n_aVec - (int)      - length of activation vector (N)
      seed   - (int)      - random seed (for consistency across workers)
      track_actions - (bool) - whether to track action distributions

    PseudoReturn (sent to master):
      result - (float)    - total fitness value of network (shaped)
      action_dist - (np_array) - action distribution (if track_actions=True)
      raw_result - (float)    - raw fitness value (actual game reward)
    """
    # Check for verbose mode (needed early for curriculum logging)
    verbose = os.environ.get("NEAT_VERBOSE", "0") == "1"

    # Check if self-play is enabled
    selfplay_enabled = hyp.get("selfplay_enabled", False)

    # Get task config and apply curriculum override if specified
    # This allows switching curriculum on/off via enable_curriculum config parameter
    with suppress_stderr():
        from domain import games

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
        from domain.task_gym import GymTask

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

        n_w_vec = comm.recv(source=0, tag=1)
        if n_w_vec > 0:
            job_count += 1
            w_vec = np.empty(n_w_vec, dtype="d")
            comm.Recv(w_vec, source=0, tag=2)

            n_a_vec = comm.recv(source=0, tag=3)
            a_vec = np.empty(n_a_vec, dtype="d")
            comm.Recv(a_vec, source=0, tag=4)
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
                        w_vec, a_vec, track_actions=True
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
                    result = task.getFitness(w_vec, a_vec, track_actions=False)
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

        if n_w_vec < 0:
            print("Worker # ", rank, " shutting down.")
            break


def stop_all_workers(n_worker: int):
    """Sends signal to all workers to shutdown."""
    n_slave = n_worker - 1
    print("stopping workers")
    for i_work in range(n_slave):
        comm.send(-1, dest=(i_work) + 1, tag=1)


def mpi_fork(n: int):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    (from https://github.com/garymcintire/mpi_util/)
    """
    import sys
    import subprocess

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
        return "child"
