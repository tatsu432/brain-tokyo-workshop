"""
Self-Play Task for NEAT Training on SlimeVolley

This module extends the basic GymTask to support:
1. Self-play with opponent archive
2. Curriculum learning (progressive difficulty)
3. Evaluation against both baseline and archived opponents

Key Features:
- Archive Management: Stores best individuals from past generations
- Mixed Evaluation: Evaluates against baseline + random archived opponents
- Curriculum Stages: 'touch' → 'rally' → 'win' progression
- Statistics Tracking: Ball touches, rallies, win rates

Self-Play Strategy:
- Each individual is evaluated against:
  1. The hard baseline opponent (fixed difficulty reference)
  2. Random opponents from the archive (co-evolution pressure)
- This creates automatic curriculum: agents improve together

Usage:
    from domain.task_gym_selfplay import SelfPlayGymTask

    # In worker:
    task = SelfPlayGymTask(game_config, nReps=3)
    task.set_opponent_archive(archived_individuals)
    fitness = task.getFitness(wVec, aVec)
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from neat_src import act, selectAct

from domain.make_env import make_env


class OpponentPolicy:
    """
    Wrapper that turns NEAT individual (wVec, aVec) into a policy.
    Used for self-play opponents from the archive.
    """

    def __init__(
        self,
        wVec: np.ndarray,
        aVec: np.ndarray,
        nInput: int,
        nOutput: int,
        actSelect: str,
    ):
        """
        Args:
            wVec: Flattened weight matrix
            aVec: Activation vector
            nInput: Number of input nodes
            nOutput: Number of output nodes
            actSelect: Action selection method ('prob', 'hard', 'all', etc.)
        """
        self.wVec = wVec.copy()
        self.aVec = aVec.copy()
        self.nInput = nInput
        self.nOutput = nOutput
        self.actSelect = actSelect

        # Clean weights
        self.wVec[np.isnan(self.wVec)] = 0

    def predict(self, obs: np.ndarray) -> Any:
        """
        Get action from observation.

        Args:
            obs: Observation from environment (12,)

        Returns:
            Action in format expected by environment
        """
        # Validate observation
        obs = np.asarray(obs, dtype=np.float64)
        obs[np.isnan(obs)] = 0
        obs[np.isinf(obs)] = 0

        annOut = act(self.wVec, self.aVec, self.nInput, self.nOutput, obs)

        # Validate output
        if np.any(np.isnan(annOut)) or np.any(np.isinf(annOut)):
            annOut = np.zeros(self.nOutput)

        action = selectAct(annOut, self.actSelect)
        return action

    def reset(self):
        """Reset any internal state (for compatibility)."""
        pass


class ArchivedIndividual:
    """
    Stores an individual from the archive with its metadata.
    """

    def __init__(
        self, wVec: np.ndarray, aVec: np.ndarray, fitness: float, generation: int
    ):
        self.wVec = wVec.copy()
        self.aVec = aVec.copy()
        self.fitness = fitness
        self.generation = generation

    def to_policy(self, nInput: int, nOutput: int, actSelect: str) -> OpponentPolicy:
        """Convert to a policy for evaluation."""
        return OpponentPolicy(self.wVec, self.aVec, nInput, nOutput, actSelect)


class SelfPlayGymTask:
    """
    Self-play task for NEAT with opponent archive and curriculum.

    Evaluation modes:
    1. BASELINE_ONLY: Only evaluate against built-in baseline
    2. ARCHIVE_ONLY: Only evaluate against archived opponents
    3. MIXED: Evaluate against both (default - recommended)

    The fitness is computed as weighted sum of performance against
    different opponent types, with baseline performance more heavily
    weighted to maintain consistent difficulty reference.
    """

    # Evaluation modes
    BASELINE_ONLY = "baseline"
    ARCHIVE_ONLY = "archive"
    MIXED = "mixed"

    def __init__(
        self,
        game,
        paramOnly: bool = False,
        nReps: int = 1,
        # Self-play configuration
        eval_mode: str = "survival",
        baseline_weight: float = 0.6,  # Weight for baseline eval
        archive_weight: float = 0.4,  # Weight for archive eval
        n_archive_opponents: int = 3,  # Number of archive opponents per eval
        # Curriculum configuration
        enable_curriculum: bool = True,
        touch_threshold: float = 5.0,  # Avg touches to advance from 'touch'
        rally_threshold: float = 0.0,  # Avg rally diff to advance from 'rally'
    ):
        """
        Initialize self-play task.

        Args:
            game: Game configuration from config.py
            paramOnly: Only load parameters (no env creation)
            nReps: Number of evaluation repetitions
            eval_mode: 'baseline', 'archive', or 'mixed'
            baseline_weight: Weight for baseline opponent evaluation
            archive_weight: Weight for archive opponent evaluation
            n_archive_opponents: Number of archive opponents to sample
            enable_curriculum: Whether to use curriculum learning
            touch_threshold: Average ball touches to advance from 'touch' stage
            rally_threshold: Average (wins-losses) to advance from 'rally' stage
        """
        # Network properties
        self.nInput = game.input_size
        self.nOutput = game.output_size
        self.actRange = game.h_act
        self.absWCap = game.weightCap
        self.layers = game.layers
        self.activations = np.r_[np.full(1, 1), game.i_act, game.o_act]

        # Environment
        self.nReps = nReps
        self.maxEpisodeLength = game.max_episode_length
        self.actSelect = game.actionSelect
        self.env_name = game.env_name

        if not paramOnly:
            self.env = make_env(game.env_name)
        else:
            self.env = None

        # Self-play configuration
        self.eval_mode = eval_mode
        self.baseline_weight = baseline_weight
        self.archive_weight = archive_weight
        self.n_archive_opponents = n_archive_opponents

        # Opponent archive
        self.archive: List[ArchivedIndividual] = []
        self.max_archive_size = 50

        # Curriculum learning
        self.enable_curriculum = enable_curriculum
        self.touch_threshold = touch_threshold
        self.rally_threshold = rally_threshold
        self.current_stage = "survival"  # 'survival', 'mixed', 'wins'

        # Statistics for curriculum advancement
        self.generation_stats = {
            "avg_touches": 0.0,
            "avg_rally_diff": 0.0,
            "avg_fitness": 0.0,
        }

        # Action distribution tracking
        self.n_action_bins = 4
        self.action_bin_edges = np.array([-np.inf, -0.5, 0.0, 0.5, np.inf])
        self.is_discrete_action = game.actionSelect in ["prob", "hard"]

    def set_curriculum_stage(self, stage: str):
        """Manually set curriculum stage."""
        if stage not in ["survival", "mixed", "wins"]:
            raise ValueError(f"Unknown stage: {stage}")

        self.current_stage = stage

        # Update environment curriculum if supported
        if hasattr(self.env, "set_curriculum_stage"):
            self.env.set_curriculum_stage(stage)

    def update_curriculum(self, population_stats: Dict[str, float]):
        """
        Check if we should advance to the next curriculum stage.

        Args:
            population_stats: Dict with 'avg_touches', 'avg_rally_diff', 'avg_fitness'
        """
        if not self.enable_curriculum:
            return

        self.generation_stats.update(population_stats)

        if self.current_stage == "survival":
            if population_stats.get("avg_touches", 0) >= self.touch_threshold:
                print("Curriculum: Advancing from 'survival' to 'mixed' stage")
                self.set_curriculum_stage("mixed")

        elif self.current_stage == "mixed":
            if population_stats.get("avg_rally_diff", 0) >= self.rally_threshold:
                print("Curriculum: Advancing from 'mixed' to 'wins' stage")
                self.set_curriculum_stage("wins")

    def add_to_archive(
        self, wVec: np.ndarray, aVec: np.ndarray, fitness: float, generation: int
    ):
        """
        Add an individual to the opponent archive.

        Only adds if fitness is better than minimum in archive (when full).
        """
        individual = ArchivedIndividual(wVec, aVec, fitness, generation)

        if len(self.archive) < self.max_archive_size:
            self.archive.append(individual)
        else:
            # Replace worst if new individual is better
            worst_idx = min(
                range(len(self.archive)), key=lambda i: self.archive[i].fitness
            )
            if fitness > self.archive[worst_idx].fitness:
                self.archive[worst_idx] = individual

    def set_opponent_archive(self, archive: List[ArchivedIndividual]):
        """Set the entire opponent archive (used for worker sync)."""
        self.archive = archive

    def get_archive_snapshot(self) -> List[Tuple[np.ndarray, np.ndarray, float, int]]:
        """
        Get a serializable snapshot of the archive for MPI transfer.

        Returns:
            List of (wVec, aVec, fitness, generation) tuples
        """
        return [
            (ind.wVec.copy(), ind.aVec.copy(), ind.fitness, ind.generation)
            for ind in self.archive
        ]

    def load_archive_snapshot(
        self, snapshot: List[Tuple[np.ndarray, np.ndarray, float, int]]
    ):
        """Load archive from snapshot (received via MPI)."""
        import os

        verbose = os.environ.get("NEAT_VERBOSE", "0") == "1"

        self.archive = []
        for i, (wVec, aVec, fitness, gen) in enumerate(snapshot):
            # Validate weights
            wVec_clean = np.copy(wVec)
            wVec_clean[np.isnan(wVec_clean)] = 0
            wVec_clean[np.isinf(wVec_clean)] = 0

            aVec_clean = np.copy(aVec)
            aVec_clean[np.isnan(aVec_clean)] = 0

            if verbose:
                nan_count = np.sum(np.isnan(wVec))
                inf_count = np.sum(np.isinf(wVec))
                if nan_count > 0 or inf_count > 0:
                    print(
                        f"  [load_archive] WARNING: Archive individual {i} has {nan_count} NaN and {inf_count} inf values in wVec",
                        flush=True,
                    )

            self.archive.append(
                ArchivedIndividual(wVec_clean, aVec_clean, fitness, gen)
            )

    def _sample_archive_opponents(self, n: int) -> List[OpponentPolicy]:
        """Sample n opponents from the archive."""
        if not self.archive:
            return []

        # Sample without replacement if possible
        n = min(n, len(self.archive))
        sampled = random.sample(self.archive, n)

        return [
            ind.to_policy(self.nInput, self.nOutput, self.actSelect) for ind in sampled
        ]

    def getFitness(
        self,
        wVec: np.ndarray,
        aVec: np.ndarray,
        hyp: Optional[dict] = None,
        view: bool = False,
        nRep: Optional[int] = None,
        seed: int = -1,
        track_actions: bool = False,
        debug: bool = False,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Get fitness of an individual with self-play evaluation.

        Returns weighted combination of:
        1. Performance against baseline opponent
        2. Performance against archived opponents

        Returns:
            fitness - (float) - total fitness (actual + shaped reward)
            OR (if track_actions=True):
            (fitness, action_dist, raw_fitness) - tuple with:
              fitness     - (float)    - total fitness (actual + shaped)
              action_dist - (np_array) - action distribution
              raw_fitness - (float)    - raw fitness (actual game reward only)
        """
        if nRep is None:
            nRep = self.nReps

        wVec = np.copy(wVec)
        wVec[np.isnan(wVec)] = 0

        # Initialize action tracking if needed
        action_dist = None
        if track_actions:
            if self.is_discrete_action:
                action_dist = np.zeros(self.nOutput)
            else:
                action_dist = np.zeros((self.nOutput, self.n_action_bins))

        # Collect episode statistics
        all_stats = []
        rewards = []
        raw_rewards = []  # Track raw (unshaped) rewards

        # Determine evaluation setup
        if self.eval_mode == self.BASELINE_ONLY or not self.archive:
            # Only baseline evaluation
            baseline_reps = nRep
            archive_reps = 0
        elif self.eval_mode == self.ARCHIVE_ONLY and self.archive:
            # Only archive evaluation
            baseline_reps = 0
            archive_reps = nRep
        else:  # MIXED
            # Split between baseline and archive
            baseline_reps = max(1, int(nRep * self.baseline_weight))
            archive_reps = nRep - baseline_reps

        # Check for verbose mode
        import os

        verbose = os.environ.get("NEAT_VERBOSE", "0") == "1"

        if verbose or debug:
            print(
                f"  [getFitness] baseline_reps={baseline_reps}, archive_reps={archive_reps}, archive_size={len(self.archive)}",
                flush=True,
            )

        # Evaluate against baseline
        for i in range(baseline_reps):
            if debug:
                print(
                    f"  [getFitness] Starting baseline episode {i + 1}/{baseline_reps}",
                    flush=True,
                )

            result = self._evaluate_episode(
                wVec,
                aVec,
                opponent_policy=None,  # Use built-in baseline
                view=view,
                seed=seed + i if seed >= 0 else -1,
                track_actions=track_actions,
            )

            if debug:
                print(
                    f"  [getFitness] Completed baseline episode {i + 1}/{baseline_reps}",
                    flush=True,
                )

            if track_actions:
                reward, ep_action_dist, stats, raw_reward = result
                action_dist += ep_action_dist
            else:
                reward, stats, raw_reward = result

            rewards.append(reward)
            raw_rewards.append(raw_reward)
            all_stats.append(stats)

        # Evaluate against archive opponents
        if archive_reps > 0 and self.archive:
            archive_opponents = self._sample_archive_opponents(archive_reps)

            if debug:
                print(
                    f"  [getFitness] Sampled {len(archive_opponents)} archive opponents for {archive_reps} reps",
                    flush=True,
                )

            for i, opponent in enumerate(archive_opponents):
                if debug:
                    print(
                        f"  [getFitness] Starting archive episode {i + 1}/{len(archive_opponents)}",
                        flush=True,
                    )

                result = self._evaluate_episode(
                    wVec,
                    aVec,
                    opponent_policy=opponent,
                    view=view,
                    seed=seed + baseline_reps + i if seed >= 0 else -1,
                    track_actions=track_actions,
                )

                if debug:
                    print(
                        f"  [getFitness] Completed archive episode {i + 1}/{len(archive_opponents)}",
                        flush=True,
                    )

                if track_actions:
                    reward, ep_action_dist, stats, raw_reward = result
                    action_dist += ep_action_dist
                else:
                    reward, stats, raw_reward = result

                rewards.append(reward)
                raw_rewards.append(raw_reward)
                all_stats.append(stats)

        # Compute weighted fitness (both total and raw)
        if archive_reps > 0:
            baseline_fitness = (
                np.mean(rewards[:baseline_reps]) if baseline_reps > 0 else 0
            )
            archive_fitness = np.mean(rewards[baseline_reps:])
            fitness = (
                self.baseline_weight * baseline_fitness
                + self.archive_weight * archive_fitness
            )

            baseline_raw_fitness = (
                np.mean(raw_rewards[:baseline_reps]) if baseline_reps > 0 else 0
            )
            archive_raw_fitness = np.mean(raw_rewards[baseline_reps:])
            raw_fitness = (
                self.baseline_weight * baseline_raw_fitness
                + self.archive_weight * archive_raw_fitness
            )
        else:
            fitness = np.mean(rewards)
            raw_fitness = np.mean(raw_rewards)

        # Normalize action distribution
        if track_actions and action_dist is not None:
            if self.is_discrete_action:
                total = np.sum(action_dist)
                action_dist = action_dist / max(total, 1)
            else:
                total = np.sum(action_dist, axis=1, keepdims=True)
                total[total == 0] = 1
                action_dist = action_dist / total

        if track_actions:
            return fitness, action_dist, raw_fitness
        return fitness

    def _evaluate_episode(
        self,
        wVec: np.ndarray,
        aVec: np.ndarray,
        opponent_policy: Optional[OpponentPolicy] = None,
        view: bool = False,
        seed: int = -1,
        track_actions: bool = False,
        debug: bool = False,
    ) -> Tuple[float, Any, Dict, float]:
        """
        Evaluate a single episode.

        Args:
            wVec, aVec: Network weights
            opponent_policy: If None, use built-in baseline
            view: Whether to render
            seed: Random seed
            track_actions: Track action distribution

        Returns:
            (reward, action_dist, episode_stats, raw_reward) if track_actions
            (reward, episode_stats, raw_reward) otherwise
        """
        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)
            self.env.seed(seed)

        # Initialize action tracking
        action_dist = None
        if track_actions:
            if self.is_discrete_action:
                action_dist = np.zeros(self.nOutput)
            else:
                action_dist = np.zeros((self.nOutput, self.n_action_bins))

        # Track raw (unshaped) reward separately
        totalRawReward = 0.0

        # Reset environment and opponent
        try:
            state = self.env.reset()
        except Exception as e:
            print(f"    [_evaluate_episode] ERROR in env.reset(): {e}", flush=True)
            raise

        self.env.t = 0

        if opponent_policy is not None:
            opponent_policy.reset()

        # Get initial action
        annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
        action = selectAct(annOut, self.actSelect)

        if track_actions:
            action_dist = self._update_action_dist(action_dist, action)

        # Get opponent action
        if opponent_policy is not None:
            # Get opponent's observation (from info or mirror)
            opp_obs = self._get_opponent_obs(state)
            opp_action = opponent_policy.predict(opp_obs)
        else:
            opp_action = None

        # Step environment
        try:
            if opp_action is not None:
                state, reward, done, info = self.env.step(action, opp_action)
            else:
                state, reward, done, info = self.env.step(action)
        except Exception as e:
            print(
                f"    [_evaluate_episode] ERROR in initial env.step(): {e}", flush=True
            )
            raise

        totalReward = reward
        # Extract raw game reward if available (for shaped environments)
        raw_reward = info.get("game_reward", reward)
        totalRawReward += raw_reward

        if view:
            self.env.render()

        # Episode loop
        import os

        verbose = os.environ.get("NEAT_VERBOSE", "0") == "1"
        log_interval = 500  # Log every N steps

        for tStep in range(self.maxEpisodeLength):
            # Progress logging for long episodes
            if verbose and tStep > 0 and tStep % log_interval == 0:
                print(
                    f"    [_evaluate_episode] Step {tStep}/{self.maxEpisodeLength}, reward so far: {totalReward:.2f}, opponent: {'archive' if opponent_policy else 'baseline'}",
                    flush=True,
                )

            annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
            action = selectAct(annOut, self.actSelect)

            if track_actions:
                action_dist = self._update_action_dist(action_dist, action)

            # Get opponent action
            if opponent_policy is not None:
                opp_obs = self._get_opponent_obs(state, info)
                opp_action = opponent_policy.predict(opp_obs)
            else:
                opp_action = None

            try:
                if opp_action is not None:
                    state, reward, done, info = self.env.step(action, opp_action)
                else:
                    state, reward, done, info = self.env.step(action)
            except Exception as e:
                print(
                    f"    [_evaluate_episode] ERROR in env.step() at step {tStep}: {e}",
                    flush=True,
                )
                raise

            totalReward += reward
            # Extract raw game reward if available
            raw_reward = info.get("game_reward", reward)
            totalRawReward += raw_reward

            if view:
                self.env.render()

            if done:
                if verbose and tStep > log_interval:
                    print(
                        f"    [_evaluate_episode] Episode done at step {tStep}, total reward: {totalReward:.2f}",
                        flush=True,
                    )
                break

        # Get episode statistics
        episode_stats = info.get("episode_stats", {})

        if track_actions:
            return totalReward, action_dist, episode_stats, totalRawReward
        return totalReward, episode_stats, totalRawReward

    def _get_opponent_obs(
        self, state: np.ndarray, info: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Get observation for opponent agent.

        SlimeVolley provides 'otherObs' in info, or we can mirror the state.
        """
        if info is not None and "otherObs" in info:
            return info["otherObs"]

        # Mirror the observation manually if needed
        # State: [agent_x, agent_y, agent_vx, agent_vy,
        #         ball_x, ball_y, ball_vx, ball_vy,
        #         opp_x, opp_y, opp_vx, opp_vy]
        # For opponent, swap agent and opp, negate x values
        mirrored = np.array(
            [
                state[8],  # opp_x becomes agent_x
                state[9],  # opp_y
                -state[10],  # opp_vx (negated)
                state[11],  # opp_vy
                -state[4],  # ball_x (negated)
                state[5],  # ball_y
                -state[6],  # ball_vx (negated)
                state[7],  # ball_vy
                state[0],  # agent_x becomes opp_x
                state[1],  # agent_y
                -state[2],  # agent_vx (negated)
                state[3],  # agent_vy
            ]
        )
        return mirrored

    def _update_action_dist(self, action_dist: np.ndarray, action) -> np.ndarray:
        """Update action distribution with new action."""
        if self.is_discrete_action:
            if isinstance(action, (int, np.integer)):
                action_idx = int(action)
            else:
                action_idx = int(np.atleast_1d(action).flatten()[0])
            if 0 <= action_idx < len(action_dist):
                action_dist[action_idx] += 1
        else:
            action_arr = np.atleast_1d(action).flatten()
            for i, act_val in enumerate(action_arr[: self.nOutput]):
                bin_idx = np.digitize(act_val, self.action_bin_edges[1:-1])
                action_dist[i, bin_idx] += 1

        return action_dist


# Convenience function for creating self-play task
def create_selfplay_task(game, nReps=3, **kwargs) -> SelfPlayGymTask:
    """
    Create a SelfPlayGymTask with sensible defaults.

    Args:
        game: Game configuration from config.py
        nReps: Number of evaluation repetitions
        **kwargs: Additional arguments for SelfPlayGymTask

    Returns:
        Configured SelfPlayGymTask
    """
    return SelfPlayGymTask(
        game,
        nReps=nReps,
        eval_mode="survival",
        baseline_weight=0.6,
        archive_weight=0.4,
        n_archive_opponents=3,
        enable_curriculum=True,
        **kwargs,
    )
