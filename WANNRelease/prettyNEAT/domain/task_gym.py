import random
import numpy as np
import sys
import time
from domain.make_env import make_env
from neat_src import *


class GymTask:
    """Problem domain to be solved by neural network. Uses OpenAI Gym patterns."""

    def __init__(self, game, paramOnly=False, nReps=1):
        """Initializes task environment

        Args:
          game - (string) - dict key of task to be solved (see domain/config.py)

        Optional:
          paramOnly - (bool)  - only load parameters instead of launching task?
          nReps     - (nReps) - number of trials to get average fitness
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
        if not paramOnly:
            self.env = make_env(game.env_name)

        # Special needs...
        self.needsClosed = game.env_name.startswith("CartPoleSwingUp")

        # Action distribution tracking
        # For continuous actions: track binned distribution per output dimension
        # Bins: [-inf, -0.5], (-0.5, 0], (0, 0.5], (0.5, inf]
        self.n_action_bins = 4
        self.action_bin_edges = np.array([-np.inf, -0.5, 0.0, 0.5, np.inf])

        # For discrete actions (prob/hard selection): track distribution over action indices
        self.is_discrete_action = game.actionSelect in ["prob", "hard"]

    def getFitness(
        self, wVec, aVec, hyp=None, view=False, nRep=False, seed=-1, track_actions=False
    ):
        """Get fitness of a single individual.

        Args:
          wVec    - (np_array) - weight matrix as a flattened vector
                    [N**2 X 1]
          aVec    - (np_array) - activation function of each node
                    [N X 1]    - stored as ints (see applyAct in ann.py)

        Optional:
          view    - (bool)     - view trial?
          nReps   - (nReps)    - number of trials to get average fitness
          seed    - (int)      - starting random seed for trials
          track_actions - (bool) - track action distribution?

        Returns:
          fitness - (float)    - mean reward over all trials (total: actual + shaped)
          OR (if track_actions=True):
          (fitness, action_dist, raw_fitness) - tuple with:
            fitness     - (float)    - total fitness (actual + shaped reward)
            action_dist - (np_array) - binned action distribution [nOutput x n_action_bins]
            raw_fitness - (float)    - raw fitness (actual game reward only, no shaping)
        """
        if nRep is False:
            nRep = self.nReps
        wVec[np.isnan(wVec)] = 0
        reward = np.empty(nRep)
        raw_reward = np.empty(nRep)  # Track raw (unshaped) reward

        if track_actions:
            # Initialize action distribution
            if self.is_discrete_action:
                # For discrete actions: track counts for each action index
                action_dist = np.zeros(self.nOutput)  # [nOutput] - one count per action
            else:
                # For continuous actions: [nOutput x n_action_bins]
                action_dist = np.zeros((self.nOutput, self.n_action_bins))

        for iRep in range(nRep):
            if track_actions:
                rep_reward, rep_action_dist, rep_raw_reward = self._testInd(
                    wVec, aVec, view=view, seed=seed + iRep, track_actions=True
                )
                reward[iRep] = rep_reward
                raw_reward[iRep] = rep_raw_reward
                action_dist += rep_action_dist
            else:
                result = self._testInd(wVec, aVec, view=view, seed=seed + iRep)
                if isinstance(result, tuple):
                    reward[iRep], raw_reward[iRep] = result
                else:
                    reward[iRep] = result
                    raw_reward[iRep] = result  # No shaping, raw = total

        fitness = np.mean(reward)
        raw_fitness = np.mean(raw_reward)

        if track_actions:
            # Normalize action distribution to get proportions
            if self.is_discrete_action:
                # For discrete: simple normalization over action counts
                total_actions = np.sum(action_dist)
                if total_actions == 0:
                    total_actions = 1
                action_dist = action_dist / total_actions
            else:
                # For continuous: normalize each output dimension
                total_actions = np.sum(action_dist, axis=1, keepdims=True)
                total_actions[total_actions == 0] = 1  # Avoid division by zero
                action_dist = action_dist / total_actions
            return fitness, action_dist, raw_fitness

        return fitness

    def _testInd(self, wVec, aVec, view=False, seed=-1, track_actions=False):
        """Evaluate individual on task
        Args:
          wVec    - (np_array) - weight matrix as a flattened vector
                    [N**2 X 1]
          aVec    - (np_array) - activation function of each node
                    [N X 1]    - stored as ints (see applyAct in ann.py)

        Optional:
          view    - (bool)     - view trial?
          seed    - (int)      - starting random seed for trials
          track_actions - (bool) - track action distribution?

        Returns:
          fitness - (float)    - total reward earned in trial
          OR (if track_actions=True):
          (fitness, action_dist, raw_fitness) - tuple with:
            fitness     - (float)    - total reward (actual + shaped)
            action_dist - (np_array) - action distribution
            raw_fitness - (float)    - raw reward (actual game reward only)
        """
        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)
            self.env.seed(seed)

        # Initialize action distribution tracking
        if track_actions:
            if self.is_discrete_action:
                action_dist = np.zeros(self.nOutput)  # [nOutput] - one count per action
            else:
                action_dist = np.zeros((self.nOutput, self.n_action_bins))

        # Track raw (unshaped) reward separately
        totalRawReward = 0.0

        state = self.env.reset()
        self.env.t = 0
        annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
        action = selectAct(annOut, self.actSelect)

        # Track action distribution
        if track_actions:
            action_dist = self._update_action_dist(action_dist, action)

        wVec[wVec != 0]
        predName = str(np.mean(wVec[wVec != 0]))
        state, reward, done, info = self.env.step(action)

        # Extract raw game reward if available (for shaped environments)
        raw_reward = info.get("game_reward", reward)
        totalRawReward += raw_reward

        if self.maxEpisodeLength == 0:
            if view:
                if self.needsClosed:
                    # For CartPoleSwingUp, close parameter is still used
                    self.env.render(close=done)
                else:
                    # Call render() without mode parameter (render_mode set at init)
                    self.env.render()
                time.sleep(0.02)  # ~50 FPS for smooth visualization
            if track_actions:
                return reward, action_dist, raw_reward
            return reward, raw_reward
        else:
            totalReward = reward

        for tStep in range(self.maxEpisodeLength):
            annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
            action = selectAct(annOut, self.actSelect)

            # Track action distribution
            if track_actions:
                action_dist = self._update_action_dist(action_dist, action)

            state, reward, done, info = self.env.step(action)
            totalReward += reward

            # Extract raw game reward if available (for shaped environments)
            raw_reward = info.get("game_reward", reward)
            totalRawReward += raw_reward

            if view:
                if self.needsClosed:
                    # For CartPoleSwingUp, close parameter is still used
                    self.env.render(close=done)
                else:
                    # Call render() without mode parameter (render_mode set at init)
                    self.env.render()
                time.sleep(0.02)  # ~50 FPS for smooth visualization
            if done:
                break

        if track_actions:
            return totalReward, action_dist, totalRawReward
        return totalReward, totalRawReward

    def _update_action_dist(self, action_dist, action):
        """Update action distribution with new action.

        Args:
          action_dist - (np_array) - current action distribution
                        For discrete: [nOutput] - count per action index
                        For continuous: [nOutput x n_action_bins]
          action      - (np_array or int) - action taken

        Returns:
          action_dist - (np_array) - updated action distribution
        """
        # Handle discrete action (integer index from 'prob' or 'hard' selection)
        if self.is_discrete_action:
            if isinstance(action, (int, np.integer)):
                action_idx = int(action)
            else:
                # If array passed, treat as single value
                action_idx = int(np.atleast_1d(action).flatten()[0])
            if 0 <= action_idx < len(action_dist):
                action_dist[action_idx] += 1
            return action_dist

        # Handle continuous actions - bin each output dimension
        action_arr = np.atleast_1d(action).flatten()
        for i, act_val in enumerate(action_arr[: self.nOutput]):
            bin_idx = np.digitize(act_val, self.action_bin_edges[1:-1])
            action_dist[i, bin_idx] += 1

        return action_dist
