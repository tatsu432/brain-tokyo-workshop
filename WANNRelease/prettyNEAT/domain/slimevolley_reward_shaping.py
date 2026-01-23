"""
Reward Shaping Module for SlimeVolley Environments

This module provides simplified reward shaping logic separated from environment concerns.
Follows clean architecture by isolating reward computation from game logic.

Simplified Reward Design:
- Survival reward: Function of number of time steps (scaled)
- Win reward: Number of rallies won (pure reward without shaping)
- Curriculum: Gradually transitions from survival-weighted to win-weighted

This design avoids reward hacking by using simple, interpretable rewards.
"""

from typing import Any, Dict, Optional

import numpy as np


class SlimeVolleyRewardShaper:
    """
    Computes simplified reward shaping for NEAT/evolutionary training.

    Reward Components:
    1. Survival reward: total_steps * survival_scale (encourages staying alive)
    2. Win reward: rallies_won (pure reward for winning)

    Curriculum blends these components:
    - Early: High weight on survival (curriculum_weight near 1.0)
    - Late: High weight on wins (curriculum_weight near 0.0)
    """

    def __init__(
        self,
        survival_scale: float = 0.05,  # INCREASED from 0.01 for better signal-to-noise ratio
        curriculum_weight: float = 1.0,
    ):
        """
        Initialize reward shaper with simplified configuration.

        Args:
            survival_scale: Scale factor for survival reward (reward = steps * survival_scale)
            curriculum_weight: Weight for survival vs wins (1.0 = all survival, 0.0 = all wins)
        """
        self.survival_scale = survival_scale
        self.curriculum_weight = curriculum_weight

        # Episode statistics
        self.episode_stats: Dict[str, Any] = {}
        self.prev_obs: Optional[np.ndarray] = None

    def reset(self):
        """Reset shaper state for new episode."""
        self.episode_stats = {
            "rallies_won": 0,
            "total_steps": 0,
            "raw_game_reward": 0.0,
        }
        self.prev_obs = None

    def compute_shaping_reward(
        self, obs: np.ndarray, game_reward: float, prev_obs: Optional[np.ndarray]
    ) -> float:
        """
        Compute simplified shaping reward per step.

        This method tracks statistics but returns 0 for per-step rewards.
        The actual reward computation happens at episode end via compute_final_reward().

        Args:
            obs: Current observation (12-dim)
            game_reward: Raw game reward from environment (+1 for win, -1 for loss, 0 otherwise)
            prev_obs: Previous observation (not used in simplified version)

        Returns:
            shaped_reward: 0.0 (all rewards computed at episode end)
        """
        # Track wins (game_reward > 0 means we scored a point)
        if game_reward > 0:
            self.episode_stats["rallies_won"] += 1

        # Track steps
        self.episode_stats["total_steps"] += 1
        self.episode_stats["raw_game_reward"] += game_reward

        # Update state
        self.prev_obs = obs.copy()

        # No per-step reward in simplified version
        return 0.0

    def compute_final_reward(self) -> float:
        """
        Compute final episode reward using curriculum blending.

        Formula:
        total_reward = curriculum_weight * survival_reward + (1 - curriculum_weight) * win_reward
        where:
        - survival_reward = total_steps * survival_scale
        - win_reward = rallies_won

        Returns:
            final_reward: Blended reward based on curriculum weight
        """
        survival_reward = self.episode_stats["total_steps"] * self.survival_scale
        win_reward = float(self.episode_stats["rallies_won"])

        # Blend based on curriculum weight
        total_reward = (
            self.curriculum_weight * survival_reward
            + (1.0 - self.curriculum_weight) * win_reward
        )

        return total_reward

    def get_episode_stats(self) -> Dict[str, Any]:
        """Return current episode statistics."""
        stats = self.episode_stats.copy()
        # Add computed rewards for analysis
        survival_reward = stats["total_steps"] * self.survival_scale
        win_reward = float(stats["rallies_won"])
        stats["survival_reward"] = survival_reward
        stats["win_reward"] = win_reward
        stats["curriculum_weight"] = self.curriculum_weight
        return stats

    def update_config(
        self,
        survival_scale: Optional[float] = None,
        curriculum_weight: Optional[float] = None,
    ):
        """
        Update shaper configuration (for curriculum learning).

        Args:
            survival_scale: New scale factor for survival reward
            curriculum_weight: New curriculum weight (1.0 = all survival, 0.0 = all wins)
        """
        if survival_scale is not None:
            self.survival_scale = survival_scale
        if curriculum_weight is not None:
            self.curriculum_weight = curriculum_weight


# Curriculum stage configurations
# Stages transition from survival-focused (early) to win-focused (late)
CURRICULUM_CONFIGS = {
    "survival": {
        "survival_scale": 0.01,  # INCREASED from 0.01 for better signal-to-noise ratio
        "curriculum_weight": 1.0,  # 100% survival, 0% wins
    },
    "mixed": {
        "survival_scale": 0.01,  # INCREASED from 0.01 for better signal-to-noise ratio
        "curriculum_weight": 0.5,  # 50% survival, 50% wins
    },
    "wins": {
        "survival_scale": 0.01,  # INCREASED from 0.01 for better signal-to-noise ratio
        "curriculum_weight": 0.0,  # 0% survival, 100% wins (pure reward)
    },
}
