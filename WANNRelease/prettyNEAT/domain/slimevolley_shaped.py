"""
SlimeVolley with Simplified Reward Shaping for NEAT Training

Simplified reward design to avoid reward hacking:
- Survival reward: Function of number of time steps (scaled)
- Win reward: Number of rallies won (pure reward without shaping)
- Curriculum: Gradually transitions from survival-weighted to win-weighted

Key Design Principles:
1. Simplicity: Avoid complex reward components that can be gamed
2. Curriculum: Start with survival focus, gradually shift to win focus
3. Clean architecture: Reward computation separated from environment logic
"""

import numpy as np

from domain.slimevolley_base import BaseSlimeVolleyEnv
from domain.slimevolley_reward_shaping import (
    CURRICULUM_CONFIGS,
    SlimeVolleyRewardShaper,
)


class SlimeVolleyShapedEnv(BaseSlimeVolleyEnv):
    """
    SlimeVolley with simplified reward shaping optimized for NEAT/evolutionary methods.

    Simplified Reward Components:
    1. Survival reward: total_steps * survival_scale
       - Encourages agent to stay alive longer
       - Simple, interpretable, hard to game

    2. Win reward: rallies_won
       - Pure reward for winning (no shaping)
       - Directly measures performance

    Curriculum Learning:
    - Blends survival and win rewards using curriculum_weight
    - Early stages: High weight on survival (curriculum_weight = 1.0)
    - Late stages: High weight on wins (curriculum_weight = 0.0)
    - Gradual transition helps agent learn basic skills before focusing on winning

    Refactored to use clean architecture with shared reward shaper.
    """

    def __init__(
        self,
        # Simplified shaping parameters
        survival_scale: float = 0.01,  # Scale factor for survival reward (INCREASED from 0.01)
        curriculum_weight: float = 1.0,  # Weight for survival vs wins (1.0 = all survival, 0.0 = all wins)
        # Curriculum support
        enable_curriculum: bool = False,  # Enable curriculum learning (allows stage switching)
        initial_curriculum_stage: str = "survival",  # Initial stage if curriculum enabled
    ):
        """
        Args:
            survival_scale: Scale factor for survival reward (reward = steps * survival_scale)
            curriculum_weight: Weight for survival vs wins (1.0 = all survival, 0.0 = all wins)
            enable_curriculum: If True, allows switching curriculum stages via set_curriculum_stage()
            initial_curriculum_stage: Initial curriculum stage ('survival', 'mixed', or 'wins') if curriculum enabled
        """
        super().__init__()

        # Curriculum support
        self.enable_curriculum = enable_curriculum
        self.current_stage = initial_curriculum_stage if enable_curriculum else None
        self.CURRICULUM_CONFIGS = CURRICULUM_CONFIGS

        # Initialize reward shaper with configuration
        if enable_curriculum and initial_curriculum_stage in CURRICULUM_CONFIGS:
            config = CURRICULUM_CONFIGS[initial_curriculum_stage]
            self.reward_shaper = SlimeVolleyRewardShaper(
                survival_scale=config["survival_scale"],
                curriculum_weight=config["curriculum_weight"],
            )
        else:
            # Use provided parameters (fixed rewards, no curriculum)
            self.reward_shaper = SlimeVolleyRewardShaper(
                survival_scale=survival_scale,
                curriculum_weight=curriculum_weight,
            )

    def step(self, action, otherAction=None):
        """
        Take a step in the environment.

        Args:
            action: Action for our agent (right side)
            otherAction: Optional action for opponent (for self-play)
                        If None, uses built-in baseline policy
        """
        # Get previous observation for reward shaping
        prev_obs = self.reward_shaper.prev_obs

        # Step in base environment (handles action processing)
        obs, game_reward, done, info = super().step(action, otherAction)

        # Track statistics (no per-step reward in simplified version)
        self.reward_shaper.compute_shaping_reward(obs, game_reward, prev_obs)

        # Store episode stats in info for analysis
        info["episode_stats"] = self.reward_shaper.get_episode_stats()
        info["game_reward"] = game_reward

        # At episode end, compute final reward using curriculum blending
        if done:
            final_reward = self.reward_shaper.compute_final_reward()
            info["final_reward"] = final_reward
            info["survival_reward"] = (
                self.reward_shaper.episode_stats["total_steps"]
                * self.reward_shaper.survival_scale
            )
            info["win_reward"] = float(self.reward_shaper.episode_stats["rallies_won"])
            return obs, final_reward, done, info

        # During episode, return 0 reward (all reward computed at end)
        return obs, 0.0, done, info

    def set_curriculum_stage(self, stage: str):
        """
        Switch to a different curriculum stage (only works if enable_curriculum=True).

        Args:
            stage: Curriculum stage ('survival', 'mixed', or 'wins')
        """
        if not self.enable_curriculum:
            raise ValueError(
                "Curriculum is not enabled. Set enable_curriculum=True in __init__()"
            )

        if stage not in self.CURRICULUM_CONFIGS:
            raise ValueError(
                f"Unknown stage: {stage}. Use 'survival', 'mixed', or 'wins'"
            )

        config = self.CURRICULUM_CONFIGS[stage]
        self.reward_shaper.update_config(
            survival_scale=config["survival_scale"],
            curriculum_weight=config["curriculum_weight"],
        )
        self.current_stage = stage

    def reset(self):
        """Reset environment and reward shaper state."""
        obs = super().reset()
        self.reward_shaper.reset()
        return obs

    def get_episode_stats(self):
        """Return current episode statistics."""
        return self.reward_shaper.get_episode_stats()


class SlimeVolleyCurriculumEnv(SlimeVolleyShapedEnv):
    """
    SlimeVolley with curriculum learning support.

    Curriculum stages:
    1. SURVIVAL: Focus on staying alive (curriculum_weight = 1.0, 100% survival reward)
    2. MIXED: Balanced focus (curriculum_weight = 0.5, 50% survival, 50% wins)
    3. WINS: Focus on winning (curriculum_weight = 0.0, 100% win reward, pure reward)

    Call set_curriculum_stage() to switch between stages.

    This is a convenience class that automatically enables curriculum mode.
    """

    def __init__(self, initial_stage: str = "survival"):
        """
        Args:
            initial_stage: Starting curriculum stage ('survival', 'mixed', or 'wins')
        """
        super().__init__(
            enable_curriculum=True,
            initial_curriculum_stage=initial_stage,
        )


# Backward compatibility aliases
SlimeVolleyStrongShapedEnv = SlimeVolleyShapedEnv  # Use default params


if __name__ == "__main__":
    # Test the shaped environment
    print("Testing SlimeVolleyShapedEnv...")

    env = SlimeVolleyShapedEnv(survival_scale=0.01, curriculum_weight=1.0)

    # Test with random actions
    obs = env.reset()
    total_reward = 0

    for _ in range(1000):
        # Random action: 3 outputs [forward, jump, back] with random values
        action = np.random.randn(3)  # Random continuous action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print(f"Episode ended. Final reward: {reward:.2f}")
            if "final_reward" in info:
                print("Final reward breakdown:")
                print(f"  Survival reward: {info.get('survival_reward', 0):.2f}")
                print(f"  Win reward: {info.get('win_reward', 0):.2f}")
            break

    stats = env.get_episode_stats()
    print(f"Total reward: {total_reward:.2f}")
    print(f"Episode stats: {stats}")

    env.close()

    # Test curriculum env
    print("\nTesting SlimeVolleyCurriculumEnv...")
    env = SlimeVolleyCurriculumEnv(initial_stage="survival")
    print(f"Initial stage: {env.current_stage}")

    env.set_curriculum_stage("mixed")
    print(f"Changed to: {env.current_stage}")

    env.set_curriculum_stage("wins")
    print(f"Changed to: {env.current_stage}")

    env.close()
    print("Test complete!")
