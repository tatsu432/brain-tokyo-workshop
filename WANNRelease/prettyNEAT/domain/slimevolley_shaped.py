"""
SlimeVolley with Dense Reward Shaping for NEAT Training

The problem: SlimeVolley's sparse reward (+1/-1 only when points are scored)
makes it very hard for NEAT to learn because:
1. Random policies always lose 5-0 or 5-1
2. There's no gradient between "random" and "slightly better than random"
3. Selection can't distinguish networks that all get -4 or -5

Solution: Add dense reward shaping with decomposed fitness:
- Ball touches (primary skill signal)
- Rallies won (game progress)
- Ball time on opponent side (offensive pressure)
- Avoid self-goals heavily

The shaped rewards are carefully weighted so that the final game outcome
dominates once the agent starts scoring, avoiding reward hacking.

Key Design Principles for NEAT:
1. Early learning signal: Small rewards for basic behaviors (moving, touching ball)
2. Progressive rewards: Hitting > Touching > Tracking
3. Game outcome dominance: Final score weighted heavily to prevent gaming
4. No gradient reliance: Works with evolutionary selection

Refactored to use clean architecture with shared components.
"""

import numpy as np

from domain.slimevolley_base import BaseSlimeVolleyEnv
from domain.slimevolley_reward_shaping import (
    CURRICULUM_CONFIGS,
    SlimeVolleyRewardShaper,
)


class SlimeVolleyShapedEnv(BaseSlimeVolleyEnv):
    """
    SlimeVolley with dense reward shaping optimized for NEAT/evolutionary methods.

    Decomposed Fitness Components (configurable via constructor):

    1. Ball Touches: +touch_bonus when agent hits the ball
       - Primary skill signal for early learning
       - Detects velocity changes when ball is near agent

    2. Rallies Won: +rally_bonus for each point scored
       - Directly tied to winning
       - Scaled to dominate once agent learns to score

    3. Ball Position: +ball_opponent_side_bonus * time_ratio when ball on opponent's side
       - Encourages offensive play
       - Small continuous reward for maintaining pressure

    4. Self-Goals Penalty: -self_goal_penalty for losing points
       - Discourages risky play that leads to losing
       - Weighted to make net positive play beneficial

    5. Movement/Tracking (small): Tiny rewards for moving toward ball
       - Helps initial random search find "ball-aware" networks
       - Minimal weight to not dominate once hitting is learned

    Final episode fitness combines all components with heavy weight on
    the actual game outcome (rallies_won - self_goals) to ensure
    the agent optimizes for winning, not reward hacking.

    Refactored to use shared reward shaper and base environment.
    """

    def __init__(
        self,
        # Shaping weights - tune these for your training curriculum
        touch_bonus: float = 0.1,  # Reward per ball touch
        rally_bonus: float = 1.0,  # Reward per rally/point won
        self_goal_penalty: float = 0.5,  # Penalty per point lost
        ball_opponent_side_bonus: float = 0.01,  # Per-step bonus when ball on opponent side
        tracking_bonus: float = 0.02,  # Small bonus for moving toward ball
        game_outcome_weight: float = 10.0,  # Final multiplier for net game score
        shaping_scale: float = 1.0,  # Global scale for all shaping rewards
        # Curriculum support
        enable_curriculum: bool = False,  # Enable curriculum learning (allows stage switching)
        initial_curriculum_stage: str = "touch",  # Initial stage if curriculum enabled
    ):
        """
        Args:
            touch_bonus: Reward for each ball touch (detected via velocity change)
            rally_bonus: Reward for winning a rally (scoring a point)
            self_goal_penalty: Penalty for losing a rally
            ball_opponent_side_bonus: Small per-step reward when ball is on opponent's side
            tracking_bonus: Small reward for moving toward the ball
            game_outcome_weight: Weight multiplier for final game outcome (wins - losses)
            shaping_scale: Global multiplier for all shaping rewards (set 0 to disable)
            enable_curriculum: If True, allows switching curriculum stages via set_curriculum_stage()
            initial_curriculum_stage: Initial curriculum stage ('touch', 'rally', or 'win') if curriculum enabled
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
                touch_bonus=config["touch_bonus"],
                rally_bonus=config["rally_bonus"],
                self_goal_penalty=config["self_goal_penalty"],
                ball_opponent_side_bonus=config["ball_opponent_side_bonus"],
                tracking_bonus=config["tracking_bonus"],
                proximity_bonus=config.get("proximity_bonus", 0.0),
                game_outcome_weight=config["game_outcome_weight"],
                shaping_scale=shaping_scale,
            )
        else:
            # Use provided parameters (fixed rewards, no curriculum)
            self.reward_shaper = SlimeVolleyRewardShaper(
                touch_bonus=touch_bonus,
                rally_bonus=rally_bonus,
                self_goal_penalty=self_goal_penalty,
                ball_opponent_side_bonus=ball_opponent_side_bonus,
                tracking_bonus=tracking_bonus,
                proximity_bonus=0.0,  # Default: no proximity reward if not in curriculum
                game_outcome_weight=game_outcome_weight,
                shaping_scale=shaping_scale,
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

        # Track raw game reward
        self.reward_shaper.episode_stats["raw_game_reward"] += game_reward

        # Compute shaped reward using shared shaper
        total_reward = game_reward  # Start with actual game reward
        if self.reward_shaper.shaping_scale > 0:
            shaping = self.reward_shaper.compute_shaping_reward(
                obs, game_reward, prev_obs
            )
            total_reward += shaping
            info["shaping_reward"] = shaping

        # Store episode stats in info for analysis
        info["episode_stats"] = self.reward_shaper.get_episode_stats()
        info["game_reward"] = game_reward
        info["raw_game_reward"] = self.reward_shaper.episode_stats["raw_game_reward"]

        # At episode end, compute final fitness with game outcome weight
        if done:
            final_bonus = self.reward_shaper.compute_final_bonus()
            total_reward += final_bonus
            info["final_game_bonus"] = final_bonus

        return obs, total_reward, done, info

    def set_curriculum_stage(self, stage: str):
        """
        Switch to a different curriculum stage (only works if enable_curriculum=True).

        Args:
            stage: Curriculum stage ('touch', 'rally', or 'win')
        """
        if not self.enable_curriculum:
            raise ValueError(
                "Curriculum is not enabled. Set enable_curriculum=True in __init__()"
            )

        if stage not in self.CURRICULUM_CONFIGS:
            raise ValueError(f"Unknown stage: {stage}. Use 'touch', 'rally', or 'win'")

        config = self.CURRICULUM_CONFIGS[stage]
        self.reward_shaper.update_config(
            touch_bonus=config["touch_bonus"],
            rally_bonus=config["rally_bonus"],
            self_goal_penalty=config["self_goal_penalty"],
            ball_opponent_side_bonus=config["ball_opponent_side_bonus"],
            tracking_bonus=config["tracking_bonus"],
            proximity_bonus=config.get("proximity_bonus", 0.0),
            game_outcome_weight=config["game_outcome_weight"],
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
    1. TOUCH: Focus on hitting the ball (high touch_bonus)
    2. RALLY: Focus on keeping rallies going (moderate all rewards)
    3. WIN: Focus on winning games (high game_outcome_weight)

    Call set_curriculum_stage() to switch between stages.

    This is a convenience class that automatically enables curriculum mode.
    """

    def __init__(self, initial_stage: str = "touch", shaping_scale: float = 1.0):
        """
        Args:
            initial_stage: Starting curriculum stage ('touch', 'rally', or 'win')
            shaping_scale: Global scale for shaping rewards
        """
        super().__init__(
            enable_curriculum=True,
            initial_curriculum_stage=initial_stage,
            shaping_scale=shaping_scale,
        )


# Backward compatibility aliases
SlimeVolleyStrongShapedEnv = SlimeVolleyShapedEnv  # Use default params


if __name__ == "__main__":
    # Test the shaped environment
    print("Testing SlimeVolleyShapedEnv...")

    env = SlimeVolleyShapedEnv()

    # Test with random actions
    obs = env.reset()
    total_reward = 0
    total_shaping = 0

    for _ in range(1000):
        action = np.random.randint(0, 6)  # Random discrete action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if "shaping_reward" in info:
            total_shaping += info["shaping_reward"]
        if done:
            break

    stats = env.get_episode_stats()
    print(f"Total reward: {total_reward:.2f}")
    print(f"Shaping component: {total_shaping:.2f}")
    print(f"Episode stats: {stats}")

    env.close()

    # Test curriculum env
    print("\nTesting SlimeVolleyCurriculumEnv...")
    env = SlimeVolleyCurriculumEnv(initial_stage="touch")
    print(f"Initial stage: {env.current_stage}")

    env.set_curriculum_stage("rally")
    print(f"Changed to: {env.current_stage}")

    env.set_curriculum_stage("win")
    print(f"Changed to: {env.current_stage}")

    env.close()
    print("Test complete!")
