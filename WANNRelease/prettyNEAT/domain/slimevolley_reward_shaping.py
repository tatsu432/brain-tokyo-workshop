"""
Reward Shaping Module for SlimeVolley Environments

This module provides reward shaping logic separated from environment concerns.
Follows clean architecture by isolating reward computation from game logic.

Key Design Principles:
1. Early learning signal: Small rewards for basic behaviors (moving, touching ball)
2. Progressive rewards: Hitting > Touching > Tracking
3. Game outcome dominance: Final score weighted heavily to prevent gaming
4. No gradient reliance: Works with evolutionary selection
"""

import numpy as np
from typing import Optional, Dict, Any


class SlimeVolleyRewardShaper:
    """
    Computes dense reward shaping for NEAT/evolutionary training.
    
    Decomposed Fitness Components:
    1. Ball Touches: +touch_bonus when agent hits the ball
    2. Rallies Won: +rally_bonus for each point scored
    3. Ball Position: +ball_opponent_side_bonus * time_ratio when ball on opponent's side
    4. Self-Goals Penalty: -self_goal_penalty for losing points
    5. Movement/Tracking: Tiny rewards for moving toward ball
    6. Proximity: Reward for being close to ball (curriculum stages only)
    """

    def __init__(
        self,
        touch_bonus: float = 0.1,
        rally_bonus: float = 1.0,
        self_goal_penalty: float = 0.5,
        ball_opponent_side_bonus: float = 0.01,
        tracking_bonus: float = 0.02,
        proximity_bonus: float = 0.0,
        game_outcome_weight: float = 10.0,
        shaping_scale: float = 1.0,
        touch_cooldown_steps: int = 10,
    ):
        """
        Initialize reward shaper with configuration.
        
        Args:
            touch_bonus: Reward per ball touch
            rally_bonus: Reward per rally/point won
            self_goal_penalty: Penalty per point lost
            ball_opponent_side_bonus: Per-step bonus when ball on opponent side
            tracking_bonus: Small bonus for moving toward ball
            proximity_bonus: Reward for being close to ball (0 = disabled)
            game_outcome_weight: Final multiplier for net game score
            shaping_scale: Global scale for all shaping rewards
            touch_cooldown_steps: Cooldown period after touch detection
        """
        self.touch_bonus = touch_bonus
        self.rally_bonus = rally_bonus
        self.self_goal_penalty = self_goal_penalty
        self.ball_opponent_side_bonus = ball_opponent_side_bonus
        self.tracking_bonus = tracking_bonus
        self.proximity_bonus = proximity_bonus
        self.game_outcome_weight = game_outcome_weight
        self.shaping_scale = shaping_scale
        self.touch_cooldown_steps = touch_cooldown_steps

        # Episode statistics
        self.episode_stats: Dict[str, Any] = {}
        self.touch_cooldown = 0
        self.prev_obs: Optional[np.ndarray] = None

    def reset(self):
        """Reset shaper state for new episode."""
        self.episode_stats = {
            "ball_touches": 0,
            "rallies_won": 0,
            "rallies_lost": 0,
            "ball_time_opponent_side": 0,
            "total_steps": 0,
            "tracking_reward": 0.0,
            "proximity_reward": 0.0,
            "raw_game_reward": 0.0,
        }
        self.touch_cooldown = 0
        self.prev_obs = None

    def _detect_ball_touch(self, obs: np.ndarray, prev_obs: Optional[np.ndarray]) -> bool:
        """
        Detect if agent touched the ball using velocity change detection.
        
        Returns True if:
        1. Ball was on our side (ball_x < 0)
        2. Ball velocity changed significantly (indicating contact)
        3. Agent was close to ball at the time of contact
        4. Cooldown period has passed
        """
        if prev_obs is None:
            return False

        # Check cooldown
        if self.touch_cooldown > 0:
            return False

        # Extract observations (scaled by 10 in slimevolleygym)
        agent_x = obs[0]
        agent_y = obs[1]
        ball_x = obs[4]
        ball_y = obs[5]
        ball_vx = obs[6]
        ball_vy = obs[7]

        prev_ball_x = prev_obs[4]
        prev_ball_y = prev_obs[5]
        prev_ball_vx = prev_obs[6]
        prev_ball_vy = prev_obs[7]

        # Ball must have been on our side
        if prev_ball_x >= 0:
            return False

        # Agent must be close to ball at PREVIOUS position (when contact happened)
        dist_x = abs(agent_x - prev_ball_x)
        dist_y = abs(agent_y - prev_ball_y)
        dist_2d = np.sqrt(dist_x**2 + dist_y**2)

        # Strict hit threshold
        HIT_THRESHOLD = 0.4
        if dist_2d >= HIT_THRESHOLD:
            return False

        # Check for significant velocity change
        vx_change = abs(ball_vx - prev_ball_vx)
        vy_change = abs(ball_vy - prev_ball_vy)

        # Primary signal: Velocity flipped up
        vy_flipped = (prev_ball_vy < -0.1) and (ball_vy > 0.1)

        # Secondary signal: Significant velocity change
        significant_vx_change = vx_change > 0.4
        significant_vy_change = vy_change > 0.4

        if not (vy_flipped or significant_vx_change or significant_vy_change):
            return False

        # Additional check: ball should have moved
        ball_moved = (abs(ball_x - prev_ball_x) > 0.01) or (abs(ball_y - prev_ball_y) > 0.01)
        if not ball_moved:
            return False

        # Valid touch detected
        self.touch_cooldown = self.touch_cooldown_steps
        return True

    def compute_shaping_reward(
        self, obs: np.ndarray, game_reward: float, prev_obs: Optional[np.ndarray]
    ) -> float:
        """
        Compute decomposed shaping reward for NEAT fitness.
        
        Observation format (all values divided by scaleFactor=10.0):
        [agent_x, agent_y, agent_vx, agent_vy,
         ball_x, ball_y, ball_vx, ball_vy,
         opp_x, opp_y, opp_vx, opp_vy]
        
        Key: ball_x < 0 means ball is on OUR side
        
        Args:
            obs: Current observation (12-dim)
            game_reward: Raw game reward from environment
            prev_obs: Previous observation (for touch detection)
            
        Returns:
            shaped_reward: Additional reward from shaping
        """
        shaped_reward = 0.0

        # 1. BALL TOUCH DETECTION
        if self._detect_ball_touch(obs, prev_obs):
            shaped_reward += self.touch_bonus
            self.episode_stats["ball_touches"] += 1

        # 2. RALLY OUTCOME
        if game_reward > 0:  # We scored!
            shaped_reward += self.rally_bonus
            self.episode_stats["rallies_won"] += 1
        elif game_reward < 0:  # We lost a point
            shaped_reward -= self.self_goal_penalty
            self.episode_stats["rallies_lost"] += 1

        # 3. BALL POSITION PRESSURE
        ball_x = obs[4]
        if ball_x > 0:  # Ball on opponent's side
            shaped_reward += self.ball_opponent_side_bonus
            self.episode_stats["ball_time_opponent_side"] += 1

        # 4. TRACKING/ANTICIPATION
        if prev_obs is not None:
            agent_x = obs[0]
            agent_y = obs[1]
            ball_x = obs[4]
            ball_y = obs[5]
            prev_agent_x = prev_obs[0]
            prev_agent_y = prev_obs[1]

            ball_on_our_side = ball_x < 0

            if ball_on_our_side:
                # Distance to ball improved?
                prev_dist_x = abs(prev_agent_x - ball_x)
                prev_dist_y = abs(prev_agent_y - ball_y)
                prev_dist_2d = np.sqrt(prev_dist_x**2 + prev_dist_y**2)

                curr_dist_x = abs(agent_x - ball_x)
                curr_dist_y = abs(agent_y - ball_y)
                curr_dist_2d = np.sqrt(curr_dist_x**2 + curr_dist_y**2)

                improvement = prev_dist_2d - curr_dist_2d

                if improvement > 0:
                    tracking = self.tracking_bonus * min(improvement / 0.1, 1.0)
                    shaped_reward += tracking
                    self.episode_stats["tracking_reward"] += tracking

        # 5. PROXIMITY REWARD (only if enabled and ball in back court)
        if self.proximity_bonus > 0:
            agent_x = obs[0]
            agent_y = obs[1]
            ball_x = obs[4]
            ball_y = obs[5]

            ball_on_our_side = ball_x < 0

            # Anti-reward-hacking: Only reward proximity if ball is in back court
            if ball_on_our_side and ball_x < -0.3:
                dist_x = abs(agent_x - ball_x)
                dist_y = abs(agent_y - ball_y)
                dist_2d = np.sqrt(dist_x**2 + dist_y**2)

                if dist_2d < 1.0:
                    proximity = self.proximity_bonus * max(0, 1.0 - dist_2d / 1.0)
                    shaped_reward += proximity
                    self.episode_stats["proximity_reward"] += proximity

        self.episode_stats["total_steps"] += 1

        # Update state
        self.prev_obs = obs.copy()
        if self.touch_cooldown > 0:
            self.touch_cooldown -= 1

        return shaped_reward * self.shaping_scale

    def compute_final_bonus(self) -> float:
        """
        Compute final episode bonus based on game outcome.
        
        Returns:
            final_bonus: Weighted bonus for net game score
        """
        return self.game_outcome_weight * (
            self.episode_stats["rallies_won"] - self.episode_stats["rallies_lost"]
        )

    def get_episode_stats(self) -> Dict[str, Any]:
        """Return current episode statistics."""
        return self.episode_stats.copy()

    def update_config(
        self,
        touch_bonus: Optional[float] = None,
        rally_bonus: Optional[float] = None,
        self_goal_penalty: Optional[float] = None,
        ball_opponent_side_bonus: Optional[float] = None,
        tracking_bonus: Optional[float] = None,
        proximity_bonus: Optional[float] = None,
        game_outcome_weight: Optional[float] = None,
        shaping_scale: Optional[float] = None,
    ):
        """Update shaper configuration (for curriculum learning)."""
        if touch_bonus is not None:
            self.touch_bonus = touch_bonus
        if rally_bonus is not None:
            self.rally_bonus = rally_bonus
        if self_goal_penalty is not None:
            self.self_goal_penalty = self_goal_penalty
        if ball_opponent_side_bonus is not None:
            self.ball_opponent_side_bonus = ball_opponent_side_bonus
        if tracking_bonus is not None:
            self.tracking_bonus = tracking_bonus
        if proximity_bonus is not None:
            self.proximity_bonus = proximity_bonus
        if game_outcome_weight is not None:
            self.game_outcome_weight = game_outcome_weight
        if shaping_scale is not None:
            self.shaping_scale = shaping_scale


# Curriculum stage configurations
CURRICULUM_CONFIGS = {
    "touch": {
        "touch_bonus": 0.3,
        "rally_bonus": 0.2,
        "self_goal_penalty": 0.1,
        "ball_opponent_side_bonus": 0.02,
        "tracking_bonus": 0.1,
        "proximity_bonus": 0.05,
        "game_outcome_weight": 2.0,
    },
    "rally": {
        "touch_bonus": 0.15,
        "rally_bonus": 0.5,
        "self_goal_penalty": 0.3,
        "ball_opponent_side_bonus": 0.01,
        "tracking_bonus": 0.02,
        "proximity_bonus": 0.01,
        "game_outcome_weight": 5.0,
    },
    "win": {
        "touch_bonus": 0.05,
        "rally_bonus": 1.0,
        "self_goal_penalty": 0.5,
        "ball_opponent_side_bonus": 0.005,
        "tracking_bonus": 0.01,
        "proximity_bonus": 0.0,
        "game_outcome_weight": 10.0,
    },
}
