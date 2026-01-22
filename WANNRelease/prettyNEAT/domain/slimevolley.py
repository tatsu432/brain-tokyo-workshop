"""
SlimeVolley Environment Wrapper for NEAT/WANN

Based on SlimeVolleyGym: https://github.com/hardmaru/slimevolleygym
A simple multi-agent volleyball game environment.

Environment specs:
- Observation: 12-dim vector [agent_x, agent_y, agent_vx, agent_vy, 
                              ball_x, ball_y, ball_vx, ball_vy,
                              opponent_x, opponent_y, opponent_vx, opponent_vy]
- Action: MultiBinary(3) - [forward, backward, jump]
- Reward: +1 when opponent loses life, -1 when you lose life
- Episode: Ends when either agent loses all 5 lives or 3000 timesteps

Refactored to use clean architecture with shared components.
"""

from domain.slimevolley_base import BaseSlimeVolleyEnv


class SlimeVolleyEnv(BaseSlimeVolleyEnv):
    """
    Wrapper for SlimeVolley-v0 environment to work with prettyNEAT framework.

    This wrapper handles the conversion between NEAT's continuous action space
    and SlimeVolley's MultiBinary(3) action space.

    Uses shared base class and action processor to eliminate code duplication.
    """


class SlimeVolleySelfPlayEnv(SlimeVolleyEnv):
    """
    Extended SlimeVolley environment that supports self-play training.

    This environment allows you to specify an opponent policy for training
    against previous versions of your agent.
    """

    def __init__(self, opponent_policy=None):
        """
        Initialize self-play environment.

        Args:
            opponent_policy: Optional callable that takes observation and
                           returns action for the opponent. If None, uses
                           the built-in baseline policy.
        """
        super().__init__()
        self.opponent_policy = opponent_policy

    def step(self, action, otherAction=None):
        """
        Execute one timestep with opponent action.

        Args:
            action: Action from your NEAT agent
            otherAction: Optional opponent action (for self-play)

        Returns:
            observation, reward, done, info (same as parent class)
        """
        # Use parent's step which handles otherAction
        return super().step(action, otherAction)


class SlimeVolleyRewardShapingEnv(SlimeVolleyEnv):
    """
    SlimeVolley with dense reward shaping to help learning.

    This wrapper adds intermediate rewards to guide the agent toward
    better behaviors before it learns to win points. Use this if the
    basic sparse reward version is too hard to learn.

    Shaped rewards:
    - Proximity to ball (encourages engagement)
    - Penalty for staying at edges (fixes edge-hugging behavior)
    - Bonus for good positioning when ball is high

    Note: This is a simpler shaping variant. For more advanced shaping
    with touch detection and curriculum learning, use SlimeVolleyShapedEnv
    from slimevolley_shaped.py.
    """

    def __init__(self, shaping_weight=0.01):
        """
        Initialize reward shaping environment.

        Args:
            shaping_weight: Multiplier for shaped rewards (default 0.01).
                          Set to 0.0 to disable shaping.
        """
        super().__init__()
        self.shaping_weight = shaping_weight
        self.prev_ball_dist = None

    def reset(self):
        """Reset environment and shaping state"""
        obs = super().reset()
        self.prev_ball_dist = None
        return obs

    def step(self, action, otherAction=None):
        """Execute step with reward shaping"""
        obs, reward, done, info = super().step(action, otherAction)

        if self.shaping_weight > 0:
            shaped_reward = self._compute_shaped_reward(obs, reward)
            reward = reward + self.shaping_weight * shaped_reward

        return obs, reward, done, info

    def _compute_shaped_reward(self, obs, original_reward):
        """
        Compute shaped reward based on observation.

        Observation format: [agent_x, agent_y, agent_vx, agent_vy,
                            ball_x, ball_y, ball_vx, ball_vy,
                            opponent_x, opponent_y, opponent_vx, opponent_vy]
        """
        import numpy as np

        agent_x, agent_y = obs[0], obs[1]
        ball_x, ball_y = obs[4], obs[5]

        shaped_reward = 0.0

        # 1. Reward for being close to ball horizontally
        horizontal_dist = abs(agent_x - ball_x)
        if horizontal_dist < 0.3:
            shaped_reward += 2.0  # Strong reward for proximity
        elif horizontal_dist < 0.5:
            shaped_reward += 1.0

        # 2. Penalty for being at edges (fixes the "moving around edges" bug!)
        # Assuming coordinate range is approximately [-1, 1]
        edge_threshold = 0.85
        if abs(agent_x) > edge_threshold:
            shaped_reward -= 3.0  # Strong penalty for edge-hugging

        # 3. Reward for getting closer to ball (incremental progress)
        if self.prev_ball_dist is not None:
            dist_change = self.prev_ball_dist - horizontal_dist
            shaped_reward += dist_change * 5.0  # Reward moving toward ball
        self.prev_ball_dist = horizontal_dist

        # 4. Bonus for good positioning when ball is high
        # (encourages jumping to intercept)
        if horizontal_dist < 0.4 and ball_y > agent_y + 0.2:
            shaped_reward += 1.0

        # 5. Penalty for being far when ball is on agent's side
        # (ball_x and agent_x have same sign = same side of net)
        if np.sign(ball_x) == np.sign(agent_x) and horizontal_dist > 0.6:
            shaped_reward -= 1.0

        return shaped_reward
