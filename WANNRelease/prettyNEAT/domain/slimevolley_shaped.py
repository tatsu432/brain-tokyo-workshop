"""
SlimeVolley with Dense Reward Shaping for NEAT Training

The problem: SlimeVolley's sparse reward (+1/-1 only when points are scored)
makes it very hard for NEAT to learn because:
1. Random policies always lose 5-0 or 5-1
2. There's no gradient between "random" and "slightly better than random"
3. Selection can't distinguish networks that all get -4 or -5

Solution: Add dense reward shaping to give learning signal for:
- Moving toward the ball (engagement)
- Being in good position when ball is on your side
- Actually hitting the ball
- Avoiding edges

The shaped rewards are small enough that the real game reward dominates
once the agent learns to score points.

USAGE:
1. Copy this file to domain/slimevolley_shaped.py
2. Register it in domain/make_env.py
3. Use 'SlimeVolley-Shaped-v0' as env_name in config.py
"""

import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.seeding import np_random

logger = logging.getLogger(__name__)

try:
    import slimevolleygym
    import gym as old_gym
    SLIMEVOLLEY_AVAILABLE = True
except ImportError:
    SLIMEVOLLEY_AVAILABLE = False
    old_gym = None


class SlimeVolleyShapedEnv(gym.Env):
    """
    SlimeVolley with dense reward shaping to help NEAT learn.
    
    Shaped rewards (all small, so real rewards dominate once learned):
    - Ball proximity: +0.01 for being close to ball horizontally
    - Hit bonus: +0.05 when ball velocity changes (indicates hit)
    - Position penalty: -0.01 for being at edges
    - Progress reward: +0.02 for moving toward ball
    
    These small shaping rewards help NEAT find the gradient from
    "completely random" to "tracks the ball" to "hits the ball"
    to "scores points".
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, shaping_scale=0.1):
        """
        Args:
            shaping_scale: Multiplier for shaped rewards (default 0.1)
                          Set to 0.0 to disable shaping entirely
        """
        if not SLIMEVOLLEY_AVAILABLE:
            raise ImportError("slimevolleygym not installed")
        
        self.env = old_gym.make('SlimeVolley-v0')
        self.shaping_scale = shaping_scale
        
        self.max_steps = 3000
        self.t = 0
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.MultiBinary(3)
        
        # State for reward shaping
        self.prev_ball_x = None
        self.prev_ball_vx = None
        self.prev_ball_vy = None
        self.prev_agent_x = None
        
        self.np_random = None
        self.seed()
    
    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        return [seed]
    
    def _process_action(self, action):
        """Convert continuous NEAT output to binary actions"""
        action = np.array(action).flatten()
        action = np.tanh(action)  # Ensure bounded even if linear output
        
        if len(action) >= 2:
            if action[0] > 0.3:
                forward, backward = 1, 0
            elif action[0] < -0.3:
                forward, backward = 0, 1
            else:
                forward, backward = 0, 0
            jump = 1 if action[1] > 0.3 else 0
        else:
            val = float(action[0]) if len(action) > 0 else 0
            if val > 0.5:
                forward, backward, jump = 1, 0, 1
            elif val > 0.1:
                forward, backward, jump = 1, 0, 0
            elif val > -0.1:
                forward, backward, jump = 0, 0, 1
            elif val > -0.5:
                forward, backward, jump = 0, 1, 0
            else:
                forward, backward, jump = 0, 1, 1
        
        return np.array([forward, backward, jump], dtype=np.int8)
    
    def _compute_shaping_reward(self, obs, prev_obs):
        """
        Compute dense shaping reward to guide learning.
        
        Observation: [agent_x, agent_y, agent_vx, agent_vy,
                      ball_x, ball_y, ball_vx, ball_vy,
                      opp_x, opp_y, opp_vx, opp_vy]
        """
        if prev_obs is None:
            return 0.0
        
        agent_x = obs[0]
        ball_x = obs[4]
        ball_y = obs[5]
        ball_vx = obs[6]
        ball_vy = obs[7]
        
        prev_agent_x = prev_obs[0]
        prev_ball_x = prev_obs[4]
        prev_ball_vx = prev_obs[6]
        prev_ball_vy = prev_obs[7]
        
        shaped_reward = 0.0
        
        # 1. PROXIMITY REWARD: Being close to ball horizontally
        dist_to_ball = abs(agent_x - ball_x)
        if dist_to_ball < 0.2:
            shaped_reward += 0.02  # Very close
        elif dist_to_ball < 0.4:
            shaped_reward += 0.01  # Close
        
        # 2. PROGRESS REWARD: Moving toward ball
        prev_dist = abs(prev_agent_x - ball_x)
        if dist_to_ball < prev_dist:
            shaped_reward += 0.01  # Getting closer
        
        # 3. HIT DETECTION: Ball velocity changed significantly
        # This indicates the agent (or opponent) hit the ball
        vx_change = abs(ball_vx - prev_ball_vx)
        vy_change = abs(ball_vy - prev_ball_vy)
        
        if vx_change > 0.5 or vy_change > 0.5:
            # Ball was hit - reward if agent was close
            if dist_to_ball < 0.3:
                shaped_reward += 0.05  # Likely our hit!
        
        # 4. POSITION PENALTY: Don't hug the edges
        if abs(agent_x) > 0.8:
            shaped_reward -= 0.02
        
        # 5. ENGAGEMENT: Ball is on our side and we're tracking it
        # Agent is on left side (negative x), ball on left = our side
        ball_on_our_side = (agent_x < 0 and ball_x < 0.1) or (agent_x > 0 and ball_x > -0.1)
        if ball_on_our_side and dist_to_ball < 0.5:
            shaped_reward += 0.01
        
        return shaped_reward * self.shaping_scale
    
    def step(self, action):
        binary_action = self._process_action(action)
        
        prev_obs = self._prev_obs if hasattr(self, '_prev_obs') else None
        
        obs, reward, done, info = self.env.step(binary_action)
        
        # Add shaping reward
        if self.shaping_scale > 0:
            shaping = self._compute_shaping_reward(obs, prev_obs)
            reward = reward + shaping
            info['shaping_reward'] = shaping
        
        self._prev_obs = obs.copy()
        self.t += 1
        
        if self.t >= self.max_steps:
            done = True
        
        return obs, reward, done, info
    
    def reset(self):
        self.t = 0
        obs = self.env.reset()
        self._prev_obs = obs.copy()
        return obs
    
    def render(self, mode='human'):
        return self.env.render(mode=mode)
    
    def close(self):
        if self.env:
            self.env.close()


# Also create a version with STRONGER shaping for very early training
class SlimeVolleyStrongShapedEnv(SlimeVolleyShapedEnv):
    """Version with stronger shaping for initial exploration"""
    def __init__(self):
        super().__init__(shaping_scale=0.3)


if __name__ == "__main__":
    # Test the shaped environment
    print("Testing SlimeVolleyShapedEnv...")
    
    env = SlimeVolleyShapedEnv(shaping_scale=0.1)
    
    # Test with random actions
    obs = env.reset()
    total_reward = 0
    total_shaping = 0
    
    for _ in range(1000):
        action = np.random.randn(2)  # Random continuous action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if 'shaping_reward' in info:
            total_shaping += info['shaping_reward']
        if done:
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print(f"Shaping component: {total_shaping:.2f}")
    print(f"Game reward: {total_reward - total_shaping:.2f}")
    
    env.close()
    print("Test complete!")