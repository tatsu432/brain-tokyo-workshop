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
    
    # Action mapping for 6 discrete actions:
    # Index 0: left + no jump  → [forward=0, backward=1, jump=0]
    # Index 1: stay + no jump  → [forward=0, backward=0, jump=0]
    # Index 2: right + no jump → [forward=1, backward=0, jump=0]
    # Index 3: left + jump     → [forward=0, backward=1, jump=1]
    # Index 4: stay + jump     → [forward=0, backward=0, jump=1]
    # Index 5: right + jump    → [forward=1, backward=0, jump=1]
    DISCRETE_ACTION_MAP = [
        [0, 1, 0],  # 0: left + no jump
        [0, 0, 0],  # 1: stay + no jump
        [1, 0, 0],  # 2: right + no jump
        [0, 1, 1],  # 3: left + jump
        [0, 0, 1],  # 4: stay + jump
        [1, 0, 1],  # 5: right + jump
    ]
    
    def _process_action(self, action):
        """Convert NEAT output to binary actions.
        
        Supports two modes:
        1. Discrete mode (int): action is an index 0-5 from probabilistic selection
        2. Continuous mode (array): action is continuous values with thresholds
        """
        # Handle discrete action index (from 'prob' actionSelect)
        if isinstance(action, (int, np.integer)):
            action_idx = int(action) % 6  # Ensure valid index
            return np.array(self.DISCRETE_ACTION_MAP[action_idx], dtype=np.int8)
        
        # Handle continuous action (legacy mode or different actionSelect)
        action = np.array(action).flatten()
        
        # If 6 outputs (discrete mode but passed as array), take argmax
        if len(action) == 6:
            action_idx = np.argmax(action)
            return np.array(self.DISCRETE_ACTION_MAP[action_idx], dtype=np.int8)
        
        # Legacy 2-output continuous mode with thresholds
        action = np.tanh(action)  # Ensure bounded
        
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
        
        Observation format (all values divided by scaleFactor=10.0 in slimevolleygym):
        [agent_x, agent_y, agent_vx, agent_vy,
         ball_x, ball_y, ball_vx, ball_vy,
         opp_x, opp_y, opp_vx, opp_vy]
        
        CORRECT Scaled Ranges (after ÷10):
        Index  Variable   Range           Notes
        0      agent_x    [0.2, 2.25]     Always positive (agent sees self on right)
        1      agent_y    [0.15, ~1.5]    Ground=0.15, max jump ~1.5
        2      agent_vx   [-1.75, 1.75]   PLAYER_SPEED_X / 10
        3      agent_vy   [-3, 1.35]      Gravity pulls down, jump up
        4      ball_x     [-2.35, 2.35]   Negative = agent's side, Positive = opponent's side
        5      ball_y     [0.2, 4.75]     Ground to ceiling
        6      ball_vx    [-2.25, 2.25]   MAX_BALL_SPEED / 10
        7      ball_vy    [-2.25, 2.25]   MAX_BALL_SPEED / 10
        8      opp_x      [0.2, 2.25]     Always positive (mirrored)
        9      opp_y      [0.15, ~1.5]    Same as agent
        10     opp_vx     [-1.75, 1.75]   Mirrored velocity
        11     opp_vy     [-3, 1.35]      Same as agent
        
        Key insight: ball_x < 0 means ball is on OUR side (we need to defend/hit it)
        """
        if prev_obs is None:
            return 0.0
        
        # Extract current observations
        agent_x = obs[0]
        agent_y = obs[1]
        ball_x = obs[4]
        ball_y = obs[5]
        ball_vx = obs[6]
        ball_vy = obs[7]
        
        # Extract previous observations
        prev_agent_x = prev_obs[0]
        prev_ball_x = prev_obs[4]
        prev_ball_vx = prev_obs[6]
        prev_ball_vy = prev_obs[7]
        
        shaped_reward = 0.0
        
        # Determine if ball is on our side (negative x in transformed coordinates)
        ball_on_our_side = ball_x < 0.0
        
        # Horizontal distance to ball (agent_x is always positive, ball_x can be negative)
        dist_to_ball_x = abs(agent_x - ball_x)
        
        # Max possible horizontal distance is ~4.6 (agent at 2.25, ball at -2.35)
        MAX_DIST_X = 4.6
        
        # =================================================================
        # 1. HIT DETECTION (Primary reward - most important skill)
        # =================================================================
        # A successful hit is the core skill. Give strong reward.
        # Detection: ball was on our side + velocity changed + agent was close
        
        ball_was_on_our_side = prev_ball_x < 0.0
        
        if ball_was_on_our_side:
            # Detect hit via velocity change
            vx_change = abs(ball_vx - prev_ball_vx)
            vy_change = abs(ball_vy - prev_ball_vy)
            
            # Hit indicators:
            # 1. vy flips from negative to positive (upward hit)
            # 2. Significant velocity magnitude change
            vy_flipped_up = (prev_ball_vy < -0.1) and (ball_vy > 0.1)
            significant_change = (vx_change > 0.3) or (vy_change > 0.3)
            
            if vy_flipped_up or significant_change:
                # Agent must have been close to cause the hit
                # Use 2D distance for better accuracy
                dist_2d = np.sqrt(dist_to_ball_x**2 + (agent_y - ball_y)**2)
                HIT_THRESHOLD = 0.5  # (agent_r + ball_r) / 10 + margin
                
                if dist_2d < HIT_THRESHOLD:
                    shaped_reward += 0.15  # Strong reward for hitting
                    
                    # Bonus for hitting toward opponent (offensive play)
                    if ball_vx > 0.5:
                        shaped_reward += 0.05
        
        # =================================================================
        # 2. ANTICIPATION REWARD (move toward predicted ball position)
        # =================================================================
        # Instead of chasing current ball position, reward moving toward
        # where the ball WILL BE. This is more strategic than ball chasing.
        
        if ball_on_our_side:
            # Simple prediction: where will ball be in ~10 frames?
            # ball_x + ball_vx * dt, but ball_vx is already velocity
            LOOKAHEAD = 0.3  # seconds worth of prediction (scaled)
            predicted_ball_x = ball_x + ball_vx * LOOKAHEAD
            
            # Clamp to valid range (ball can't go past walls)
            predicted_ball_x = np.clip(predicted_ball_x, -2.35, 2.35)
            
            # Only reward if predicted position is still on our side
            if predicted_ball_x < 0:
                # Distance to predicted position
                dist_to_predicted = abs(agent_x - predicted_ball_x)
                prev_dist_to_predicted = abs(prev_agent_x - predicted_ball_x)
                
                # Reward for moving toward predicted position
                improvement = prev_dist_to_predicted - dist_to_predicted
                if improvement > 0:
                    shaped_reward += 0.03 * min(improvement / 0.06, 1.0)
        
        # =================================================================
        # 3. BALL APPROACHING AWARENESS (prepare when ball comes to us)
        # =================================================================
        # When ball is on opponent's side but coming toward us, start moving
        # This replaces the static "defensive position" reward
        
        ball_coming_to_us = (ball_vx < -0.3)  # Ball moving toward our side
        
        if not ball_on_our_side and ball_coming_to_us:
            # Predict where ball will cross into our side
            # Rough estimate: ball needs to travel ball_x distance at ball_vx speed
            if ball_vx < -0.1:  # Avoid division by near-zero
                time_to_cross = -ball_x / (-ball_vx)  # Time until ball_x = 0
                # Predict x position when it's at our hitting range
                predicted_x_at_hit = ball_x + ball_vx * (time_to_cross + 0.2)
                predicted_x_at_hit = np.clip(predicted_x_at_hit, -2.35, 0)
                
                # Reward being closer to predicted interception point
                dist_to_intercept = abs(agent_x - abs(predicted_x_at_hit))
                intercept_reward = 0.01 * np.exp(-2.0 * dist_to_intercept)
                shaped_reward += intercept_reward
        
        # =================================================================
        # 4. GROUNDED PENALTY (encourage jumping for high balls)
        # =================================================================
        # If ball is high and on our side, penalize staying on ground
        # This encourages learning to jump
        
        if ball_on_our_side and ball_y > 0.8 and dist_to_ball_x < 1.5:
            # Ball is high and we should be jumping
            GROUND_LEVEL = 0.2
            if agent_y < GROUND_LEVEL + 0.1:  # Agent is on/near ground
                # Small penalty for not jumping when ball is high
                shaped_reward -= 0.01
        
        # =================================================================
        # 5. EXTREME EDGE PENALTY (only penalize being stuck at far wall)
        # =================================================================
        # Only penalize being at the far edge where agent can't return
        # Don't penalize being near net - sometimes necessary for hits
        
        if agent_x > 2.1:  # Very close to back wall
            shaped_reward -= 0.02
        
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