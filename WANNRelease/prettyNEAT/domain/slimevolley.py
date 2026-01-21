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
"""

import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.seeding import np_random

logger = logging.getLogger(__name__)

# Try to import slimevolleygym
try:
    import slimevolleygym
    import gym as old_gym  # slimevolleygym uses old gym API
    
    # Apply rendering patch to fix gym rendering compatibility issues
    try:
        from domain.slimevolley_rendering_patch import apply_rendering_patch
        apply_rendering_patch()
        logger.info("Applied slimevolleygym rendering patch")
    except Exception as e:
        logger.warning(f"Could not apply rendering patch: {e}")
    
    SLIMEVOLLEY_AVAILABLE = True
except ImportError:
    SLIMEVOLLEY_AVAILABLE = False
    old_gym = None
    print("Warning: slimevolleygym not installed. Install with: pip install slimevolleygym")


class SlimeVolleyEnv(gym.Env):
    """
    Wrapper for SlimeVolley-v0 environment to work with prettyNEAT framework.
    
    This wrapper handles the conversion between NEAT's continuous action space
    and SlimeVolley's MultiBinary(3) action space.
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    
    def __init__(self):
        if not SLIMEVOLLEY_AVAILABLE:
            raise ImportError(
                "slimevolleygym is not installed. "
                "Install it with: pip install slimevolleygym"
            )
        
        # Create the base SlimeVolley environment
        # slimevolleygym uses old gym API, not gymnasium
        # Disable env checker to avoid numpy 2.x compatibility issues
        self.env = old_gym.make('SlimeVolley-v0', disable_env_checker=True)
        
        # Environment properties
        self.max_steps = 3000  # Maximum steps per episode
        self.t = 0  # Current timestep
        
        # Observation space: 12-dim continuous state
        # [agent_x, agent_y, agent_vx, agent_vy, 
        #  ball_x, ball_y, ball_vx, ball_vy,
        #  opponent_x, opponent_y, opponent_vx, opponent_vy]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(12,), 
            dtype=np.float32
        )
        
        # Action space: 3 binary actions [forward, backward, jump]
        # NEAT will output continuous values, which we'll threshold to binary
        self.action_space = spaces.MultiBinary(3)
        
        # Rendering
        self.viewer = None
        self._render_warning_shown = False
        
        # Random seed
        self.np_random = None
        self.seed()
    
    def seed(self, seed=None):
        """Set random seed for reproducibility"""
        self.np_random, seed = np_random(seed)
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        return [seed]
    
    def _process_action(self, action):
        """
        Convert NEAT's continuous action output to SlimeVolley's binary actions.
        
        CRITICAL FIX: Clip linear outputs to [-1, 1] like CartPole SwingUp does!
        This prevents unbounded linear outputs from saturating thresholds while
        maintaining good gradient flow for learning.
        
        Args:
            action: Can be:
                - numpy array of shape (2,) with continuous values [horizontal, jump]
                - numpy array of shape (1,) or scalar (single continuous value)
                
        Returns:
            binary_action: numpy array of shape (3,) with binary values [0 or 1]
                          [forward, backward, jump]
        """
        action = np.array(action).flatten()
        
        # CRITICAL: Clip actions to [-1, 1] like SwingUp does!
        # This handles unbounded linear outputs without using tanh
        action = np.clip(action, -1.0, 1.0)
        
        if len(action) == 1:
            # Single output: map to discrete action combinations
            # This prevents conflicting actions and creates clear action zones
            val = float(action[0])
            if val > 0.5:
                binary_action = np.array([1, 0, 1], dtype=np.int8)  # forward + jump
            elif val > 0.1:
                binary_action = np.array([1, 0, 0], dtype=np.int8)  # forward only
            elif val > -0.1:
                binary_action = np.array([0, 0, 1], dtype=np.int8)  # jump only
            elif val > -0.5:
                binary_action = np.array([0, 1, 0], dtype=np.int8)  # backward only
            else:
                binary_action = np.array([0, 1, 1], dtype=np.int8)  # backward + jump
        else:
            # Multiple outputs: [horizontal_movement, jump]
            # V4 FIX: Linear outputs are now clipped to [-1, 1]
            # Use moderate thresholds that work well with full clipped range
            # This matches the SwingUp pattern which works!
            if action[0] > 0.3:
                forward = 1
                backward = 0
            elif action[0] < -0.3:
                forward = 0
                backward = 1
            else:
                # Deadband: no horizontal movement if |action[0]| < 0.3
                forward = 0
                backward = 0
            
            # Jump threshold: higher since jumping is more deliberate
            jump = 1 if action[1] > 0.4 else 0
            
            binary_action = np.array([forward, backward, jump], dtype=np.int8)
        
        return binary_action
    
    def step(self, action):
        """
        Execute one timestep in the environment.
        
        Args:
            action: Action from NEAT (continuous values)
            
        Returns:
            observation: 12-dim state vector
            reward: Reward from environment (+1, -1, or 0)
            done: Whether episode is finished
            info: Additional information dict
        """
        # Convert continuous action to binary
        binary_action = self._process_action(action)
        
        # Execute action in environment
        obs, reward, done, info = self.env.step(binary_action)
        
        # Increment timestep counter
        self.t += 1
        
        # Check if max steps reached
        if self.t >= self.max_steps:
            done = True
        
        return obs, reward, done, info
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial 12-dim state vector
        """
        self.t = 0
        obs = self.env.reset()
        return obs
    
    def render(self, mode='human', close=False):
        """
        Render the environment using slimevolleygym's native rendering.
        
        The rendering patch enables the actual game rendering which looks
        much better than a custom implementation.
        
        Args:
            mode: Render mode ('human' or 'rgb_array')
            close: Whether to close the rendering
            
        Returns:
            None or rgb_array depending on mode
        """
        if close:
            if self.viewer is not None:
                self.env.close()
                self.viewer = None
            return
        
        # Use the actual slimevolleygym rendering
        # With our rendering patch, this now works properly
        try:
            return self.env.render(mode=mode)
        except Exception as e:
            if not self._render_warning_shown:
                logger.warning(f"Rendering error: {e}")
                print(f"Warning: Rendering error: {e}")
                self._render_warning_shown = True
            return None
    
    def close(self):
        """Clean up environment resources"""
        if self.env is not None:
            self.env.close()


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
    
    def step(self, action):
        """
        Execute one timestep with opponent action.
        
        Args:
            action: Action from your NEAT agent
            
        Returns:
            observation, reward, done, info (same as parent class)
        """
        # Convert agent action to binary
        agent_action = self._process_action(action)
        
        # Get opponent action
        if self.opponent_policy is not None:
            # Get opponent's observation from info
            # Note: We'd need to track this from previous step
            # For now, use default opponent
            obs, reward, done, info = self.env.step(agent_action)
        else:
            # Use default opponent policy
            obs, reward, done, info = self.env.step(agent_action)
        
        self.t += 1
        if self.t >= self.max_steps:
            done = True
        
        return obs, reward, done, info


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
    
    def step(self, action):
        """Execute step with reward shaping"""
        obs, reward, done, info = super().step(action)
        
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
