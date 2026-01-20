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
        
        Args:
            action: Can be:
                - numpy array of shape (3,) with continuous values
                - numpy array of shape (1,) or scalar (single continuous value)
                
        Returns:
            binary_action: numpy array of shape (3,) with binary values [0 or 1]
        """
        if isinstance(action, (int, float, np.number)):
            # Single scalar action - map to 3 binary actions
            # Use different thresholds for each action
            val = float(action)
            binary_action = np.array([
                1 if val > 0.33 else 0,   # forward
                1 if val < -0.33 else 0,   # backward  
                1 if abs(val) > 0.5 else 0 # jump
            ], dtype=np.int8)
        elif len(action) == 1:
            # Single value in array form
            val = float(action[0])
            binary_action = np.array([
                1 if val > 0.33 else 0,
                1 if val < -0.33 else 0,
                1 if abs(val) > 0.5 else 0
            ], dtype=np.int8)
        else:
            # Multiple continuous values - threshold each independently
            # Values > 0 activate the corresponding action
            binary_action = np.array([
                1 if action[0] > 0 else 0,  # forward
                1 if action[1] > 0 else 0,  # backward
                1 if action[2] > 0 else 0   # jump
            ], dtype=np.int8)
        
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
        Render the environment.
        
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
        
        try:
            return self.env.render(mode=mode)
        except Exception as e:
            if not self._render_warning_shown:
                print(f"Warning: Could not render environment: {e}")
                print("Rendering may not be available in headless mode.")
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
