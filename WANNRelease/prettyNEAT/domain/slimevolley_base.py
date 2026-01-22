"""
Base SlimeVolley Environment Wrapper

This module provides a common base class for all SlimeVolley environment variants.
Follows clean architecture by separating environment wrapping from domain-specific logic.

All SlimeVolley wrappers should inherit from this base class to ensure consistency.
"""

import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.seeding import np_random
from typing import Optional

from domain.slimevolley_actions import SlimeVolleyActionProcessor

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
    print(
        "Warning: slimevolleygym not installed. Install with: pip install slimevolleygym"
    )


class BaseSlimeVolleyEnv(gym.Env):
    """
    Base wrapper for SlimeVolley-v0 environment.
    
    This class handles common functionality:
    - Environment initialization and wrapping
    - Action processing via shared action processor
    - Observation space definition
    - Basic step/reset/render/close methods
    - Timestep tracking
    
    Subclasses should override step() and reset() to add domain-specific logic.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(
        self,
        env_id: str = "SlimeVolley-v0",
        max_steps: int = 3000,
        clip_actions: bool = True,
    ):
        """
        Initialize base SlimeVolley environment.
        
        Args:
            env_id: Gym environment ID (default: "SlimeVolley-v0")
            max_steps: Maximum steps per episode
            clip_actions: Whether to clip continuous actions to [-1, 1]
        """
        if not SLIMEVOLLEY_AVAILABLE:
            raise ImportError(
                "slimevolleygym is not installed. "
                "Install it with: pip install slimevolleygym"
            )

        # Create the base SlimeVolley environment
        # slimevolleygym uses old gym API, not gymnasium
        # Disable env checker to avoid numpy 2.x compatibility issues
        self.env = old_gym.make(env_id, disable_env_checker=True)

        # Environment properties
        self.max_steps = max_steps
        self.t = 0  # Current timestep

        # Observation space: 12-dim continuous state
        # [agent_x, agent_y, agent_vx, agent_vy,
        #  ball_x, ball_y, ball_vx, ball_vy,
        #  opponent_x, opponent_y, opponent_vx, opponent_vy]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        # Action space: 3 binary actions [forward, backward, jump]
        self.action_space = spaces.MultiBinary(3)

        # Action processor (shared across all variants)
        self.action_processor = SlimeVolleyActionProcessor(clip_actions=clip_actions)

        # Rendering
        self.viewer = None
        self._render_warning_shown = False

        # Random seed
        self.np_random = None
        self.seed()

    def seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility"""
        self.np_random, seed = np_random(seed)
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
        return [seed]

    def _process_action(self, action):
        """
        Convert NEAT's continuous action output to SlimeVolley's binary actions.
        
        Delegates to shared action processor to eliminate duplication.
        
        Args:
            action: Action from NEAT network (various formats supported)
            
        Returns:
            binary_action: numpy array of shape (3,) with binary values
                          [forward, backward, jump]
        """
        return self.action_processor.process(action)

    def step(self, action, otherAction=None):
        """
        Execute one timestep in the environment.
        
        Base implementation handles action processing and timestep tracking.
        Subclasses should override to add reward shaping, statistics, etc.
        
        Args:
            action: Action from NEAT (continuous values)
            otherAction: Optional action for opponent (for self-play)
            
        Returns:
            observation: 12-dim state vector
            reward: Reward from environment
            done: Whether episode is finished
            info: Additional information dict
        """
        # Convert continuous action to binary
        binary_action = self._process_action(action)

        # Handle opponent action if provided
        if otherAction is not None:
            other_binary = self._process_action(otherAction)
            obs, reward, done, info = self.env.step(binary_action, other_binary)
        else:
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
        
        Subclasses should override to reset additional state (reward shapers, etc.).
        
        Returns:
            observation: Initial 12-dim state vector
        """
        self.t = 0
        obs = self.env.reset()
        return obs

    def render(self, mode="human", close=False):
        """
        Render the environment using slimevolleygym's native rendering.
        
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
