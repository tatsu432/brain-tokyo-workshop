"""
Base SlimeVolley Environment Wrapper

This module provides a common base class for all SlimeVolley environment variants.
Follows clean architecture by separating environment wrapping from domain-specific logic.

All SlimeVolley wrappers should inherit from this base class to ensure consistency.
"""

import contextlib
import logging
import os
import sys
import warnings
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils.seeding import np_random

# Suppress gym step API deprecation warning (from old gym package used by slimevolleygym)
# Must be set before importing old gym to catch warnings during import
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*Initializing environment in old step API.*",
)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="gym.wrappers.step_api_compatibility"
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
# Suppress Gym/NumPy 2.0 compatibility warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*does not support NumPy 2.0.*")
warnings.filterwarnings("ignore", message=".*upgrade to Gymnasium.*")

from domain.slimevolley_actions import SlimeVolleyActionProcessor

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


# Try to import slimevolleygym
try:
    # Suppress warnings and stderr during import (gym prints directly to stderr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with suppress_stderr():
            import gym as old_gym  # slimevolleygym uses old gym API
            import slimevolleygym

    # Apply rendering patch to fix gym rendering compatibility issues
    try:
        from domain.slimevolley_rendering_patch import apply_rendering_patch

        apply_rendering_patch()
        logger.debug("Applied slimevolleygym rendering patch")
    except Exception as e:
        logger.warning(f"Could not apply rendering patch: {e}")

    SLIMEVOLLEY_AVAILABLE = True
except ImportError:
    SLIMEVOLLEY_AVAILABLE = False
    old_gym = None
    logger.warning(
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
        # Try to directly instantiate from slimevolleygym to avoid wrapper issues
        # slimevolleygym uses old gym API, not gymnasium
        # Disable env checker to avoid numpy 2.x compatibility issues
        # Suppress stderr to hide gym deprecation warnings (gym prints directly to stderr)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            with suppress_stderr():
                # Try to directly import and instantiate SlimeVolleyEnv from slimevolleygym
                # This avoids any wrapper issues from gym.make()
                try:
                    from slimevolleygym.slimevolley import SlimeVolleyEnv as SlimeVolleyBaseEnv
                    # Direct instantiation avoids wrapper issues
                    self.env = SlimeVolleyBaseEnv()
                    logger.debug("Created slimevolleygym environment directly")
                except (ImportError, AttributeError):
                    # Fall back to gym.make() if direct import fails
                    self.env = old_gym.make(env_id, disable_env_checker=True)
                    logger.debug("Created environment via gym.make()")
        
        # Unwrap environment to get to the actual slimevolleygym environment
        # This is needed because gymnasium may wrap it with OrderEnforcing which only accepts one argument
        # We need the underlying environment that supports two-argument step() for self-play
        original_env = self.env
        unwrapped_env = self.env
        
        # Try to get unwrapped environment
        if hasattr(self.env, 'unwrapped'):
            unwrapped_env = self.env.unwrapped
        elif hasattr(self.env, 'env'):
            # Common wrapper pattern: wrapper.env is the wrapped environment
            unwrapped_env = self.env.env
            # Keep unwrapping if there are multiple layers
            while hasattr(unwrapped_env, 'env') and unwrapped_env.env is not unwrapped_env:
                unwrapped_env = unwrapped_env.env
        
        # Verify the unwrapped environment supports two-argument step
        import inspect
        try:
            sig = inspect.signature(unwrapped_env.step)
            # Check if step accepts at least 2 arguments (self + action + optional otherAction)
            # slimevolleygym's step signature should be: step(self, action, otherAction=None)
            if len(sig.parameters) >= 2:
                self.env = unwrapped_env
                logger.debug(f"Unwrapped environment to access underlying slimevolleygym environment")
            else:
                logger.warning(
                    f"Unwrapped environment's step() only accepts {len(sig.parameters)} arguments. "
                    f"Self-play may not work correctly. Environment type: {type(unwrapped_env)}"
                )
        except Exception as e:
            logger.warning(f"Could not inspect step signature: {e}. Using original environment.")
            self.env = original_env

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
            # slimevolleygym's step() supports two arguments: step(action, otherAction)
            # The unwrapped environment should support this
            try:
                obs, reward, done, info = self.env.step(binary_action, other_binary)
            except TypeError as e:
                # If step() doesn't accept two arguments, try with just one
                # This should not happen if unwrapping worked, but handle gracefully
                if "takes 2 positional arguments but 3 were given" in str(e):
                    logger.error(
                        f"Environment step() doesn't support two arguments. "
                        f"Environment type: {type(self.env)}, "
                        f"Has 'unwrapped': {hasattr(self.env, 'unwrapped')}, "
                        f"Has 'env': {hasattr(self.env, 'env')}"
                    )
                    # Fall back to single-argument step (opponent will use default policy)
                    obs, reward, done, info = self.env.step(binary_action)
                else:
                    raise
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
            mode: Render mode ('human' or 'rgb_array') - deprecated, kept for compatibility
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
        # Note: old gym (slimevolleygym) still uses mode parameter, so we pass it
        # but suppress the deprecation warning since this is for old gym compatibility
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=DeprecationWarning, message=".*render.*mode.*"
                )
                return self.env.render(mode=mode)
        except Exception as e:
            if not self._render_warning_shown:
                logger.warning(f"Rendering error: {e}")
                self._render_warning_shown = True
            return None

    def close(self):
        """Clean up environment resources"""
        if self.env is not None:
            self.env.close()
