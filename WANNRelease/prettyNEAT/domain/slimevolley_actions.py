"""
Action Processing Module for SlimeVolley Environments

This module provides shared action conversion logic to eliminate duplication
between different SlimeVolley environment variants.

Clean Architecture: Separates action processing concerns from environment logic.
"""

import numpy as np
from typing import Union


class SlimeVolleyActionProcessor:
    """
    Converts NEAT/WANN continuous outputs to SlimeVolley binary actions.
    
    Supports multiple action modes:
    1. Discrete mode: 6 discrete actions (0-5) mapped to action combinations
    2. Continuous mode: 2D continuous values [horizontal, jump]
    3. Legacy mode: 1D continuous value mapped to discrete zones
    """

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

    def __init__(self, clip_actions: bool = True):
        """
        Initialize action processor.
        
        Args:
            clip_actions: If True, clip continuous actions to [-1, 1] range.
                         This prevents unbounded linear outputs from saturating
                         thresholds while maintaining good gradient flow.
        """
        self.clip_actions = clip_actions

    def process(self, action: Union[int, float, np.ndarray]) -> np.ndarray:
        """
        Convert NEAT output to binary actions [forward, backward, jump].
        
        Supports multiple input formats:
        - Discrete index (int): 0-5 for discrete action selection
        - Continuous array (2D): [horizontal, jump] with thresholds
        - Continuous scalar (1D): Single value mapped to discrete zones
        
        Args:
            action: Action from NEAT network (various formats supported)
            
        Returns:
            binary_action: numpy array of shape (3,) with binary values
                          [forward, backward, jump] where each is 0 or 1
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

        # Clip actions to prevent unbounded outputs (critical for linear activations)
        if self.clip_actions:
            action = np.clip(action, -1.0, 1.0)

        # Legacy 1D continuous mode with thresholds
        if len(action) == 1:
            val = float(action[0])
            if val > 0.5:
                return np.array([1, 0, 1], dtype=np.int8)  # forward + jump
            elif val > 0.1:
                return np.array([1, 0, 0], dtype=np.int8)  # forward only
            elif val > -0.1:
                return np.array([0, 0, 1], dtype=np.int8)  # jump only
            elif val > -0.5:
                return np.array([0, 1, 0], dtype=np.int8)  # backward only
            else:
                return np.array([0, 1, 1], dtype=np.int8)  # backward + jump

        # 2D continuous mode: [horizontal_movement, jump]
        if len(action) >= 2:
            # Horizontal movement with deadband
            if action[0] > 0.3:
                forward, backward = 1, 0
            elif action[0] < -0.3:
                forward, backward = 0, 1
            else:
                # Deadband: no horizontal movement if |action[0]| < 0.3
                forward, backward = 0, 0

            # Jump threshold: higher since jumping is more deliberate
            jump = 1 if action[1] > 0.4 else 0

            return np.array([forward, backward, jump], dtype=np.int8)

        # Fallback: default to no-op
        return np.array([0, 0, 0], dtype=np.int8)
