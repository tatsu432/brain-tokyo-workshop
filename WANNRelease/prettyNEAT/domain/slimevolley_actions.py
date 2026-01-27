"""
Action Processing Module for SlimeVolley Environments

This module provides shared action conversion logic to eliminate duplication
between different SlimeVolley environment variants.

Clean Architecture: Separates action processing concerns from environment logic.
"""

from typing import Union

import numpy as np


class SlimeVolleyActionProcessor:
    """
    Converts NEAT/WANN continuous outputs to SlimeVolley binary actions.

    Supports multiple action modes:
    1. Discrete mode: 6 discrete actions (0-5) mapped to action combinations
    2. Continuous mode: 3D continuous values [forward, jump, back] with thresholds
    3. Legacy mode: 2D continuous values [horizontal, jump] or 1D continuous value
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

    def process(
        self, action: Union[int, float, np.ndarray], threshold: float = 3/4
    ) -> np.ndarray:
        """
        Convert NEAT output to binary actions [forward, backward, jump].

        Supports multiple input formats:
        - Discrete index (int): 0-5 for discrete action selection (legacy 6-action mode)
        - Continuous array (3D): [forward, jump, back] with threshold-based activation
        - Continuous array (2D): [horizontal, jump] with thresholds (legacy mode)
        - Continuous scalar (1D): Single value mapped to discrete zones (legacy mode)

        Args:
            action: Action from NEAT network (various formats supported)
            threshold: Threshold for activating actions (default: 0.0)

        Returns:
            binary_action: numpy array of shape (3,) with binary values
                          [forward, backward, jump] where each is 0 or 1
        """
        # Handle discrete action index (from 'prob' actionSelect - legacy 6-action mode)
        if isinstance(action, (int, np.integer)):
            action_idx = int(action) % 6  # Ensure valid index
            return np.array(self.DISCRETE_ACTION_MAP[action_idx], dtype=np.int8)

        # Handle continuous action (legacy mode or different actionSelect)
        action = np.array(action).flatten()

        # If 3 outputs: [forward, jump, back] - threshold-based activation
        if len(action) == 3:
            if self.clip_actions:
                action = np.clip(action, -1.0, 1.0)
            forward = 1 if action[0] > threshold else 0
            jump = 1 if action[1] > threshold else 0
            back = 1 if action[2] > threshold else 0
            return np.array([forward, back, jump], dtype=np.int8)

        # If 6 outputs (discrete mode but passed as array), take argmax (legacy)
        if len(action) == 6:
            action_idx = np.argmax(action)
            return np.array(self.DISCRETE_ACTION_MAP[action_idx], dtype=np.int8)

        # Legacy 2D mode: [horizontal, jump] - horizontal controls forward/backward
        if len(action) == 2:
            if self.clip_actions:
                action = np.clip(action, -1.0, 1.0)
            # Horizontal: negative = backward, positive = forward
            forward = 1 if action[0] > threshold else 0
            back = 1 if action[0] < -threshold else 0
            jump = 1 if action[1] > threshold else 0
            return np.array([forward, back, jump], dtype=np.int8)

        # Legacy 1D mode: single value mapped to zones
        if len(action) == 1:
            val = float(action[0])
            if self.clip_actions:
                val = np.clip(val, -1.0, 1.0)
            # Map single value to actions
            forward = 1 if val > 0.33 else 0
            back = 1 if val < -0.33 else 0
            jump = 1 if abs(val) > 0.5 else 0
            return np.array([forward, back, jump], dtype=np.int8)

        # Fallback: raise error for unsupported action format
        raise ValueError(
            f"Unsupported action format: expected 1, 2, 3, or 6 outputs, got {len(action)}"
        )
