"""
ExplorationDirectionTracker - Tracks and smooths robot exploration direction.

Computes the general direction of robot movement from trajectory history
using a low-pass filter for smoothness.
"""

import numpy as np
from robot.robot_state import RobotState


class ExplorationDirectionTracker:
    """
    Tracks the general direction of robot exploration.

    Analyzes recent trajectory to determine the robot's exploration
    direction, applying smoothing to avoid jitter.
    """

    def __init__(
        self,
        window_size: int = 20,
        min_distance: float = 0.5,
        smoothing_alpha: float = 0.3
    ):
        """
        Initialize the exploration direction tracker.

        Args:
            window_size: Number of recent trajectory points to analyze
            min_distance: Minimum distance traveled to update direction
            smoothing_alpha: Low-pass filter coefficient (0-1, higher = more responsive)
        """
        self.window_size = window_size
        self.min_distance = min_distance
        self.smoothing_alpha = smoothing_alpha

    def update(self, state: RobotState):
        """
        Update the exploration direction based on recent trajectory.

        Modifies state.exploration_direction in place.

        Args:
            state: Robot state containing trajectory and exploration_direction
        """
        if len(state.trajectory) < 2:
            return

        # 1. Get recent trajectory segment
        recent_points = state.trajectory[-min(self.window_size, len(state.trajectory)):]

        if len(recent_points) >= 2:
            # 2. Calculate direction from start to end of segment
            start = np.array(recent_points[0])
            end = np.array(recent_points[-1])
            direction = end - start

            norm = np.linalg.norm(direction)
            if norm > self.min_distance:
                direction = direction / norm

                # 3. Apply low-pass filter for smoothing
                state.exploration_direction = (
                    self.smoothing_alpha * direction +
                    (1 - self.smoothing_alpha) * state.exploration_direction
                )
                # Normalize
                state.exploration_direction /= np.linalg.norm(state.exploration_direction)
