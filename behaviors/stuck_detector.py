"""
StuckDetector - Detects when a robot is stuck.

Monitors robot movement and detects when the robot hasn't moved
significantly for a prolonged period.
"""

import numpy as np
from typing import Tuple
from robot.robot_state import RobotState
from robot.robot_driver import RobotDriver


class StuckDetector:
    """
    Detects when a robot is stuck based on movement history.

    The detector tracks position changes and increments a counter
    when the robot hasn't moved enough, resetting when movement is detected.
    """

    def __init__(self, threshold: float = 0.3, stuck_limit: int = 200):
        """
        Initialize the stuck detector.

        Args:
            threshold: Minimum distance (meters) to consider as movement
            stuck_limit: Number of checks below threshold before declaring stuck
        """
        self.threshold = threshold
        self.stuck_limit = stuck_limit

    def is_stuck(self, state: RobotState, driver: RobotDriver) -> bool:
        """
        Check if the robot is stuck.

        Updates the robot state's stuck counter based on movement.

        Args:
            state: Robot state containing stuck counter and last position
            driver: Hardware driver to get current position

        Returns:
            True if robot is stuck (counter > limit), False otherwise
        """
        # 1. Get current position from driver
        current_pos, _ = driver.get_pose()

        # 2. Check distance moved since last check
        distance_moved = np.linalg.norm(current_pos - state.last_position)

        # 3. Update stuck counter
        if distance_moved < self.threshold:
            state.stuck_counter += 1
        else:
            state.stuck_counter = 0
            state.last_position = current_pos

        return state.stuck_counter > self.stuck_limit

    def reset(self, state: RobotState, driver: RobotDriver):
        """
        Reset the stuck detection state.

        Clears the stuck counter and updates the last known position.

        Args:
            state: Robot state to reset
            driver: Hardware driver to get current position
        """
        state.stuck_counter = 0
        current_pos, _ = driver.get_pose()
        state.last_position = current_pos
