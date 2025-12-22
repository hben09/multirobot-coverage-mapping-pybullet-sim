"""
PathFollower - Pure pursuit path following controller.

Computes velocity commands to follow a path represented as a list
of waypoints in grid coordinates.
"""

import numpy as np
from typing import Tuple, Optional
from robot.robot_state import RobotState
from robot.robot_driver import RobotDriver
from utils.geometry import distance_to, angle_to


class PathFollower:
    """
    Pure pursuit path following controller.

    Generates linear and angular velocity commands to follow a path,
    with automatic waypoint advancement and goal reaching detection.
    """

    def __init__(
        self,
        waypoint_threshold: float = 0.6,
        goal_threshold: float = 0.5,
        max_linear_vel: float = 8.0,
        max_angular_vel: float = 4.0,
        turn_slowdown_angle: float = 0.5
    ):
        """
        Initialize the path follower.

        Args:
            waypoint_threshold: Distance to waypoint before advancing (meters)
            goal_threshold: Distance to final goal before stopping (meters)
            max_linear_vel: Maximum linear velocity (m/s)
            max_angular_vel: Maximum angular velocity (rad/s)
            turn_slowdown_angle: Angle threshold for slowing down (radians)
        """
        self.waypoint_threshold = waypoint_threshold
        self.goal_threshold = goal_threshold
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.turn_slowdown_angle = turn_slowdown_angle

    def compute_velocities(
        self,
        state: RobotState,
        driver: RobotDriver,
        grid_to_world_fn
    ) -> Tuple[float, float]:
        """
        Compute velocity commands to follow the path.

        Args:
            state: Robot state containing path and goal
            driver: Hardware driver for position queries
            grid_to_world_fn: Function to convert grid coords to world coords
                             Should accept (grid_x, grid_y) and return (world_x, world_y)

        Returns:
            Tuple of (linear_vel, angular_vel)
            Returns (0, 0) if no path or goal
        """
        # 1. Check if there's anything to follow
        if not state.path and state.goal is None:
            return 0.0, 0.0

        target_pos = state.goal

        # 2. Get current pose from driver
        current_pos, current_yaw = driver.get_pose()

        # 3. Get next waypoint if path exists
        if state.path:
            next_wp = state.path[0]
            wx, wy = grid_to_world_fn(next_wp[0], next_wp[1])
            target_pos = (wx, wy)

            # 4. Check if waypoint reached
            dist = distance_to((current_pos[0], current_pos[1]), (wx, wy))

            if dist < self.waypoint_threshold:
                state.path.pop(0)
                # Recursively compute for next waypoint
                if state.path:
                    return self.compute_velocities(state, driver, grid_to_world_fn)

        # 5. Calculate direction to target using geometry utilities
        distance = distance_to((current_pos[0], current_pos[1]), target_pos)
        angle_diff = angle_to((current_pos[0], current_pos[1]), current_yaw, target_pos)

        # 6. Check if final goal reached
        if distance < self.goal_threshold and not state.path:
            state.goal = None
            return 0.0, 0.0

        # 7. Compute velocities with limits
        linear_vel = min(self.max_linear_vel, distance * 2.0)
        angular_vel = np.clip(angle_diff * 4.0, -self.max_angular_vel, self.max_angular_vel)

        # 8. Slow down for sharp turns
        if abs(angle_diff) > self.turn_slowdown_angle:
            linear_vel *= 0.3

        return linear_vel, angular_vel
