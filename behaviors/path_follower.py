"""
PathFollower - Pure pursuit path following controller.

Computes velocity commands to follow a path represented as a list
of waypoints in grid coordinates.
"""

import numpy as np
from typing import Tuple, Optional
from robot.robot_state import RobotState
from robot.robot_hardware import RobotHardware


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
        hardware: RobotHardware,
        grid_to_world_fn
    ) -> Tuple[float, float]:
        """
        Compute velocity commands to follow the path.

        Args:
            state: Robot state containing path and goal
            hardware: Hardware interface for position queries
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

        # 2. Get next waypoint if path exists
        if state.path:
            next_wp = state.path[0]
            wx, wy = grid_to_world_fn(next_wp[0], next_wp[1])
            target_pos = (wx, wy)

            # 3. Check if waypoint reached
            dist = hardware.distance_to((wx, wy))

            if dist < self.waypoint_threshold:
                state.path.pop(0)
                # Recursively compute for next waypoint
                if state.path:
                    return self.compute_velocities(state, hardware, grid_to_world_fn)

        # 4. Calculate direction to target using hardware
        distance = hardware.distance_to(target_pos)
        angle_diff = hardware.angle_to(target_pos)

        # 5. Check if final goal reached
        if distance < self.goal_threshold and not state.path:
            state.goal = None
            return 0.0, 0.0

        # 6. Compute velocities with limits
        linear_vel = min(self.max_linear_vel, distance * 2.0)
        angular_vel = np.clip(angle_diff * 4.0, -self.max_angular_vel, self.max_angular_vel)

        # 7. Slow down for sharp turns
        if abs(angle_diff) > self.turn_slowdown_angle:
            linear_vel *= 0.3

        return linear_vel, angular_vel
