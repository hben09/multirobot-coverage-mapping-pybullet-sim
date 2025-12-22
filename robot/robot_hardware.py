"""
RobotHardware - Interface to physical/simulated robot hardware.

This module handles all PyBullet interactions, keeping hardware-specific
code separate from state and behavior logic.
"""

import pybullet as p
import numpy as np
from typing import Tuple, List


class RobotHardware:
    """
    Hardware interface for robot (PyBullet wrapper).

    Handles all low-level physics interactions:
    - Reading pose (position, orientation)
    - Setting velocities
    - LIDAR scanning
    - Physics queries
    """

    def __init__(self, pybullet_id: int):
        """
        Initialize hardware interface.

        Args:
            pybullet_id: The PyBullet body ID for this robot
        """
        self.id = pybullet_id

    # === POSE READING ===

    def get_pose(self) -> Tuple[np.ndarray, float]:
        """
        Get robot position and orientation from physics.

        Returns:
            Tuple of (position_2d, yaw_angle)
            - position_2d: np.array([x, y])
            - yaw_angle: rotation in radians
        """
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        return np.array([pos[0], pos[1]]), yaw

    def get_position_3d(self) -> Tuple[float, float, float]:
        """
        Get full 3D position from physics.

        Returns:
            (x, y, z) position tuple
        """
        pos, _ = p.getBasePositionAndOrientation(self.id)
        return pos[0], pos[1], pos[2]

    def get_heading_vector(self) -> np.ndarray:
        """
        Get the robot's current heading as a unit vector.

        Returns:
            np.array([cos(yaw), sin(yaw)])
        """
        _, yaw = self.get_pose()
        return np.array([np.cos(yaw), np.sin(yaw)])

    # === VELOCITY CONTROL ===

    def set_velocity(self, linear: float, angular: float):
        """
        Send velocity command to robot.

        Args:
            linear: Linear velocity (m/s)
            angular: Angular velocity (rad/s)
        """
        _, yaw = self.get_pose()

        # Convert linear velocity to x,y components
        vx = linear * np.cos(yaw)
        vy = linear * np.sin(yaw)

        # Apply velocity to physics
        p.resetBaseVelocity(
            self.id,
            linearVelocity=[vx, vy, 0],
            angularVelocity=[0, 0, angular]
        )

    # === LIDAR SENSING ===

    def get_lidar_scan(self, num_rays: int = 360, max_range: float = 10.0) -> List[Tuple[float, float, bool]]:
        """
        Perform LIDAR scan around the robot.

        Args:
            num_rays: Number of rays in 360 degrees
            max_range: Maximum range of LIDAR (meters)

        Returns:
            List of (x, y, hit) tuples:
            - x, y: World coordinates of scan point
            - hit: True if obstacle detected, False if max range
        """
        # 1. Get robot pose
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        # 2. Calculate LIDAR position (slightly above robot)
        lidar_height_offset = 0.3
        lidar_z = pos[2] + lidar_height_offset

        # 3. Generate ray directions
        angles = yaw + np.linspace(0, 2.0 * np.pi, num_rays, endpoint=False)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        # 4. Build ray endpoints
        ray_from = [[pos[0], pos[1], lidar_z]] * num_rays
        ray_to = [
            [pos[0] + max_range * cos_angles[i],
             pos[1] + max_range * sin_angles[i],
             lidar_z]
            for i in range(num_rays)
        ]

        # 5. Perform batch raycasting
        results = p.rayTestBatch(ray_from, ray_to)

        # 6. Process ray results
        scan_points = []
        for i, result in enumerate(results):
            hit_object_id = result[0]
            hit_fraction = result[2]

            if hit_fraction < 1.0 and hit_object_id != self.id:
                # Hit an obstacle
                hit_position = result[3]
                scan_points.append((hit_position[0], hit_position[1], True))
            else:
                # No hit - max range
                scan_points.append((ray_to[i][0], ray_to[i][1], False))

        return scan_points

    # === UTILITY METHODS ===

    def distance_to(self, target_pos: Tuple[float, float]) -> float:
        """
        Calculate distance to a target position.

        Args:
            target_pos: (x, y) target position

        Returns:
            Euclidean distance in meters
        """
        pos, _ = self.get_pose()
        dx = target_pos[0] - pos[0]
        dy = target_pos[1] - pos[1]
        return np.sqrt(dx**2 + dy**2)

    def angle_to(self, target_pos: Tuple[float, float]) -> float:
        """
        Calculate angle difference to target position.

        Args:
            target_pos: (x, y) target position

        Returns:
            Angle difference in radians (normalized to [-pi, pi])
        """
        pos, yaw = self.get_pose()
        dx = target_pos[0] - pos[0]
        dy = target_pos[1] - pos[1]

        desired_angle = np.arctan2(dy, dx)
        angle_diff = desired_angle - yaw

        # Normalize to [-pi, pi]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        return angle_diff
