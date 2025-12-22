"""
RobotDriver - Abstract interface for robot hardware.

This defines the contract between autonomy logic and physical/simulated hardware.
Implementations can be PyBullet, ROS drivers, or real robot interfaces.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np


class RobotDriver(ABC):
    """
    Abstract interface for robot hardware control and sensing.

    This interface defines the boundary between your autonomy stack
    and the underlying hardware (simulation or physical robot).

    Professional robotics systems maintain this separation to enable:
    - Hardware independence (swap PyBullet for real robots)
    - Testing with mock drivers
    - Multiple hardware backends
    - Clear architectural boundaries
    """

    @abstractmethod
    def get_pose(self) -> Tuple[np.ndarray, float]:
        """
        Get robot position and orientation.

        Returns:
            Tuple of (position_2d, yaw_angle)
            - position_2d: np.array([x, y]) in meters
            - yaw_angle: rotation in radians
        """
        pass

    @abstractmethod
    def get_position_3d(self) -> Tuple[float, float, float]:
        """
        Get full 3D position.

        Returns:
            (x, y, z) position tuple in meters
        """
        pass

    @abstractmethod
    def get_heading_vector(self) -> np.ndarray:
        """
        Get the robot's current heading as a unit vector.

        Returns:
            np.array([cos(yaw), sin(yaw)])
        """
        pass

    @abstractmethod
    def set_velocity(self, linear: float, angular: float):
        """
        Send velocity command to robot.

        Args:
            linear: Linear velocity (m/s)
            angular: Angular velocity (rad/s)
        """
        pass

    @abstractmethod
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
        pass
