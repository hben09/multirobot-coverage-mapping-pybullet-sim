"""
Geometric utility functions for navigation and control.

These functions perform mathematical computations for robot navigation.
They are separate from the hardware layer because they represent
navigation logic, not hardware capabilities.

A real laser scanner doesn't know where your goal is - these calculations
belong in your autonomy stack, not your driver.
"""

import numpy as np
from typing import Tuple


def distance_to(current_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two positions.

    Args:
        current_pos: (x, y) current position
        target_pos: (x, y) target position

    Returns:
        Euclidean distance in meters
    """
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    return np.sqrt(dx**2 + dy**2)


def angle_to(current_pos: Tuple[float, float], current_yaw: float, target_pos: Tuple[float, float]) -> float:
    """
    Calculate angle difference from current heading to target position.

    Args:
        current_pos: (x, y) current position
        current_yaw: Current heading in radians
        target_pos: (x, y) target position

    Returns:
        Angle difference in radians, normalized to [-pi, pi]
        Positive means target is to the left, negative means right
    """
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]

    desired_angle = np.arctan2(dy, dx)
    angle_diff = desired_angle - current_yaw

    # Normalize to [-pi, pi]
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

    return angle_diff


def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to [-pi, pi].

    Args:
        angle: Angle in radians

    Returns:
        Normalized angle in radians
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def heading_vector(yaw: float) -> np.ndarray:
    """
    Convert yaw angle to unit heading vector.

    Args:
        yaw: Heading angle in radians

    Returns:
        Unit vector [cos(yaw), sin(yaw)]
    """
    return np.array([np.cos(yaw), np.sin(yaw)])
