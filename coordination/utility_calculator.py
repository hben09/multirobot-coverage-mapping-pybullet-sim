"""
FrontierUtilityCalculator - Calculates utility values for frontier assignments.

This module computes how valuable a frontier is for a given robot based on:
- Distance to frontier (cost)
- Frontier size (gain/information value)
- Alignment with robot's exploration direction (directional bias)
"""

import numpy as np
from typing import Tuple, Dict, Any


class FrontierUtilityCalculator:
    """
    Calculates utility of frontiers for robots.

    The utility function balances multiple factors:
    1. Size gain - Larger frontiers provide more information
    2. Distance cost - Closer frontiers are preferred
    3. Direction bonus - Frontiers aligned with exploration direction
    """

    def __init__(
        self,
        direction_bias_weight: float,
        size_weight: float,
        distance_weight: float
    ):
        """
        Initialize the utility calculator.

        Args:
            direction_bias_weight: Weight for directional alignment bonus
            size_weight: Weight for frontier size gain
            distance_weight: Weight for distance cost (penalty)
        """
        self.direction_bias_weight = direction_bias_weight
        self.size_weight = size_weight
        self.distance_weight = distance_weight

    def calculate(
        self,
        robot_position: np.ndarray,
        exploration_direction: np.ndarray,
        frontier_position: np.ndarray,
        frontier_size: int
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate utility of a frontier for a robot.

        Args:
            robot_position: Robot's current position (x, y)
            exploration_direction: Robot's exploration direction unit vector
            frontier_position: Frontier's position (x, y)
            frontier_size: Size of the frontier (number of cells)

        Returns:
            Tuple of (utility_value, debug_info_dict)
            - utility_value: Combined utility score
            - debug_info: Dictionary with breakdown of components
        """
        # 1. Calculate distance cost
        to_frontier = frontier_position - robot_position
        distance = np.linalg.norm(to_frontier)
        distance_cost = distance * self.distance_weight

        # 2. Calculate size gain
        size_gain = frontier_size * self.size_weight

        # 3. Calculate direction alignment bonus
        to_frontier_norm = np.linalg.norm(to_frontier)

        if to_frontier_norm > 0.1:
            # Normalize direction to frontier
            to_frontier_unit = to_frontier / to_frontier_norm

            # Dot product gives alignment (-1 to 1)
            alignment = np.dot(exploration_direction, to_frontier_unit)

            # Normalize to [0, 1] range
            alignment_normalized = (alignment + 1.0) / 2.0

            # Apply weight
            direction_bonus = alignment_normalized * self.direction_bias_weight
        else:
            # Frontier is very close - no directional preference
            alignment = 0.0
            direction_bonus = 0.0

        # 4. Combine into final utility
        utility = size_gain - distance_cost + direction_bonus

        # 5. Build debug info
        debug_info = {
            'distance': distance,
            'distance_cost': distance_cost,
            'size_gain': size_gain,
            'alignment': alignment if to_frontier_norm > 0.1 else 0.0,
            'direction_bonus': direction_bonus,
            'utility': utility
        }

        return utility, debug_info
