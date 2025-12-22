"""
TaskAllocator - Market-based task allocation for multi-robot systems.

This module implements a market-based allocation strategy where:
- Robots bid on frontiers based on utility
- Crowding penalties prevent clustering
- Persistence bias reduces goal switching
- Global best-first assignment ensures optimal allocation
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional, Any
from robot.robot_state import RobotState


class TaskAllocator:
    """
    Handles multi-robot task allocation using market-based approach.

    The allocator iteratively assigns frontiers to robots by selecting
    the robot-frontier pair with highest global utility, considering:
    - Individual utility (from utility calculator)
    - Crowding penalties (avoid clustering)
    - Persistence bias (prefer keeping current goal)
    """

    def __init__(
        self,
        utility_calculator,
        crowding_penalty_weight: float,
        crowding_radius: float,
        persistence_bias: float = 5.0
    ):
        """
        Initialize the task allocator.

        Args:
            utility_calculator: FrontierUtilityCalculator instance
            crowding_penalty_weight: Weight for crowding penalty
            crowding_radius: Radius within which robots interfere
            persistence_bias: Bonus for keeping current goal
        """
        self.utility_calculator = utility_calculator
        self.crowding_penalty_weight = crowding_penalty_weight
        self.crowding_radius = crowding_radius
        self.persistence_bias = persistence_bias

    def allocate_tasks(
        self,
        robot_states: List[RobotState],
        robot_positions: List[np.ndarray],
        frontiers: List[Dict[str, Any]],
        current_step: int
    ) -> Dict[int, Dict[str, Any]]:
        """
        Allocate frontiers to robots using global best-first selection.

        Args:
            robot_states: List of RobotState objects
            robot_positions: List of robot positions (parallel to robot_states)
            frontiers: List of frontier dicts with 'pos', 'grid_pos', 'size'
            current_step: Current simulation step (for blacklist cleanup)

        Returns:
            Dictionary mapping robot_id to assigned frontier dict.
            Returns empty dict for robots with no assignment.
        """
        if not frontiers:
            return {}

        # 1. Clean up blacklists for all robots
        for state in robot_states:
            state.cleanup_blacklist(current_step)

        # 2. Track assignments for crowding calculation
        current_round_goals = []
        unassigned_indices = list(range(len(robot_states)))
        assignments = {}

        # 3. Iteratively assign best robot-frontier pairs
        while unassigned_indices:
            best_global_utility = -float('inf')
            best_pair = None

            # 4. Find best robot-frontier pair globally
            for idx in unassigned_indices:
                state = robot_states[idx]
                robot_pos = robot_positions[idx]

                # Current goal for persistence bias
                current_goal_pos = None
                if state.goal:
                    current_goal_pos = np.array(state.goal)

                # Evaluate each frontier
                for frontier in frontiers:
                    # Skip blacklisted frontiers
                    if frontier['grid_pos'] in state.blacklisted_goals:
                        continue

                    # Calculate base utility
                    target_pos = np.array(frontier['pos'])
                    utility, _ = self.utility_calculator.calculate(
                        robot_position=robot_pos,
                        exploration_direction=state.exploration_direction,
                        frontier_position=target_pos,
                        frontier_size=frontier['size']
                    )

                    # Apply crowding penalty
                    crowding_penalty = self._calculate_crowding_penalty(
                        target_pos,
                        current_round_goals
                    )

                    # Apply persistence bias (prefer keeping current goal)
                    persistence_bonus = 0.0
                    if current_goal_pos is not None:
                        dist_to_current = np.linalg.norm(target_pos - current_goal_pos)
                        if dist_to_current < 2.0:
                            persistence_bonus = self.persistence_bias

                    # Final utility after all adjustments
                    final_utility = utility - crowding_penalty + persistence_bonus

                    # Track best pair globally
                    if final_utility > best_global_utility:
                        best_global_utility = final_utility
                        best_pair = (idx, frontier, final_utility)

            # 5. Assign best pair if found
            if best_pair:
                winner_idx, winning_frontier, final_utility = best_pair
                winner_state = robot_states[winner_idx]

                # Record assignment
                assignments[winner_state.id] = {
                    'frontier': winning_frontier,
                    'utility': final_utility,
                    'should_replan': self._should_replan(
                        winner_state.goal,
                        winning_frontier['pos']
                    )
                }

                # Track for crowding calculation
                current_round_goals.append(winning_frontier['pos'])
                unassigned_indices.remove(winner_idx)
            else:
                # No valid assignments remaining
                break

        return assignments

    def _calculate_crowding_penalty(
        self,
        target_pos: np.ndarray,
        assigned_goals: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate crowding penalty for a target position.

        Penalizes frontiers that are close to already-assigned goals
        to encourage spatial distribution of robots.

        Args:
            target_pos: Position to evaluate
            assigned_goals: List of already-assigned goal positions

        Returns:
            Crowding penalty value (higher = more crowded)
        """
        crowding_penalty = 0.0

        for assigned_goal in assigned_goals:
            assigned_pos = np.array(assigned_goal)
            distance = np.linalg.norm(target_pos - assigned_pos)

            if distance < self.crowding_radius:
                # Linear falloff: closer = higher penalty
                factor = 1.0 - (distance / self.crowding_radius)
                crowding_penalty += factor * self.crowding_penalty_weight

        return crowding_penalty

    def _should_replan(
        self,
        old_goal: Optional[Tuple[float, float]],
        new_goal: Tuple[float, float]
    ) -> bool:
        """
        Determine if robot should replan path.

        Replanning is skipped if the new goal is very close to the old one
        to avoid unnecessary computation and behavior disruption.

        Args:
            old_goal: Previous goal position (None if no goal)
            new_goal: New goal position

        Returns:
            True if should replan, False to keep existing path
        """
        if old_goal is None:
            return True

        distance_change = np.linalg.norm(
            np.array(new_goal) - np.array(old_goal)
        )

        # Replan only if goal moved significantly
        return distance_change >= 2.0
