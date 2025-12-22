"""
CoordinationController - Manages multi-robot task assignment and coordination.

This class handles all coordination logic including:
- Utility calculation for frontier evaluation
- Market-based goal assignment
- Crowding penalty and persistence bias
- Path planning coordination
"""

import numpy as np
import pybullet as p


class CoordinationController:
    """Manages multi-robot coordination and goal assignment."""

    def __init__(self, direction_bias_weight, size_weight, distance_weight,
                 crowding_penalty_weight, crowding_radius):
        """
        Initialize the coordination controller.

        Args:
            direction_bias_weight: Weight for directional bias in utility
            size_weight: Weight for frontier size in utility
            distance_weight: Weight for distance cost in utility
            crowding_penalty_weight: Weight for crowding penalty
            crowding_radius: Radius within which robots interfere with each other
        """
        # Utility function weights
        self.direction_bias_weight = direction_bias_weight
        self.size_weight = size_weight
        self.distance_weight = distance_weight

        # Coordination / Crowding weights
        self.crowding_penalty_weight = crowding_penalty_weight
        self.crowding_radius = crowding_radius

    def calculate_utility(self, robot, frontier):
        """
        Calculate utility of a frontier for a robot.

        Utility Function with Direction Bias and Volumetric Gain.

        Args:
            robot: Robot object
            frontier: Frontier dict with 'pos' and 'size'

        Returns:
            tuple: (utility_value, debug_info_dict)
        """
        pos, orn = p.getBasePositionAndOrientation(robot.id)
        robot_pos = np.array([pos[0], pos[1]])
        frontier_pos = np.array(frontier['pos'])

        dist = np.linalg.norm(frontier_pos - robot_pos)
        distance_cost = dist * self.distance_weight
        size_gain = frontier['size'] * self.size_weight

        to_frontier = frontier_pos - robot_pos
        to_frontier_norm = np.linalg.norm(to_frontier)

        if to_frontier_norm > 0.1:
            to_frontier_unit = to_frontier / to_frontier_norm
            alignment = np.dot(robot.exploration_direction, to_frontier_unit)
            alignment_normalized = (alignment + 1.0) / 2.0
            direction_bonus = alignment_normalized * self.direction_bias_weight
        else:
            alignment = 0.0
            direction_bonus = 0.0

        utility = size_gain - distance_cost + direction_bonus
        return utility, {}

    def assign_global_goals(self, robots, frontiers, step, world_to_grid_fn, plan_path_fn):
        """
        Assign goals to robots using market-based coordination (Global Best-First).

        Args:
            robots: List of robot objects
            frontiers: List of frontier dicts from frontier detection
            step: Current simulation step
            world_to_grid_fn: Function to convert world coords to grid coords
            plan_path_fn: Function to plan path (start_grid, goal_grid) -> path

        Returns:
            None (modifies robot goals and paths in place)
        """
        if not frontiers:
            return

        active_robots = robots
        for robot in active_robots:
            robot.mode = 'GLOBAL_RELOCATE'
            robot.cleanup_blacklist(step)

        current_round_goals = []
        unassigned_robots = active_robots.copy()

        while unassigned_robots:
            best_global_utility = -float('inf')
            best_pair = None

            for robot in unassigned_robots:
                current_goal_pos = None
                if robot.goal:
                    current_goal_pos = np.array(robot.goal)

                for f in frontiers:
                    if f['grid_pos'] in robot.blacklisted_goals:
                        continue

                    target_pos = np.array(f['pos'])
                    util, _ = self.calculate_utility(robot, f)

                    # Calculate crowding penalty
                    crowding_penalty = 0.0
                    for assigned_goal in current_round_goals:
                        dist_to_assigned = np.linalg.norm(target_pos - np.array(assigned_goal))
                        if dist_to_assigned < self.crowding_radius:
                            factor = 1.0 - (dist_to_assigned / self.crowding_radius)
                            crowding_penalty += factor * self.crowding_penalty_weight

                    # Persistence bias - prefer to keep current goal
                    persistence_bias = 5.0
                    if current_goal_pos is not None:
                        dist_to_current = np.linalg.norm(target_pos - current_goal_pos)
                        if dist_to_current < 2.0:
                            util += persistence_bias

                    final_utility = util - crowding_penalty

                    if final_utility > best_global_utility:
                        best_global_utility = final_utility
                        best_pair = (robot, f)

            if best_pair:
                winner_robot, winning_frontier = best_pair
                target_pos = winning_frontier['pos']

                old_goal = winner_robot.goal
                winner_robot.goal = target_pos

                should_replan = True
                if old_goal is not None:
                    dist_change = np.linalg.norm(np.array(target_pos) - np.array(old_goal))
                    if dist_change < 2.0:
                        should_replan = False

                if should_replan:
                    winner_robot.reset_stuck_state()
                    winner_robot.goal_attempts = 0

                    pos, _ = p.getBasePositionAndOrientation(winner_robot.id)
                    start_grid = world_to_grid_fn(pos[0], pos[1])
                    path = plan_path_fn(start_grid, winning_frontier['grid_pos'])

                    if path:
                        winner_robot.path = path
                        print(f"Robot {winner_robot.id}: Assigned goal (Utility: {best_global_utility:.1f})")
                    else:
                        winner_robot.goal = None
                        winner_robot.blacklisted_goals[winning_frontier['grid_pos']] = step + 500
                        print(f"Robot {winner_robot.id}: Path failed to {winning_frontier['grid_pos']}, blacklisted.")

                if winner_robot.goal is not None:
                    current_round_goals.append(target_pos)
                    unassigned_robots.remove(winner_robot)

            else:
                break

        # Clear goals for unassigned robots
        for loser_robot in unassigned_robots:
            if loser_robot.goal is not None:
                loser_robot.goal = None
                loser_robot.path = []

    def plan_return_path(self, robot, world_to_grid_fn, plan_path_fn):
        """
        Plan a path home for a single robot.

        Args:
            robot: Robot object
            world_to_grid_fn: Function to convert world coords to grid coords
            plan_path_fn: Function to plan path (start_grid, goal_grid) -> path

        Returns:
            None (modifies robot goal and path in place)
        """
        pos, _ = p.getBasePositionAndOrientation(robot.id)
        start_grid = world_to_grid_fn(pos[0], pos[1])
        home_grid = world_to_grid_fn(robot.home_position[0], robot.home_position[1])

        robot.path = plan_path_fn(start_grid, home_grid)
        if robot.path:
            robot.goal = tuple(robot.home_position)
            robot.reset_stuck_state()

    def trigger_return_home(self, robots, world_to_grid_fn, plan_path_fn):
        """
        Trigger all robots to return to their home positions.

        Args:
            robots: List of robot objects
            world_to_grid_fn: Function to convert world coords to grid coords
            plan_path_fn: Function to plan path (start_grid, goal_grid) -> path

        Returns:
            None (modifies robot modes, goals, and paths in place)
        """
        for robot in robots:
            robot.mode = 'RETURNING_HOME'
            self.plan_return_path(robot, world_to_grid_fn, plan_path_fn)
            print(f"Robot {robot.id}: Returning home with {len(robot.path)} waypoints")

    def execute_exploration_logic(self, robot, step, plan_return_path_fn):
        """
        Execute exploration logic for a single robot.

        Args:
            robot: Robot object
            step: Current simulation step
            plan_return_path_fn: Function to plan return path for robot

        Returns:
            None (commands robot movement)
        """
        if robot.mode == 'HOME':
            robot.move(0.0, 0.0)
            return

        if robot.goal:
            if robot.check_if_stuck(threshold=0.2, stuck_limit=200):
                print(f"Robot {robot.id} is stuck! Abandoning goal and replanning...")
                robot.goal = None
                robot.path = []
                robot.reset_stuck_state()
                robot.goal_attempts += 1

                if robot.goal_attempts > robot.max_goal_attempts:
                    robot.move(0.0, 2.0)
                    robot.goal_attempts = 0
                return

            # Follow path requires mapper context
            # This stays in the main class for now
            return 'FOLLOW_PATH'
        else:
            if robot.mode == 'RETURNING_HOME':
                plan_return_path_fn(robot)
                if robot.goal:
                    return 'FOLLOW_PATH'
                else:
                    robot.move(0.0, 0.5)
            else:
                robot.move(0.0, 0.5)

        return None
