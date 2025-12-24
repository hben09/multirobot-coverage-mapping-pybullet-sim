"""
CoordinationController - Manages multi-robot task assignment and coordination.

This class now serves as a facade that orchestrates:
- Task allocation (via TaskAllocator)
- Utility calculation (via FrontierUtilityCalculator)
- Goal assignment (the "what", not the "how")

Path planning is now decentralized - each robot plans its own path to assigned goals.
This follows professional robotics patterns where a fleet manager assigns destinations,
but individual robots are responsible for navigation.
"""

import numpy as np
import pybullet as p
from coordination.utility_calculator import FrontierUtilityCalculator
from coordination.task_allocator import TaskAllocator


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
        # Create utility calculator
        self.utility_calculator = FrontierUtilityCalculator(
            direction_bias_weight=direction_bias_weight,
            size_weight=size_weight,
            distance_weight=distance_weight
        )

        # Create task allocator
        self.task_allocator = TaskAllocator(
            utility_calculator=self.utility_calculator,
            crowding_penalty_weight=crowding_penalty_weight,
            crowding_radius=crowding_radius
        )

    def assign_global_goals_auction(self, robots, frontiers, step):
        """
        Assign goals to robots using auction-based coordination.

        AUCTIONEER PATTERN:
        1. Broadcast frontiers to all agents
        2. Each agent calculates its own bid
        3. Assign tasks to highest bidders

        This approach is more realistic because the coordinator doesn't need
        to know the robot's internal state (battery, precise kinematics, etc.).

        Args:
            robots: List of RobotContainer objects
            frontiers: List of frontier dicts from frontier detection
            step: Current simulation step

        Returns:
            None (modifies robot goals in place)
        """
        if not frontiers:
            return

        # 1. Set all robots to exploration mode
        for robot in robots:
            robot.state.mode = 'GLOBAL_RELOCATE'

        # 2. Extract robot agents for auction
        robot_agents = [robot.agent for robot in robots]

        # 3. Use TaskAllocator's auction method
        assignments = self.task_allocator.allocate_tasks_auction(
            robot_agents=robot_agents,
            frontiers=frontiers,
            current_step=step
        )

        # 4. Apply assignments to robots
        for robot in robots:
            robot_id = robot.state.id

            if robot_id in assignments:
                # Robot was assigned a frontier
                assignment = assignments[robot_id]
                frontier = assignment['frontier']
                should_replan = assignment['should_replan']
                utility = assignment['utility']
                bid_debug = assignment.get('bid_debug', {})

                # If we need to replan, let the robot plan its own path
                if should_replan:
                    robot.stuck_detector.reset(robot.state, robot.driver)
                    robot.state.goal_attempts = 0

                    # Robot plans its own path to the goal
                    success = robot.agent.plan_path_to_goal(frontier['pos'])

                    if not success:
                        # Path planning failed - blacklist this frontier
                        robot.state.goal = None
                        robot.state.blacklisted_goals[frontier['grid_pos']] = step + 500
                else:
                    # Keep existing goal (no replanning needed)
                    robot.state.goal = frontier['pos']
            else:
                # Robot was not assigned - clear goal
                if robot.state.goal is not None:
                    robot.state.goal = None
                    robot.state.path = []

    def trigger_return_home(self, robots):
        """
        Trigger all robots to return to their home positions.

        The coordinator only sets the mode - each robot plans its own path home.

        Args:
            robots: List of RobotContainer objects

        Returns:
            None (modifies robot modes in place)
        """
        for robot in robots:
            robot.state.mode = 'RETURNING_HOME'

            # Robot plans its own path home
            robot.agent.plan_path_home()
