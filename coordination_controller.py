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

    def calculate_utility(self, robot, frontier):
        """
        Calculate utility of a frontier for a robot.

        DEPRECATED: This method is kept for compatibility but delegates
        to the FrontierUtilityCalculator.

        Args:
            robot: RobotContainer object
            frontier: Frontier dict with 'pos' and 'size'

        Returns:
            tuple: (utility_value, debug_info_dict)
        """
        # Get robot position from hardware
        pos, orn = p.getBasePositionAndOrientation(robot.state.id)
        robot_pos = np.array([pos[0], pos[1]])
        frontier_pos = np.array(frontier['pos'])

        # Delegate to utility calculator
        return self.utility_calculator.calculate(
            robot_position=robot_pos,
            exploration_direction=robot.state.exploration_direction,
            frontier_position=frontier_pos,
            frontier_size=frontier['size']
        )

    def assign_global_goals(self, robots, frontiers, step):
        """
        Assign goals to robots using market-based coordination (Global Best-First).

        The coordinator now ONLY assigns goals (the "what"), not paths (the "how").
        Each robot is responsible for planning its own path to the assigned goal.

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

        # 2. Extract robot states and positions for allocator
        robot_states = [robot.state for robot in robots]
        robot_positions = []
        for robot in robots:
            pos, _ = p.getBasePositionAndOrientation(robot.state.id)
            robot_positions.append(np.array([pos[0], pos[1]]))

        # 3. Use TaskAllocator to get assignments
        assignments = self.task_allocator.allocate_tasks(
            robot_states=robot_states,
            robot_positions=robot_positions,
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

                # If we need to replan, let the robot plan its own path
                if should_replan:
                    robot.stuck_detector.reset(robot.state, robot.driver)
                    robot.state.goal_attempts = 0

                    # Robot plans its own path to the goal
                    success = robot.agent.plan_path_to_goal(frontier['pos'])

                    if success:
                        print(f"Robot {robot_id}: Assigned goal (Utility: {utility:.1f})")
                    else:
                        # Path planning failed - blacklist this frontier
                        robot.state.goal = None
                        robot.state.blacklisted_goals[frontier['grid_pos']] = step + 500
                        print(f"Robot {robot_id}: Path failed to {frontier['grid_pos']}, blacklisted.")
                else:
                    # Keep existing goal (no replanning needed)
                    robot.state.goal = frontier['pos']
            else:
                # Robot was not assigned - clear goal
                if robot.state.goal is not None:
                    robot.state.goal = None
                    robot.state.path = []

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

                    if success:
                        print(f"Robot {robot_id}: Assigned goal (Bid: {bid_debug.get('utility', utility):.1f}, "
                              f"Final: {utility:.1f})")
                    else:
                        # Path planning failed - blacklist this frontier
                        robot.state.goal = None
                        robot.state.blacklisted_goals[frontier['grid_pos']] = step + 500
                        print(f"Robot {robot_id}: Path failed to {frontier['grid_pos']}, blacklisted.")
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
            success = robot.agent.plan_path_home()

            if success:
                print(f"Robot {robot.state.id}: Returning home with {len(robot.state.path)} waypoints")
            else:
                print(f"Robot {robot.state.id}: Failed to plan path home")
