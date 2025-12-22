"""
CoordinationController - Manages multi-robot task assignment and coordination.

This class now serves as a facade that orchestrates:
- Task allocation (via TaskAllocator)
- Utility calculation (via FrontierUtilityCalculator)
- Path planning coordination
- Behavior execution
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

    def assign_global_goals(self, robots, frontiers, step, world_to_grid_fn, plan_path_fn):
        """
        Assign goals to robots using market-based coordination (Global Best-First).

        Args:
            robots: List of RobotContainer objects
            frontiers: List of frontier dicts from frontier detection
            step: Current simulation step
            world_to_grid_fn: Function to convert world coords to grid coords
            plan_path_fn: Function to plan path (start_grid, goal_grid) -> path

        Returns:
            None (modifies robot goals and paths in place)
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

                # Update goal
                robot.state.goal = frontier['pos']

                # Plan path if needed
                if should_replan:
                    robot.stuck_detector.reset(robot.state, robot.hardware)
                    robot.state.goal_attempts = 0

                    # Get current position and plan path
                    pos, _ = p.getBasePositionAndOrientation(robot.state.id)
                    start_grid = world_to_grid_fn(pos[0], pos[1])
                    path = plan_path_fn(start_grid, frontier['grid_pos'])

                    if path:
                        robot.state.path = path
                        print(f"Robot {robot_id}: Assigned goal (Utility: {utility:.1f})")
                    else:
                        # Path planning failed - blacklist this frontier
                        robot.state.goal = None
                        robot.state.blacklisted_goals[frontier['grid_pos']] = step + 500
                        print(f"Robot {robot_id}: Path failed to {frontier['grid_pos']}, blacklisted.")
            else:
                # Robot was not assigned - clear goal
                if robot.state.goal is not None:
                    robot.state.goal = None
                    robot.state.path = []

    def plan_return_path(self, robot, world_to_grid_fn, plan_path_fn):
        """
        Plan a path home for a single robot.

        Args:
            robot: RobotContainer object
            world_to_grid_fn: Function to convert world coords to grid coords
            plan_path_fn: Function to plan path (start_grid, goal_grid) -> path

        Returns:
            None (modifies robot goal and path in place)
        """
        pos, _ = p.getBasePositionAndOrientation(robot.state.id)
        start_grid = world_to_grid_fn(pos[0], pos[1])
        home_grid = world_to_grid_fn(robot.state.home_position[0], robot.state.home_position[1])

        robot.state.path = plan_path_fn(start_grid, home_grid)
        if robot.state.path:
            robot.state.goal = tuple(robot.state.home_position)
            robot.stuck_detector.reset(robot.state, robot.hardware)

    def trigger_return_home(self, robots, world_to_grid_fn, plan_path_fn):
        """
        Trigger all robots to return to their home positions.

        Args:
            robots: List of RobotContainer objects
            world_to_grid_fn: Function to convert world coords to grid coords
            plan_path_fn: Function to plan path (start_grid, goal_grid) -> path

        Returns:
            None (modifies robot modes, goals, and paths in place)
        """
        for robot in robots:
            robot.state.mode = 'RETURNING_HOME'
            self.plan_return_path(robot, world_to_grid_fn, plan_path_fn)
            print(f"Robot {robot.state.id}: Returning home with {len(robot.state.path)} waypoints")

    def execute_exploration_logic(self, robot, step, plan_return_path_fn):
        """
        Execute exploration logic for a single robot.

        Args:
            robot: RobotContainer object
            step: Current simulation step
            plan_return_path_fn: Function to plan return path for robot

        Returns:
            None (commands robot movement)
        """
        if robot.state.mode == 'HOME':
            robot.hardware.set_velocity(0.0, 0.0)
            return

        if robot.state.goal:
            if robot.stuck_detector.is_stuck(robot.state, robot.hardware):
                print(f"Robot {robot.state.id} is stuck! Abandoning goal and replanning...")
                robot.state.goal = None
                robot.state.path = []
                robot.stuck_detector.reset(robot.state, robot.hardware)
                robot.state.goal_attempts += 1

                if robot.state.goal_attempts > robot.state.max_goal_attempts:
                    robot.hardware.set_velocity(0.0, 2.0)
                    robot.state.goal_attempts = 0
                return

            # Follow path requires mapper context
            # This stays in the main class for now
            return 'FOLLOW_PATH'
        else:
            if robot.state.mode == 'RETURNING_HOME':
                plan_return_path_fn(robot)
                if robot.state.goal:
                    return 'FOLLOW_PATH'
                else:
                    robot.hardware.set_velocity(0.0, 0.5)
            else:
                robot.hardware.set_velocity(0.0, 0.5)

        return None
