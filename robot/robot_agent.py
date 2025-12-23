"""
RobotAgent - Autonomous robot control node.

This class represents the "onboard computer" that runs on each robot.
It implements the sense-think-act cycle independently, similar to a ROS 2 node.

In professional robotics:
- ROS 1/2: Each robot runs a node with its own control loop
- Real robots: Onboard computer runs autonomy stack
- This pattern: Each agent manages its own behavior

The simulation loop simply "ticks" agents; it doesn't micromanage them.
"""

from typing import Tuple, Optional, Callable, TYPE_CHECKING, Dict, Any
import numpy as np
from robot.robot_state import RobotState
from robot.robot_driver import RobotDriver
from behaviors.stuck_detector import StuckDetector
from behaviors.path_follower import PathFollower
from behaviors.exploration_direction_tracker import ExplorationDirectionTracker

if TYPE_CHECKING:
    from mapping.grid_manager import OccupancyGridManager
    from navigation.pathfinding import NumbaAStarHelper
    from coordination.utility_calculator import FrontierUtilityCalculator


class RobotAgent:
    """
    Autonomous robot agent implementing sense-think-act cycle.

    This is the "brain" of the robot that runs independently.
    It handles:
    - Sensing (LIDAR scans)
    - Thinking (stuck detection, path planning, path following)
    - Acting (velocity commands)

    The agent operates autonomously once initialized and only needs
    external coordination for goal assignment (not path planning).
    """

    def __init__(
        self,
        state: RobotState,
        driver: RobotDriver,
        stuck_detector: StuckDetector,
        path_follower: PathFollower,
        direction_tracker: ExplorationDirectionTracker,
        lidar_num_rays: int = 360,
        lidar_max_range: float = 10.0,
        grid_manager: Optional['OccupancyGridManager'] = None,
        planner: Optional['NumbaAStarHelper'] = None,
        safety_margin: float = 0.3,
        utility_calculator: Optional['FrontierUtilityCalculator'] = None
    ):
        """
        Initialize the robot agent.

        Args:
            state: Robot state container
            driver: Hardware driver interface
            stuck_detector: Stuck detection behavior
            path_follower: Path following behavior
            direction_tracker: Exploration direction tracker
            lidar_num_rays: Number of LIDAR rays
            lidar_max_range: Maximum LIDAR range (meters)
            grid_manager: Occupancy grid manager (simulates "map server" subscription)
            planner: Path planner instance
            safety_margin: Safety margin for obstacle inflation
            utility_calculator: Utility calculator for bid calculation
        """
        self.state = state
        self.driver = driver
        self.stuck_detector = stuck_detector
        self.path_follower = path_follower
        self.direction_tracker = direction_tracker

        # LIDAR configuration
        self.lidar_num_rays = lidar_num_rays
        self.lidar_max_range = lidar_max_range

        # Planning dependencies (simulating map server access)
        self.grid_manager = grid_manager
        self.planner = planner
        self.safety_margin = safety_margin

        # Utility calculator for autonomous bidding
        self.utility_calculator = utility_calculator

    def sense(self):
        """
        SENSE phase: Gather sensor data from the environment.

        This method:
        1. Performs LIDAR scan
        2. Updates robot position
        3. Records trajectory
        4. Updates exploration direction
        """
        # 1. Perform LIDAR scan
        scan_points = self.driver.get_lidar_scan(
            num_rays=self.lidar_num_rays,
            max_range=self.lidar_max_range
        )
        self.state.lidar_data = scan_points

        # 2. Update position and trajectory
        pos_array, _ = self.driver.get_pose()
        self.state.trajectory.append((pos_array[0], pos_array[1]))
        self.state.trim_trajectory_if_needed()

        # 3. Update exploration direction from trajectory
        self.direction_tracker.update(self.state)

    def think(self) -> Optional[str]:
        """
        THINK phase: Process sensor data and decide on actions.

        This method:
        1. Checks if robot is stuck
        2. Handles stuck recovery
        3. Plans paths autonomously when needed
        4. Determines navigation action

        Returns:
            Action command: 'FOLLOW_PATH', None, etc.
        """
        # 1. Check if we're home and should stop
        if self.state.mode == 'HOME':
            return 'STOP'

        # 2. If we have a goal, check for stuck condition
        if self.state.goal:
            if self.stuck_detector.is_stuck(self.state, self.driver):
                print(f"Robot {self.state.id} is stuck! Abandoning goal and replanning...")
                self.state.goal = None
                self.state.path = []
                self.stuck_detector.reset(self.state, self.driver)
                self.state.goal_attempts += 1

                # If stuck too many times, just spin
                if self.state.goal_attempts > self.state.max_goal_attempts:
                    self.state.goal_attempts = 0
                    return 'SPIN'

                # If returning home, try to replan path home
                if self.state.mode == 'RETURNING_HOME':
                    if self.plan_path_home():
                        return 'FOLLOW_PATH'

                return None

            # We have a goal and we're not stuck - follow the path
            return 'FOLLOW_PATH'

        # 3. No goal - handle based on mode
        else:
            if self.state.mode == 'RETURNING_HOME':
                # Autonomously plan path home
                if self.plan_path_home():
                    return 'FOLLOW_PATH'

                # Couldn't plan path home - just spin slowly
                return 'SPIN_SLOW'
            else:
                # Exploring but no goal - spin slowly to wait for assignment
                return 'SPIN_SLOW'

    def act(self, action: Optional[str], grid_to_world_fn: Optional[Callable] = None):
        """
        ACT phase: Execute motor commands based on decided action.

        Args:
            action: Action command from think() phase
            grid_to_world_fn: Function to convert grid to world coords
                             Required for 'FOLLOW_PATH' action
        """
        if action == 'STOP':
            self.driver.set_velocity(0.0, 0.0)

        elif action == 'FOLLOW_PATH':
            if grid_to_world_fn is None:
                raise ValueError("grid_to_world_fn required for FOLLOW_PATH action")

            linear_vel, angular_vel = self.path_follower.compute_velocities(
                self.state,
                self.driver,
                grid_to_world_fn
            )
            self.driver.set_velocity(linear_vel, angular_vel)

        elif action == 'SPIN':
            self.driver.set_velocity(0.0, 2.0)

        elif action == 'SPIN_SLOW':
            self.driver.set_velocity(0.0, 0.5)

        else:
            # No action or unknown action - stop
            self.driver.set_velocity(0.0, 0.0)

    def update(
        self,
        should_sense: bool = True,
        grid_to_world_fn: Optional[Callable] = None
    ):
        """
        Complete sense-think-act cycle.

        This is the main "tick" function that runs the robot's autonomy loop.

        Args:
            should_sense: Whether to perform sensing this tick
            grid_to_world_fn: Function to convert grid to world coords (passed to act())
        """
        # SENSE
        if should_sense:
            self.sense()

        # THINK
        action = self.think()

        # ACT
        self.act(action, grid_to_world_fn)

    def check_if_home(self) -> bool:
        """
        Check if robot has reached home position.

        Returns:
            True if robot is within 1.0m of home position
        """
        pos_array, _ = self.driver.get_pose()
        dx = pos_array[0] - self.state.home_position[0]
        dy = pos_array[1] - self.state.home_position[1]

        if dx*dx + dy*dy < 1.0:
            self.state.mode = 'HOME'
            self.state.goal = None
            self.state.path = []
            return True

        return False

    def set_planning_dependencies(
        self,
        grid_manager: 'OccupancyGridManager',
        planner: 'NumbaAStarHelper',
        safety_margin: float,
        utility_calculator: Optional['FrontierUtilityCalculator'] = None
    ):
        """
        Inject planning dependencies (called after construction).

        In a real ROS system, this simulates subscribing to the map server.

        Args:
            grid_manager: Occupancy grid manager
            planner: Path planner instance
            safety_margin: Safety margin for obstacle inflation
            utility_calculator: Utility calculator for bid calculation
        """
        self.grid_manager = grid_manager
        self.planner = planner
        self.safety_margin = safety_margin
        if utility_calculator is not None:
            self.utility_calculator = utility_calculator

    def calculate_bid(self, frontier: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate bid for a frontier using the robot's internal state.

        This implements the "Auctioneer" pattern where robots calculate their own
        bids based on their internal state (position, battery, exploration direction).
        The central server doesn't need to know these details.

        Args:
            frontier: Frontier dict with 'pos' and 'size'

        Returns:
            Tuple of (bid_value, debug_info_dict)
            - bid_value: The robot's bid for this frontier
            - debug_info: Dictionary with breakdown of bid components

        Raises:
            ValueError: If utility calculator is not available
        """
        if self.utility_calculator is None:
            raise ValueError(f"Robot {self.state.id}: Cannot calculate bid - no utility calculator!")

        # Get current position from driver (robot's internal state)
        pos_array, _ = self.driver.get_pose()
        robot_pos = np.array([pos_array[0], pos_array[1]])

        # Get frontier position
        frontier_pos = np.array(frontier['pos'])

        # Use utility calculator to compute bid
        # This uses robot's internal state like exploration_direction
        bid_value, debug_info = self.utility_calculator.calculate(
            robot_position=robot_pos,
            exploration_direction=self.state.exploration_direction,
            frontier_position=frontier_pos,
            frontier_size=frontier['size']
        )

        return bid_value, debug_info

    def plan_path_to_goal(self, goal_position: Tuple[float, float]) -> bool:
        """
        Plan a path from current position to goal position.

        This is the robot's own path planning capability - it receives a goal
        from the coordinator but plans the path itself using its map access.

        Args:
            goal_position: Goal position in world coordinates (x, y)

        Returns:
            True if path planning succeeded, False otherwise
        """
        if self.grid_manager is None or self.planner is None:
            print(f"Robot {self.state.id}: Cannot plan path - no map access!")
            return False

        # Get current position
        pos_array, _ = self.driver.get_pose()

        # Convert to grid coordinates
        start_grid = self.grid_manager.world_to_grid(pos_array[0], pos_array[1])
        goal_grid = self.grid_manager.world_to_grid(goal_position[0], goal_position[1])

        # Update planner with current grid state (DIRECT NUMPY REFERENCE)
        numpy_grid = self.grid_manager.get_numpy_grid()
        grid_offset_x, grid_offset_y = self.grid_manager.get_grid_offset()

        self.planner.update_grid(
            numpy_grid,
            grid_offset_x,
            grid_offset_y,
            self.safety_margin
        )

        # Plan path
        path = self.planner.plan_path(start_grid, goal_grid, use_inflation=True)

        if path:
            self.state.path = path
            self.state.goal = goal_position
            return True
        else:
            return False

    def plan_path_home(self) -> bool:
        """
        Plan a path back to home position.

        Returns:
            True if path planning succeeded, False otherwise
        """
        return self.plan_path_to_goal(tuple(self.state.home_position))
