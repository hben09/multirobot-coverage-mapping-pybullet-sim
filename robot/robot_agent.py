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

from typing import Tuple, Optional, Callable
from robot.robot_state import RobotState
from robot.robot_driver import RobotDriver
from behaviors.stuck_detector import StuckDetector
from behaviors.path_follower import PathFollower
from behaviors.exploration_direction_tracker import ExplorationDirectionTracker


class RobotAgent:
    """
    Autonomous robot agent implementing sense-think-act cycle.

    This is the "brain" of the robot that runs independently.
    It handles:
    - Sensing (LIDAR scans)
    - Thinking (stuck detection, path following)
    - Acting (velocity commands)

    The agent operates autonomously once initialized and only needs
    external coordination for goal assignment.
    """

    def __init__(
        self,
        state: RobotState,
        driver: RobotDriver,
        stuck_detector: StuckDetector,
        path_follower: PathFollower,
        direction_tracker: ExplorationDirectionTracker,
        lidar_num_rays: int = 360,
        lidar_max_range: float = 10.0
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
        """
        self.state = state
        self.driver = driver
        self.stuck_detector = stuck_detector
        self.path_follower = path_follower
        self.direction_tracker = direction_tracker

        # LIDAR configuration
        self.lidar_num_rays = lidar_num_rays
        self.lidar_max_range = lidar_max_range

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

    def think(self, plan_return_path_fn: Optional[Callable] = None) -> Optional[str]:
        """
        THINK phase: Process sensor data and decide on actions.

        This method:
        1. Checks if robot is stuck
        2. Handles stuck recovery
        3. Determines navigation action

        Args:
            plan_return_path_fn: Optional function to plan return path
                                 Should accept (robot_agent) and plan path home

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

                return None

            # We have a goal and we're not stuck - follow the path
            return 'FOLLOW_PATH'

        # 3. No goal - handle based on mode
        else:
            if self.state.mode == 'RETURNING_HOME':
                # Try to plan a path home if function provided
                if plan_return_path_fn:
                    plan_return_path_fn(self)
                    if self.state.goal:
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
        plan_return_path_fn: Optional[Callable] = None,
        grid_to_world_fn: Optional[Callable] = None
    ):
        """
        Complete sense-think-act cycle.

        This is the main "tick" function that runs the robot's autonomy loop.

        Args:
            should_sense: Whether to perform sensing this tick
            plan_return_path_fn: Function to plan return path (passed to think())
            grid_to_world_fn: Function to convert grid to world coords (passed to act())
        """
        # SENSE
        if should_sense:
            self.sense()

        # THINK
        action = self.think(plan_return_path_fn)

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
