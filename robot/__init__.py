"""Robot module containing robot state and hardware interfaces."""

from .robot_state import RobotState
from .robot_driver import RobotDriver
from .pybullet_driver import PyBulletDriver
from .robot_agent import RobotAgent


class RobotContainer:
    """
    Lightweight container that holds robot components together.

    This is a simple namespace that groups related components:
    - state: RobotState (pure data)
    - driver: RobotDriver (hardware abstraction layer)
    - agent: RobotAgent (autonomous control node)
    - Individual behavior components (for direct access if needed)

    The agent is the primary interface for robot control, implementing
    the sense-think-act cycle. The simulation loop should primarily
    interact with the agent, not the individual components.

    The driver field is a RobotDriver interface, which can be:
    - PyBulletDriver for simulation
    - A real robot driver for physical hardware
    - A mock driver for testing

    This separation enables hardware independence and clean architecture.
    """

    def __init__(self, robot_id, position, color, lidar_config=None):
        """
        Initialize robot container with all components.

        Args:
            robot_id: PyBullet body ID
            position: Initial position (x, y, z)
            color: RGB color tuple
            lidar_config: Optional dict with 'num_rays' and 'max_range'
        """
        # Lazy import to avoid circular dependencies
        from behaviors.stuck_detector import StuckDetector
        from behaviors.path_follower import PathFollower
        from behaviors.exploration_direction_tracker import ExplorationDirectionTracker

        # Default LIDAR config
        if lidar_config is None:
            lidar_config = {'num_rays': 360, 'max_range': 10.0}

        # Core components
        self.state = RobotState(id=robot_id, position=position, color=color)
        self.driver = PyBulletDriver(pybullet_id=robot_id)

        # Behavior components (exposed for backwards compatibility)
        self.stuck_detector = StuckDetector(threshold=0.3, stuck_limit=200)
        self.path_follower = PathFollower(
            waypoint_threshold=0.6,
            goal_threshold=0.5,
            max_linear_vel=8.0,
            max_angular_vel=4.0,
            turn_slowdown_angle=0.5
        )
        self.direction_tracker = ExplorationDirectionTracker(
            window_size=20,
            min_distance=0.5,
            smoothing_alpha=0.3
        )

        # Agent: The autonomous control node
        self.agent = RobotAgent(
            state=self.state,
            driver=self.driver,
            stuck_detector=self.stuck_detector,
            path_follower=self.path_follower,
            direction_tracker=self.direction_tracker,
            lidar_num_rays=lidar_config['num_rays'],
            lidar_max_range=lidar_config['max_range']
        )


__all__ = ['RobotState', 'RobotDriver', 'PyBulletDriver', 'RobotAgent', 'RobotContainer']
