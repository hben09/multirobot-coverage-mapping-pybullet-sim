"""Robot module containing robot state and hardware interfaces."""

from .robot_state import RobotState
from .robot_hardware import RobotHardware


class RobotContainer:
    """
    Lightweight container that holds robot components together.

    This is a simple namespace that groups related components:
    - state: RobotState (pure data)
    - hardware: RobotHardware (PyBullet interface)
    - stuck_detector: StuckDetector (behavior)
    - path_follower: PathFollower (behavior)
    - direction_tracker: ExplorationDirectionTracker (behavior)

    Unlike the old Robot class, this doesn't provide convenience methods.
    Users should access components directly (e.g., robot.state.position).
    """

    def __init__(self, robot_id, position, color):
        """
        Initialize robot container with all components.

        Args:
            robot_id: PyBullet body ID
            position: Initial position (x, y, z)
            color: RGB color tuple
        """
        # Lazy import to avoid circular dependencies
        from behaviors.stuck_detector import StuckDetector
        from behaviors.path_follower import PathFollower
        from behaviors.exploration_direction_tracker import ExplorationDirectionTracker

        # Core components
        self.state = RobotState(id=robot_id, position=position, color=color)
        self.hardware = RobotHardware(pybullet_id=robot_id)

        # Behavior components
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


__all__ = ['RobotState', 'RobotHardware', 'RobotContainer']
