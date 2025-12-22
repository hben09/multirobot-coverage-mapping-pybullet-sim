"""
RobotState - Pure data container for robot state.

This module contains only state data, no behaviors or PyBullet dependencies.
Behaviors are implemented in separate modules.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class RobotState:
    """
    Pure state container for a robot (no behaviors).

    All state data the robot needs, separated from hardware interface
    and behavior logic.
    """

    # === 1. Basic Identification ===
    id: int
    position: Tuple[float, float, float]
    color: Tuple[float, float, float]

    # === 2. Sensor and Trajectory Data ===
    lidar_data: List[Tuple[float, float, bool]] = field(default_factory=list)
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

    # === 3. Autonomy State ===
    goal: Optional[Tuple[float, float]] = None
    path: List[Tuple[int, int]] = field(default_factory=list)
    mode: str = 'IDLE'  # IDLE, GLOBAL_RELOCATE, RETURNING_HOME, HOME

    # === 4. Goal Management ===
    blacklisted_goals: Dict[Tuple[int, int], int] = field(default_factory=dict)
    goal_attempts: int = 0
    max_goal_attempts: int = 3

    # === 5. Exploration Direction Tracking ===
    exploration_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    direction_history: List = field(default_factory=list)
    direction_smoothing_window: int = 10

    # === 6. Stuck Detection State ===
    stuck_counter: int = 0
    last_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    # === 7. Home Position ===
    home_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    # === Constants ===
    MAX_TRAJECTORY_LENGTH: int = field(default=200, init=False)
    TRAJECTORY_TRIM_SIZE: int = field(default=150, init=False)

    def __post_init__(self):
        """Initialize computed fields after dataclass initialization."""
        # Initialize last_position from position
        self.last_position = np.array([self.position[0], self.position[1]])

        # Initialize home_position from position
        self.home_position = np.array([self.position[0], self.position[1]])

    def cleanup_blacklist(self, current_step: int):
        """Remove expired entries from the blacklist."""
        expired = [
            pos for pos, expiry in self.blacklisted_goals.items()
            if current_step >= expiry
        ]
        for pos in expired:
            del self.blacklisted_goals[pos]

    def update_position(self, position: Tuple[float, float]):
        """Update the robot's position (called from hardware interface)."""
        self.position = position

    def trim_trajectory_if_needed(self):
        """Trim trajectory if it exceeds maximum length."""
        if len(self.trajectory) > self.MAX_TRAJECTORY_LENGTH:
            self.trajectory = self.trajectory[-self.TRAJECTORY_TRIM_SIZE:]
