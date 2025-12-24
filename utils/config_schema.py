"""
Type-safe configuration schema using dataclasses.

This module provides strongly-typed configuration classes that replace
dictionary-based configuration access, preventing spelling errors and
enabling better IDE support.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvironmentConfig:
    """Environment generation configuration."""
    maze_size: int
    cell_size: float
    type: str  # options: maze, blank_box, cave, tunnel, rooms, sewer, corridor_rooms
    seed: Optional[int] = None  # null for random, or integer
    safety_margin: int = 1  # Grid cells to inflate obstacles


@dataclass
class RobotConfig:
    """Robot creation and physics configuration."""
    count: int
    spacing: float
    radius: float
    mass: float
    spawn_height: float


@dataclass
class LidarConfig:
    """LIDAR sensor configuration."""
    num_rays: int
    max_range: float


@dataclass
class SensorConfig:
    """Sensor configuration container."""
    lidar: LidarConfig


@dataclass
class PhysicsConfig:
    """Physics simulation parameters."""
    lateral_friction: float
    spinning_friction: float
    rolling_friction: float


@dataclass
class UtilityWeightsConfig:
    """Utility calculation weights for planning."""
    direction_bias: float
    size: float
    distance: float


@dataclass
class CoordinationConfig:
    """Multi-robot coordination parameters."""
    crowding_penalty: float
    crowding_radius: float


@dataclass
class PlanningConfig:
    """Planning algorithm configuration."""
    grid_resolution: float
    utility_weights: UtilityWeightsConfig
    coordination: CoordinationConfig


@dataclass
class IntervalConfig:
    """Timing intervals for various operations."""
    scan: int
    viz_update: int
    performance_report: float


@dataclass
class SystemConfig:
    """System-level configuration."""
    use_gui: bool
    viz_mode: str  # options: realtime, logging, both, none
    render_video: bool
    show_partitions: bool
    max_steps: Optional[int] = None  # null for unlimited
    intervals: IntervalConfig = field(default_factory=lambda: IntervalConfig(
        scan=10,
        viz_update=50,
        performance_report=3.0
    ))


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    environment: EnvironmentConfig
    robots: RobotConfig
    sensors: SensorConfig
    physics: PhysicsConfig
    planning: PlanningConfig
    system: SystemConfig
