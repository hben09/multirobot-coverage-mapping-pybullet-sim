"""
Simulation Initialization Module

Handles environment setup, map generation, and robot creation for the coverage mapping simulation.
Uses type-safe dataclass configuration.
"""

import pybullet as p
from simulation.level_generator import MapGenerator
from simulation.physics_engine import PybulletRenderer
from robot import RobotContainer
from utils.config_schema import EnvironmentConfig, RobotConfig, PhysicsConfig, LidarConfig

class SimulationInitializer:
    """Handles all initialization logic for the coverage mapping simulation."""

    def __init__(self, env_config: EnvironmentConfig):
        """
        Initialize simulation parameters from typed configuration object.

        Args:
            env_config: Environment configuration with type-safe access.
        """
        self.config = env_config

        self.use_gui = False # This is handled by the renderer, but good to track
        self.maze_size = (self.config.maze_size, self.config.maze_size)
        self.cell_size = self.config.cell_size
        self.env_seed = self.config.seed
        self.env_type = self.config.type

        # Note: num_robots is now passed to create_robots directly or handled by manager
        # We store it here if needed for metadata
        self.num_robots = 0 

    def initialize_environment(self, use_gui=False):
        """
        Generate maze and create PyBullet environment.

        Args:
            use_gui (bool): Whether to show the PyBullet GUI.

        Returns:
            tuple: (env, physics_client, map_bounds)
        """
        self.use_gui = use_gui
        
        # Generate the maze grid
        map_generator = MapGenerator(
            maze_size=self.maze_size,
            seed=self.env_seed
        )
        maze_grid, entrance_cell = map_generator.generate_maze(env_type=self.env_type)

        # Create PyBullet environment from the grid
        env = PybulletRenderer(
            maze_grid=maze_grid,
            entrance_cell=entrance_cell,
            cell_size=self.cell_size,
            wall_height=2.5,
            gui=self.use_gui
        )

        env.build_walls()
        physics_client = env.physics_client

        # Calculate map bounds
        block_physical_size = env.cell_size / 2.0
        half_block = block_physical_size / 2.0

        maze_world_width = env.maze_grid.shape[1] * block_physical_size
        maze_world_height = env.maze_grid.shape[0] * block_physical_size

        map_bounds = {
            'x_min': -half_block,
            'x_max': maze_world_width - half_block,
            'y_min': -half_block,
            'y_max': maze_world_height - half_block
        }

        return env, physics_client, map_bounds

    def create_robots(self, env, robot_config: RobotConfig, physics_config: PhysicsConfig, lidar_config: LidarConfig = None):
        """
        Create and configure robots at spawn position.

        Args:
            env: PybulletRenderer instance
            robot_config: Robot configuration object
            physics_config: Physics configuration object
            lidar_config: Optional LIDAR configuration object

        Returns:
            list: List of RobotContainer instances
        """
        spawn_pos = env.get_spawn_position()
        self.num_robots = robot_config.count

        all_colors = [
            [1, 0, 0, 1],        # Red
            [0, 1, 0, 1],        # Green
            [0, 0, 1, 1],        # Blue
            [1, 1, 0, 1],        # Yellow
            [1, 0, 1, 1],        # Magenta
            [0, 1, 1, 1],        # Cyan
            [1, 0.5, 0, 1],      # Orange
            [0.5, 0, 1, 1],      # Purple
            [0.5, 0.5, 0.5, 1],  # Gray
            [1, 0.75, 0.8, 1],   # Pink
            [0, 0.5, 0, 1],      # Dark Green
            [0.5, 0.25, 0, 1],   # Brown
            [0, 0, 0.5, 1],      # Navy
            [0.5, 1, 0, 1],      # Lime
            [1, 0.5, 0.5, 1],    # Salmon
            [0, 0.5, 0.5, 1],    # Teal
        ]

        # Calculate start positions with spacing
        start_positions = []
        spacing = robot_config.spacing
        spawn_height = robot_config.spawn_height

        for i in range(self.num_robots):
            offset = (i - (self.num_robots - 1) / 2) * spacing
            start_positions.append([spawn_pos[0] + offset, spawn_pos[1], spawn_height])

        colors = [all_colors[i % len(all_colors)] for i in range(self.num_robots)]

        robots = []
        radius = robot_config.radius
        mass = robot_config.mass
        
        for i, (pos, color) in enumerate(zip(start_positions, colors)):
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)

            robot_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=pos
            )

            # Configure physics properties
            p.changeDynamics(robot_id, -1, localInertiaDiagonal=[0, 0, 1])
            p.changeDynamics(robot_id, -1,
                           lateralFriction=physics_config.lateral_friction,
                           spinningFriction=physics_config.spinning_friction,
                           rollingFriction=physics_config.rolling_friction)

            robot = RobotContainer(robot_id, pos, color, lidar_config=lidar_config)
            robots.append(robot)

        return robots

    def get_env_config(self):
        """
        Get environment configuration dictionary for logging.

        Returns:
            dict: Environment configuration parameters (for backward compatibility with logger)
        """
        # Convert dataclass to dict and augment with num_robots for the logger
        conf = {
            'maze_size': self.config.maze_size,
            'cell_size': self.config.cell_size,
            'type': self.config.type,
            'seed': self.config.seed,
            'safety_margin': self.config.safety_margin,
            'num_robots': self.num_robots,
            'env_type': self.config.type  # Backward compatibility for video_renderer/playback
        }

        return conf