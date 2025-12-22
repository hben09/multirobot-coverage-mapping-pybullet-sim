"""
Simulation Initialization Module

Handles environment setup, map generation, and robot creation for the coverage mapping simulation.
"""

import pybullet as p
from environment import MapGenerator, PybulletRenderer
from robot import RobotContainer
import sim_config as cfg


class SimulationInitializer:
    """Handles all initialization logic for the coverage mapping simulation."""

    def __init__(self, use_gui=True, maze_size=(10, 10), cell_size=2.0,
                 env_seed=None, env_type='maze', num_robots=3, show_partitions=False):
        """
        Initialize simulation parameters.

        Args:
            use_gui: Whether to show PyBullet GUI
            maze_size: Tuple of (width, height) for maze dimensions
            cell_size: Physical size of each maze cell
            env_seed: Random seed for environment generation
            env_type: Type of environment ('maze', 'rooms', etc.)
            num_robots: Number of robots to create
            show_partitions: Whether to show rectangular decomposition partitions
        """
        self.use_gui = use_gui
        self.maze_size = maze_size
        self.cell_size = cell_size
        self.env_seed = env_seed
        self.env_type = env_type
        self.num_robots = num_robots
        self.show_partitions = show_partitions

    def initialize_environment(self):
        """
        Generate maze and create PyBullet environment.

        Returns:
            tuple: (env, physics_client, map_bounds) where:
                - env: PybulletRenderer instance
                - physics_client: PyBullet physics client ID
                - map_bounds: Dictionary with x_min, x_max, y_min, y_max
        """
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

    def create_robots(self, env):
        """
        Create and configure robots at spawn position.

        Args:
            env: PybulletRenderer instance

        Returns:
            list: List of Robot instances
        """
        spawn_pos = env.get_spawn_position()

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
        spacing = cfg.ROBOT_SPACING
        for i in range(self.num_robots):
            offset = (i - (self.num_robots - 1) / 2) * spacing
            start_positions.append([spawn_pos[0] + offset, spawn_pos[1], cfg.ROBOT_HEIGHT])

        colors = [all_colors[i % len(all_colors)] for i in range(self.num_robots)]

        robots = []
        for i, (pos, color) in enumerate(zip(start_positions, colors)):
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=cfg.ROBOT_RADIUS)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=cfg.ROBOT_RADIUS, rgbaColor=color)

            robot_id = p.createMultiBody(
                baseMass=cfg.ROBOT_MASS,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=pos
            )

            # Configure physics properties
            p.changeDynamics(robot_id, -1, localInertiaDiagonal=[0, 0, 1])
            p.changeDynamics(robot_id, -1,
                           lateralFriction=cfg.LATERAL_FRICTION,
                           spinningFriction=cfg.SPINNING_FRICTION,
                           rollingFriction=cfg.ROLLING_FRICTION)

            robot = RobotContainer(robot_id, pos, color)
            robots.append(robot)

        return robots

    def get_env_config(self):
        """
        Get environment configuration dictionary for logging.

        Returns:
            dict: Environment configuration parameters
        """
        return {
            'maze_size': self.maze_size,
            'cell_size': self.cell_size,
            'env_seed': self.env_seed,
            'env_type': self.env_type,
            'num_robots': self.num_robots,
        }
