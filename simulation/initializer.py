"""
Simulation Initialization Module

Handles environment setup, map generation, and robot creation for the coverage mapping simulation.
Refactored to use dictionary-based configuration.
"""

import pybullet as p
from simulation.environment import MapGenerator, PybulletRenderer
from robot import RobotContainer

class SimulationInitializer:
    """Handles all initialization logic for the coverage mapping simulation."""

    def __init__(self, env_config):
        """
        Initialize simulation parameters from configuration dict.

        Args:
            env_config (dict): 'environment' section of config.yaml
        """
        self.config = env_config
        
        self.use_gui = False # This is handled by the renderer, but good to track
        self.maze_size = (self.config['maze_size'], self.config['maze_size'])
        self.cell_size = self.config['cell_size']
        self.env_seed = self.config['seed']
        self.env_type = self.config['type']
        
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

    def create_robots(self, env, robot_config, physics_config, lidar_config=None):
        """
        Create and configure robots at spawn position.

        Args:
            env: PybulletRenderer instance
            robot_config (dict): 'robots' section of config
            physics_config (dict): 'physics' section of config
            lidar_config (dict): Optional LIDAR config with 'num_rays' and 'max_range'

        Returns:
            list: List of RobotContainer instances
        """
        spawn_pos = env.get_spawn_position()
        self.num_robots = robot_config['count']

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
        spacing = robot_config['spacing']
        spawn_height = robot_config['spawn_height']
        
        for i in range(self.num_robots):
            offset = (i - (self.num_robots - 1) / 2) * spacing
            start_positions.append([spawn_pos[0] + offset, spawn_pos[1], spawn_height])

        colors = [all_colors[i % len(all_colors)] for i in range(self.num_robots)]

        robots = []
        radius = robot_config['radius']
        mass = robot_config['mass']
        
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
                           lateralFriction=physics_config['lateral_friction'],
                           spinningFriction=physics_config['spinning_friction'],
                           rollingFriction=physics_config['rolling_friction'])

            robot = RobotContainer(robot_id, pos, color, lidar_config=lidar_config)
            robots.append(robot)

        return robots

    def get_env_config(self):
        """
        Get environment configuration dictionary for logging.

        Returns:
            dict: Environment configuration parameters
        """
        # Return a copy of the config augmented with num_robots for the logger
        conf = self.config.copy()
        conf['num_robots'] = self.num_robots
        
        # FIX: Backward compatibility for video_renderer/playback which expect 'env_type'
        if 'type' in conf:
            conf['env_type'] = conf['type']
            
        return conf