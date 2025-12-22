"""
Configuration module for multi-robot coverage simulation.
Edit the variables below to change simulation settings.
"""

# ============================================================
# SIMULATION CONFIGURATION
# Edit these variables to change your simulation settings
# ============================================================

# Maze dimensions (e.g., 10 for 10x10)
MAZE_SIZE = 10

# Cell size in meters
CELL_SIZE = 10.0

# Random seed (None for random, or any integer for reproducible results)
ENV_SEED = None

# Environment type: 'maze', 'blank_box', 'cave', 'tunnel', 'rooms', 'sewer', 'corridor_rooms'
ENV_TYPE = 'corridor_rooms'

# Show PyBullet 3D window (True/False)
USE_GUI = False

# Show rectangular decomposition visualization (True/False)
# NOTE: Only visible in 'realtime' or 'both' visualization modes (not in videos yet)
SHOW_PARTITIONS = False

# Number of robots (1-16)
NUM_ROBOTS = 3

# Maximum simulation steps (None for unlimited)
MAX_STEPS = None

# Visualization mode: 'realtime', 'logging', 'both', 'none'
VIZ_MODE = 'realtime'

# Automatically render video after simulation (True/False)
RENDER_VIDEO = False

# ============================================================
# ALGORITHM PARAMETERS
# Advanced settings for exploration and coordination algorithms
# ============================================================

# Grid resolution in meters (smaller = more detailed but slower)
GRID_RESOLUTION = 0.5

# Utility function weights for frontier selection
DIRECTION_BIAS_WEIGHT = 2.5  # How much to reward forward motion (0-5)
SIZE_WEIGHT = 0.3            # Weight for frontier size (0-1)
DISTANCE_WEIGHT = 1.0        # Weight for distance cost (0-2)

# Multi-robot coordination parameters
CROWDING_PENALTY_WEIGHT = 100.0  # Penalty for targeting same area as other robots
CROWDING_RADIUS = 8.0            # Radius (meters) to discourage other robots

# Return-to-home settings
RETURN_HOME_COVERAGE = 100.0  # Trigger return home at this coverage percentage (0-100)

# Pathfinding safety margin
SAFETY_MARGIN = 1  # Grid cells to inflate obstacles for safe navigation

# Robot spawning spacing
ROBOT_SPACING = 1.5  # Distance (meters) between robots at spawn

# Robot physical parameters
ROBOT_RADIUS = 0.25       # Robot sphere radius (meters)
ROBOT_MASS = 1.0          # Robot mass (kg)
ROBOT_HEIGHT = 0.25       # Robot spawn height (meters)

# Robot dynamics
LATERAL_FRICTION = 1.0
SPINNING_FRICTION = 0.1
ROLLING_FRICTION = 0.0

# LIDAR sensor parameters
LIDAR_NUM_RAYS = 90       # Number of LIDAR rays
LIDAR_MAX_RANGE = 15.0    # Maximum LIDAR range (meters)

# Simulation timing
SCAN_INTERVAL = 10              # Steps between LIDAR scans
VIZ_UPDATE_INTERVAL = 50        # Steps between visualization updates
PERFORMANCE_REPORT_INTERVAL = 3.0  # Seconds between performance reports

# ============================================================


def get_simulation_config():
    """
    Get simulation configuration from file settings.

    Returns:
        dict: Configuration parameters for the simulation
    """
    return {
        'maze_size': MAZE_SIZE,
        'cell_size': CELL_SIZE,
        'env_seed': ENV_SEED,
        'env_type': ENV_TYPE,
        'use_gui': USE_GUI,
        'show_partitions': SHOW_PARTITIONS,
        'num_robots': NUM_ROBOTS,
        'max_steps': MAX_STEPS,
        'viz_mode': VIZ_MODE,
        'render_video': RENDER_VIDEO,
        'grid_resolution': GRID_RESOLUTION,
        'scan_interval': SCAN_INTERVAL,
        'viz_update_interval': VIZ_UPDATE_INTERVAL,
        'performance_report_interval': PERFORMANCE_REPORT_INTERVAL
    }


def get_default_config():
    """
    Returns default simulation configuration without prompts.

    Returns:
        dict: Default configuration parameters
    """
    return {
        'maze_size': 10,
        'cell_size': 10.0,
        'env_seed': None,
        'env_type': 'maze',
        'use_gui': False,
        'show_partitions': False,
        'num_robots': 3,
        'max_steps': None,
        'viz_mode': 'realtime',
        'render_video': False,
        'grid_resolution': GRID_RESOLUTION,
        'scan_interval': SCAN_INTERVAL,
        'viz_update_interval': VIZ_UPDATE_INTERVAL,
        'performance_report_interval': PERFORMANCE_REPORT_INTERVAL
    }


def print_config(config):
    """
    Pretty print the simulation configuration.

    Args:
        config (dict): Configuration dictionary to print
    """
    print(f"\nCreating {config['maze_size']}x{config['maze_size']} {config['env_type']} "
          f"with {config['cell_size']}m cells and {config['num_robots']} robots...")
    print(f"  - Visualization mode: {config['viz_mode']}")
    print(f"  - PyBullet GUI: {'Enabled' if config['use_gui'] else 'Disabled'}")
    print(f"  - Partition viz: {'Enabled' if config['show_partitions'] else 'Disabled'}")
    if config['max_steps'] is not None:
        print(f"  - Max steps: {config['max_steps']}")
    else:
        print(f"  - Max steps: Unlimited")
    if config['env_seed'] is not None:
        print(f"  - Random seed: {config['env_seed']}")
