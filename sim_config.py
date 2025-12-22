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


def get_simulation_config(use_prompts=False):
    """
    Get simulation configuration either from file settings or user prompts.

    Args:
        use_prompts (bool): If True, prompt user for settings. If False, use variables from this file.

    Returns:
        dict: Configuration parameters for the simulation
    """
    if use_prompts:
        print("=" * 60)
        print("Multi-Robot Coverage Mapping")
        print("=" * 60)

        # Maze size
        maze_size_input = input("\nEnter maze size (e.g., '10' for 10x10, default=10): ").strip()
        maze_size = int(maze_size_input) if maze_size_input.isdigit() else 10

        # Cell size
        cell_size_input = input("Enter cell size in meters (default=10.0): ").strip()
        try:
            cell_size = float(cell_size_input)
        except:
            cell_size = 10.0

        # Random seed
        seed_input = input("Enter random seed (press Enter for random): ").strip()
        env_seed = int(seed_input) if seed_input.isdigit() else None

        # Environment type
        print("\nEnvironment types:")
        print("  1. Maze (complex maze with walls)")
        print("  2. Blank box (empty room with single wall in middle)")
        print("  3. Cave (organic cellular automata)")
        print("  4. Tunnel (long winding corridor)")
        print("  5. Rooms (dungeon with connected chambers)")
        print("  6. Sewer (grid of interconnected pipes)")
        print("  7. Corridor Rooms (Central hall with attached rooms)")
        env_type_input = input("Choose environment type (1-7, default=1): ").strip()

        env_type_map = {
            '2': 'blank_box',
            '3': 'cave',
            '4': 'tunnel',
            '5': 'rooms',
            '6': 'sewer',
            '7': 'corridor_rooms'
        }
        env_type = env_type_map.get(env_type_input, 'maze')

        # GUI setting
        gui_input = input("Show PyBullet 3D window? (y/n, default=n): ").strip().lower()
        use_gui = gui_input == 'y'

        # Partition visualization
        part_input = input("Show Rectangular Decomposition visualization? (y/n, default=n): ").strip().lower()
        show_partitions = part_input == 'y'

        # Number of robots
        num_robots_input = input("Number of robots (1-16, default=3): ").strip()
        if num_robots_input.isdigit():
            num_robots = max(1, min(16, int(num_robots_input)))
        else:
            num_robots = 3

        # Simulation steps
        steps_input = input("Number of simulation steps (press Enter for unlimited): ").strip()
        if steps_input.isdigit():
            max_steps = int(steps_input)
        else:
            max_steps = None

        # Visualization mode
        print("\nVisualization modes:")
        print("  1. realtime - Live matplotlib window (default)")
        print("  2. logging  - Log to file for offline playback (faster)")
        print("  3. both     - Live visualization AND logging")
        print("  4. none     - No visualization (fastest)")
        viz_mode_input = input("Choose visualization mode (1-4, default=1): ").strip()

        viz_mode_map = {
            '2': 'logging',
            '3': 'both',
            '4': 'none'
        }
        viz_mode = viz_mode_map.get(viz_mode_input, 'realtime')

        # Video rendering preference
        render_video_input = input("\nRender video after simulation? (y/n, default=n): ").strip().lower()
        render_video = render_video_input == 'y'

        return {
            'maze_size': maze_size,
            'cell_size': cell_size,
            'env_seed': env_seed,
            'env_type': env_type,
            'use_gui': use_gui,
            'show_partitions': show_partitions,
            'num_robots': num_robots,
            'max_steps': max_steps,
            'viz_mode': viz_mode,
            'render_video': render_video,
            'grid_resolution': GRID_RESOLUTION,
            'scan_interval': SCAN_INTERVAL,
            'viz_update_interval': VIZ_UPDATE_INTERVAL,
            'performance_report_interval': PERFORMANCE_REPORT_INTERVAL
        }
    else:
        # Use configuration variables from this file
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
