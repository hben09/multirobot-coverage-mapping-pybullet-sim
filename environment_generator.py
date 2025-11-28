"""
PyBullet Random Maze Environment Generator for Multi-Robot Exploration

This script generates a randomized subterranean maze environment where robots
spawn on the outside and must explore the interior.
"""

import pybullet as p
import pybullet_data
import numpy as np
import random
import time
from collections import deque


class MazeEnvironment:
    """Generates and manages a random maze environment in PyBullet."""
    
    def __init__(
        self,
        maze_size=(15, 15),
        cell_size=2.0,
        wall_height=2.0,
        wall_thickness=0.2,
        gui=True,
        seed=None
    ):
        """
        Initialize the maze environment.
        
        Args:
            maze_size: Tuple (width, height) of maze in cells
            cell_size: Size of each cell in meters
            wall_height: Height of walls in meters
            wall_thickness: Thickness of walls in meters
            gui: Whether to use GUI mode
            seed: Random seed for reproducibility
        """
        self.maze_width, self.maze_height = maze_size
        self.cell_size = cell_size
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        self.gui = gui
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.maze_grid = None
        self.wall_ids = []
        self.robot_ids = []
        self.entrance_position = None
        
        self._init_pybullet()
        
    def _init_pybullet(self):
        """Initialize PyBullet simulation."""
        if self.gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        
        # Load ground plane
        self.ground_id = p.loadURDF("plane.urdf")
        
        # Set camera for better view
        if self.gui:
            maze_center_x = (self.maze_width * self.cell_size) / 2
            maze_center_y = (self.maze_height * self.cell_size) / 2
            p.resetDebugVisualizerCamera(
                cameraDistance=max(self.maze_width, self.maze_height) * self.cell_size * 0.8,
                cameraYaw=45,
                cameraPitch=-60,
                cameraTargetPosition=[maze_center_x, maze_center_y, 0]
            )
    
    def generate_maze(self, env_type='maze'):
        """
        Generate an environment based on the specified type.

        Args:
            env_type: 'maze' for maze generation, 'blank_box' for empty room with single wall

        The maze is represented as a 2D grid where:
        - 0 = passage (empty space)
        - 1 = wall
        """
        # Initialize grid with all walls
        # Use odd dimensions for proper maze generation
        grid_width = self.maze_width * 2 + 1
        grid_height = self.maze_height * 2 + 1
        self.maze_grid = np.ones((grid_height, grid_width), dtype=int)

        if env_type == 'blank_box':
            # Create empty box with perimeter walls and single wall in middle
            self._generate_blank_box()
        else:
            # Recursive backtracking maze generation
            self._carve_passages(1, 1)

            # Create entrance on the outside (bottom wall)
            entrance_x = random.choice(range(1, grid_width - 1, 2))
            self.maze_grid[0, entrance_x] = 0
            self.entrance_cell = (entrance_x, 0)

            # Optionally create an exit on the opposite side
            exit_x = random.choice(range(1, grid_width - 1, 2))
            self.maze_grid[grid_height - 1, exit_x] = 0

            # Ensure maze is fully connected and has interesting paths
            self._add_loops(0.1)  # Add some loops for more interesting exploration

        return self.maze_grid

    def _generate_blank_box(self):
        """Generate a blank box environment with perimeter walls and single wall in middle."""
        grid_height, grid_width = self.maze_grid.shape

        # Fill interior with empty space
        self.maze_grid[1:-1, 1:-1] = 0

        # Add a single wall in the middle (vertical)
        mid_x = grid_width // 2
        wall_start_y = grid_height // 4
        wall_end_y = 3 * grid_height // 4

        for y in range(wall_start_y, wall_end_y):
            self.maze_grid[y, mid_x] = 1

        # Create entrance at bottom center
        entrance_x = grid_width // 2
        self.maze_grid[0, entrance_x] = 0
        self.entrance_cell = (entrance_x, 0)
    
    def _carve_passages(self, cx, cy):
        """Carve passages in the maze using recursive backtracking."""
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            
            # Check bounds (using actual grid dimensions)
            if (0 < nx < self.maze_grid.shape[1] - 1 and 
                0 < ny < self.maze_grid.shape[0] - 1):
                
                if self.maze_grid[ny, nx] == 1:
                    # Carve passage
                    self.maze_grid[cy + dy // 2, cx + dx // 2] = 0
                    self.maze_grid[ny, nx] = 0
                    self._carve_passages(nx, ny)
    
    def _add_loops(self, probability):
        """Add some loops to the maze for more interesting exploration."""
        for y in range(1, self.maze_grid.shape[0] - 1):
            for x in range(1, self.maze_grid.shape[1] - 1):
                if self.maze_grid[y, x] == 1:
                    # Count adjacent passages
                    neighbors = 0
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        if self.maze_grid[y + dy, x + dx] == 0:
                            neighbors += 1
                    
                    # Only remove walls that connect two passages
                    if neighbors >= 2 and random.random() < probability:
                        self.maze_grid[y, x] = 0
    
    def build_walls(self):
        """Build physical walls in PyBullet based on the maze grid."""
        if self.maze_grid is None:
            self.generate_maze()
        
        # Clear existing walls
        for wall_id in self.wall_ids:
            p.removeBody(wall_id)
        self.wall_ids = []
        
        # Wall visual and collision properties
        wall_color = [0.4, 0.35, 0.3, 1.0]  # Stone-like color
        
        # Create walls for each cell marked as wall
        for y in range(self.maze_grid.shape[0]):
            for x in range(self.maze_grid.shape[1]):
                if self.maze_grid[y, x] == 1:
                    wall_id = self._create_wall_block(x, y, wall_color)
                    self.wall_ids.append(wall_id)
        
        # Add ceiling to create subterranean feel (optional)
        self._create_ceiling()

        return self.wall_ids
    
    def _create_wall_block(self, grid_x, grid_y, color):
        """Create a single wall block at the specified grid position."""
        # Convert grid coordinates to world coordinates
        world_x = grid_x * self.cell_size / 2
        world_y = grid_y * self.cell_size / 2
        world_z = self.wall_height / 2
        
        half_extent = [
            self.cell_size / 4,
            self.cell_size / 4,
            self.wall_height / 2
        ]
        
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extent
        )
        
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extent,
            rgbaColor=color
        )
        
        wall_id = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[world_x, world_y, world_z]
        )
        
        return wall_id
    
    def _create_ceiling(self):
        """Create a semi-transparent ceiling for the subterranean environment."""
        grid_width = self.maze_grid.shape[1]
        grid_height = self.maze_grid.shape[0]

        ceiling_width = grid_width * self.cell_size / 2
        ceiling_height = grid_height * self.cell_size / 2

        # Semi-transparent dark ceiling
        ceiling_color = [0.2, 0.2, 0.25, 0.3]

        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[ceiling_width / 2, ceiling_height / 2, 0.05],
            rgbaColor=ceiling_color
        )

        ceiling_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=[ceiling_width / 2, ceiling_height / 2, self.wall_height + 0.05]
        )

        self.wall_ids.append(ceiling_id)
    
    def get_spawn_position(self):
        """
        Get the spawn position for robots (outside the maze at entrance).
        
        Returns:
            Tuple (x, y, z) world coordinates for spawn position
        """
        if self.entrance_cell is None:
            raise ValueError("Maze not generated. Call generate_maze() first.")
        
        entrance_x, entrance_y = self.entrance_cell
        
        # Position just outside the entrance
        world_x = entrance_x * self.cell_size / 2
        world_y = -self.cell_size  # Outside the maze
        world_z = 0.1  # Slightly above ground
        
        self.entrance_position = (world_x, world_y, world_z)
        return self.entrance_position
    
    def spawn_robot(self, robot_urdf="r2d2.urdf", position=None, orientation=None):
        """
        Spawn a robot at the specified position or at the entrance.
        
        Args:
            robot_urdf: Path to robot URDF file
            position: Optional spawn position (x, y, z)
            orientation: Optional orientation as quaternion [x, y, z, w]
        
        Returns:
            Robot body ID
        """
        if position is None:
            position = self.get_spawn_position()
        
        if orientation is None:
            # Face towards the maze entrance
            orientation = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        
        try:
            robot_id = p.loadURDF(robot_urdf, position, orientation)
        except Exception as e:
            print(f"Could not load {robot_urdf}, creating simple robot: {e}")
            robot_id = self._create_simple_robot(position, orientation)
        
        self.robot_ids.append(robot_id)
        return robot_id
    
    def _create_simple_robot(self, position, orientation):
        """Create a simple box robot if URDF loading fails."""
        robot_size = [0.3, 0.2, 0.15]
        robot_color = [0.2, 0.6, 0.8, 1.0]
        
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=robot_size
        )
        
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=robot_size,
            rgbaColor=robot_color
        )
        
        robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        return robot_id
    
    def spawn_multi_robots(self, num_robots, robot_urdf="r2d2.urdf", spacing=1.0):
        """
        Spawn multiple robots in a line outside the maze entrance.
        
        Args:
            num_robots: Number of robots to spawn
            robot_urdf: Path to robot URDF file
            spacing: Spacing between robots in meters
        
        Returns:
            List of robot body IDs
        """
        base_position = self.get_spawn_position()
        robot_ids = []
        
        for i in range(num_robots):
            # Offset each robot along the x-axis
            offset_x = (i - num_robots / 2) * spacing
            position = (
                base_position[0] + offset_x,
                base_position[1] - i * spacing * 0.5,  # Stagger slightly
                base_position[2]
            )
            
            robot_id = self.spawn_robot(robot_urdf, position)
            robot_ids.append(robot_id)
        
        return robot_ids
    
    def get_free_cells(self):
        """
        Get list of free (passage) cells in world coordinates.
        
        Returns:
            List of (x, y) world coordinates of free cells
        """
        free_cells = []
        for y in range(self.maze_grid.shape[0]):
            for x in range(self.maze_grid.shape[1]):
                if self.maze_grid[y, x] == 0:
                    world_x = x * self.cell_size / 2
                    world_y = y * self.cell_size / 2
                    free_cells.append((world_x, world_y))
        return free_cells
    
    def is_position_valid(self, x, y):
        """Check if a world position is in a valid (free) cell."""
        grid_x = int(x / (self.cell_size / 2))
        grid_y = int(y / (self.cell_size / 2))
        
        if (0 <= grid_x < self.maze_grid.shape[1] and 
            0 <= grid_y < self.maze_grid.shape[0]):
            return self.maze_grid[grid_y, grid_x] == 0
        return False
    
    def add_exploration_markers(self, num_markers=5):
        """
        Add visual markers at random locations in the maze for exploration targets.
        
        Args:
            num_markers: Number of markers to place
        
        Returns:
            List of marker positions
        """
        free_cells = self.get_free_cells()
        
        # Filter out cells near entrance
        entrance_x = self.entrance_cell[0] * self.cell_size / 2
        valid_cells = [
            cell for cell in free_cells 
            if np.sqrt((cell[0] - entrance_x)**2 + cell[1]**2) > self.cell_size * 3
        ]
        
        if len(valid_cells) < num_markers:
            valid_cells = free_cells
        
        marker_positions = random.sample(valid_cells, min(num_markers, len(valid_cells)))
        
        for i, (mx, my) in enumerate(marker_positions):
            # Create a small sphere marker
            marker_color = [
                random.uniform(0.5, 1.0),
                random.uniform(0.5, 1.0),
                random.uniform(0.2, 0.5),
                1.0
            ]
            
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.15,
                rgbaColor=marker_color
            )
            
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape,
                basePosition=[mx, my, 0.15]
            )
        
        return marker_positions
    
    def step(self, time_step=1/240):
        """Step the simulation forward."""
        p.stepSimulation()
        if self.gui:
            time.sleep(time_step)
    
    def run_simulation(self, duration=10.0, time_step=1/240):
        """
        Run the simulation for a specified duration.
        
        Args:
            duration: Simulation duration in seconds
            time_step: Time step in seconds
        """
        steps = int(duration / time_step)
        for _ in range(steps):
            self.step(time_step)
    
    def close(self):
        """Close the PyBullet simulation."""
        p.disconnect()
    
    def print_maze(self):
        """Print ASCII representation of the maze."""
        if self.maze_grid is None:
            print("No maze generated.")
            return
        
        for y in range(self.maze_grid.shape[0] - 1, -1, -1):
            row = ""
            for x in range(self.maze_grid.shape[1]):
                if self.maze_grid[y, x] == 1:
                    row += "██"
                else:
                    row += "  "
            print(row)


def main():
    """Main function demonstrating the maze environment."""
    print("=" * 60)
    print("PyBullet Random Maze Environment for Multi-Robot Exploration")
    print("=" * 60)
    
    # Create maze environment
    env = MazeEnvironment(
        maze_size=(10, 10),  # 10x10 cell maze
        cell_size=2.0,       # 2m per cell
        wall_height=2.5,     # 2.5m walls
        gui=True,
        seed=None            # Random maze each time
    )
    
    # Generate and build the maze
    print("\nGenerating maze...")
    env.generate_maze()
    
    # Print ASCII representation
    print("\nMaze layout (ASCII):")
    env.print_maze()
    
    # Build physical walls
    print("\nBuilding walls in PyBullet...")
    env.build_walls()
    
    # Spawn multiple robots outside the entrance
    print("\nSpawning robots at entrance...")
    robot_ids = env.spawn_multi_robots(num_robots=3, spacing=1.5)
    print(f"Spawned {len(robot_ids)} robots with IDs: {robot_ids}")
    
    # Add exploration targets
    print("\nAdding exploration markers...")
    markers = env.add_exploration_markers(num_markers=5)
    print(f"Placed {len(markers)} markers at: {markers}")
    
    # Get spawn position info
    spawn_pos = env.get_spawn_position()
    print(f"\nRobot spawn position: {spawn_pos}")
    
    # Run simulation
    print("\nRunning simulation (press Ctrl+C to exit)...")
    print("Use mouse to rotate view, scroll to zoom")
    
    try:
        while True:
            env.step()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()