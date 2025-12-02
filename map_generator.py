
import random
import time
from collections import deque

import numpy as np
import pybullet as p
import pybullet_data


class ProceduralEnvironment:
    """Generates and manages procedural environments (mazes, caves, tunnels, rooms) in PyBullet."""

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
        self.entrance_cell = None
        self.entrance_position = None

        self._init_pybullet()

    # ========================================================================
    # Initialization Methods
    # ========================================================================

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

    # ========================================================================
    # Maze Generation Methods
    # ========================================================================

    def generate_maze(self, env_type='maze'):
        """
        Generate an environment based on the specified type.

        Args:
            env_type: 'maze', 'blank_box', 'cave', 'tunnel', 'rooms'

        Returns:
            The maze is represented as a 2D grid where:
            - 0 = passage (empty space)
            - 1 = wall
        """
        # Initialize grid with all walls
        # Use odd dimensions for proper maze generation
        grid_width = self.maze_width * 2 + 1
        grid_height = self.maze_height * 2 + 1
        self.maze_grid = np.ones((grid_height, grid_width), dtype=int)

        # Generate based on type
        if env_type == 'blank_box':
            self._generate_blank_box()
        elif env_type == 'cave':
            self._generate_cave()
        elif env_type == 'tunnel':
            self._generate_tunnel()
        elif env_type == 'rooms':
            self._generate_rooms()
        else:
            # Default to Maze
            self._generate_recursive_maze()

        return self.maze_grid

    def _generate_recursive_maze(self):
        """Generate standard recursive backtracker maze."""
        grid_height, grid_width = self.maze_grid.shape
        self._carve_passages(1, 1)

        # Create entrance on the outside (bottom wall)
        entrance_x = random.choice(range(1, grid_width - 1, 2))
        self.maze_grid[0, entrance_x] = 0
        self.entrance_cell = (entrance_x, 0)

        # Ensure path to start
        self.maze_grid[1, entrance_x] = 0

        # Add connectivity
        self._add_loops(0.1)

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

    def _generate_cave(self):
        """Generate an organic cave system using cellular automata."""
        grid_height, grid_width = self.maze_grid.shape

        # 1. Random noise initialization (45% walls)
        for y in range(1, grid_height - 1):
            for x in range(1, grid_width - 1):
                self.maze_grid[y, x] = 1 if random.random() < 0.45 else 0

        # 2. Cellular automata smoothing (5 iterations)
        for _ in range(5):
            new_grid = self.maze_grid.copy()
            for y in range(1, grid_height - 1):
                for x in range(1, grid_width - 1):
                    # Count wall neighbors (including diagonals)
                    neighbors = np.sum(self.maze_grid[y-1:y+2, x-1:x+2]) - self.maze_grid[y, x]

                    if neighbors > 4:
                        new_grid[y, x] = 1  # Become wall
                    elif neighbors < 4:
                        new_grid[y, x] = 0  # Become space
            self.maze_grid = new_grid

        # 3. Ensure connectivity (remove isolated caves)
        self._keep_largest_component(target_val=0)

        # 4. Create entrance and ensure it connects to the largest component
        # Find the lowest '0' cell in the grid
        entrance_x = grid_width // 2
        lowest_y = 0

        # Search for a connection point from bottom up
        found_connection = False
        for y in range(1, grid_height - 1):
            if found_connection:
                break
            for x in range(1, grid_width - 1):
                if self.maze_grid[y, x] == 0:
                    lowest_y = y
                    entrance_x = x
                    found_connection = True
                    break

        # Dig a tunnel from the bottom edge to that point
        self.entrance_cell = (entrance_x, 0)
        for y in range(0, lowest_y + 1):
            self.maze_grid[y, entrance_x] = 0

    def _generate_tunnel(self):
        """Generate a long winding tunnel (snake pattern)."""
        grid_height, grid_width = self.maze_grid.shape

        # Clear specific horizontal strips
        # Leave a wall gap of 2 blocks between strips
        strip_height = 2
        wall_gap = 2

        # Calculate how many strips fit
        num_strips = (grid_height - 2) // (strip_height + wall_gap)

        for i in range(num_strips):
            y_start = 1 + i * (strip_height + wall_gap)
            y_end = y_start + strip_height

            # Carve the horizontal strip
            self.maze_grid[y_start:y_end, 1:-1] = 0

            # Connect to the next strip (alternate left/right)
            if i < num_strips - 1:
                next_y_start = y_end
                next_y_end = y_end + wall_gap

                if i % 2 == 0:
                    # Connect on right (stay away from perimeter wall)
                    self.maze_grid[next_y_start:next_y_end, -4:-2] = 0
                else:
                    # Connect on left (stay away from perimeter wall)
                    self.maze_grid[next_y_start:next_y_end, 2:4] = 0

        # Create entrance connecting to the first strip
        entrance_x = grid_width // 2
        self.maze_grid[0, entrance_x] = 0
        # Carve up to the first strip
        for y in range(0, 2):
            self.maze_grid[y, entrance_x] = 0
        self.entrance_cell = (entrance_x, 0)

    def _generate_rooms(self):
        """Generate connected rooms (dungeon style)."""
        grid_height, grid_width = self.maze_grid.shape

        rooms = []
        num_rooms = 15  # Attempt to place 15 rooms

        for _ in range(num_rooms):
            # Random size
            w = random.randint(3, 8)
            h = random.randint(3, 8)
            # Random pos (ensure odd coords for alignment)
            # Keep rooms at least 1 cell away from perimeter walls
            x = random.randint(1, max(1, (grid_width - w - 2) // 2)) * 2 + 1
            y = random.randint(1, max(1, (grid_height - h - 2) // 2)) * 2 + 1

            new_room = {'x': x, 'y': y, 'w': w, 'h': h}

            # Check that room doesn't touch perimeter walls
            failed = False
            if x <= 0 or y <= 0 or x + w >= grid_width - 1 or y + h >= grid_height - 1:
                failed = True

            # Simple overlap check with other rooms
            if not failed:
                for other in rooms:
                    if (x < other['x'] + other['w'] and x + w > other['x'] and
                        y < other['y'] + other['h'] and y + h > other['y']):
                        failed = True
                        break

            if not failed:
                # Carve room
                self.maze_grid[y:y+h, x:x+w] = 0
                rooms.append(new_room)

        # Connect rooms sequentially to ensure graph connectivity
        # Sort rooms by Y then X to make clean paths
        rooms.sort(key=lambda r: (r['y'], r['x']))

        for i in range(len(rooms) - 1):
            r1 = rooms[i]
            r2 = rooms[i+1]

            # Center points
            c1 = (r1['x'] + r1['w']//2, r1['y'] + r1['h']//2)
            c2 = (r2['x'] + r2['w']//2, r2['y'] + r2['h']//2)

            # Carve L-shaped corridor
            if random.random() < 0.5:
                # Horizontal then vertical
                self._carve_h_corridor(c1[0], c2[0], c1[1])
                self._carve_v_corridor(c1[1], c2[1], c2[0])
            else:
                # Vertical then horizontal
                self._carve_v_corridor(c1[1], c2[1], c1[0])
                self._carve_h_corridor(c1[0], c2[0], c2[1])

        # Connect entrance to the first (lowest) room
        if rooms:
            first_room = rooms[0]
            entrance_x = first_room['x'] + first_room['w'] // 2
            self.maze_grid[0, entrance_x] = 0
            # Carve up to the room
            for y in range(0, first_room['y']):
                self.maze_grid[y, entrance_x] = 0
            self.entrance_cell = (entrance_x, 0)
        else:
            # Fallback if no rooms placed
            self._generate_blank_box()

    # ========================================================================
    # Helper Methods for Maze Generation
    # ========================================================================

    def _carve_h_corridor(self, x1, x2, y):
        """Carve a horizontal corridor between two x coordinates at a given y."""
        # Keep corridor away from perimeter walls
        grid_height, grid_width = self.maze_grid.shape
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if 0 < y < grid_height - 1 and 0 < x < grid_width - 1:
                self.maze_grid[y, x] = 0

    def _carve_v_corridor(self, y1, y2, x):
        """Carve a vertical corridor between two y coordinates at a given x."""
        # Keep corridor away from perimeter walls (except for entrance at y=0)
        grid_height, grid_width = self.maze_grid.shape
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if 0 < y < grid_height - 1 and 0 < x < grid_width - 1:
                self.maze_grid[y, x] = 0

    def _keep_largest_component(self, target_val=0):
        """Keep only the largest connected component of target_val, fill others."""
        rows, cols = self.maze_grid.shape
        visited = np.zeros_like(self.maze_grid, dtype=bool)
        components = []

        for r in range(rows):
            for c in range(cols):
                if self.maze_grid[r, c] == target_val and not visited[r, c]:
                    # Start BFS
                    component = []
                    queue = deque([(r, c)])
                    visited[r, c] = True
                    while queue:
                        curr_r, curr_c = queue.popleft()
                        component.append((curr_r, curr_c))

                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if (0 <= nr < rows and 0 <= nc < cols and
                                self.maze_grid[nr, nc] == target_val and not visited[nr, nc]):
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                    components.append(component)

        if not components:
            return

        # Sort by size (largest first)
        components.sort(key=len, reverse=True)

        # Fill all other components with 1 (wall)
        for comp in components[1:]:
            for r, c in comp:
                self.maze_grid[r, c] = 1  # Fill

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

    # ========================================================================
    # Wall Building Methods (Optimized)
    # ========================================================================

    def build_walls(self):
        """
        Build physical walls in PyBullet based on the maze grid.
        
        PERFORMANCE OPTIMIZATION: 
        Uses Greedy Geometry Merging to combine adjacent horizontal wall cells
        into single rectangular strips. This drastically reduces the number of 
        physics bodies and improves simulation performance.
        """
        if self.maze_grid is None:
            self.generate_maze()

        # Clear existing walls
        for wall_id in self.wall_ids:
            p.removeBody(wall_id)
        self.wall_ids = []

        # Wall visual and collision properties
        wall_color = [0.4, 0.35, 0.3, 1.0]  # Stone-like color
        
        rows, cols = self.maze_grid.shape

        # Scan each row to find continuous horizontal segments of walls
        for y in range(rows):
            x = 0
            while x < cols:
                if self.maze_grid[y, x] == 1:
                    # Found a wall, determine its horizontal length
                    start_x = x
                    length = 1
                    
                    # Look ahead
                    while (x + 1 < cols) and (self.maze_grid[y, x + 1] == 1):
                        x += 1
                        length += 1
                    
                    # Create a merged strip wall
                    wall_id = self._create_wall_strip(start_x, y, length, wall_color)
                    self.wall_ids.append(wall_id)
                
                # Move to next cell
                x += 1

        # Add floor to ensure robots don't fall (replaces static plane)
        self._create_floor()

        return self.wall_ids

    def _create_wall_strip(self, start_grid_x, grid_y, length_cells, color):
        """
        Create a merged wall strip covering multiple grid cells.
        
        Args:
            start_grid_x: The starting x index in the grid.
            grid_y: The y index in the grid.
            length_cells: How many cells long the wall is.
            color: RGBA color list.
        """
        # Calculate physical dimensions
        # Note: In the original script logic, the physical spacing was cell_size/2.
        # Grid index 0 -> World 0.0
        # Grid index 1 -> World 1.0 (if cell_size=2.0)
        # Block width was cell_size/2 (so 1.0)
        
        spacing = self.cell_size / 2
        total_length = length_cells * spacing
        
        # Calculate center position of the strip
        # Start World X = start_grid_x * spacing
        # The center is offset by half the total length minus half the block width
        # (Since the original blocks were centered on the grid points)
        
        # Center of the first block in the strip
        first_block_center_x = start_grid_x * spacing
        
        # Center of the strip is shifted from the first block center
        # shift = (length_cells - 1) * spacing / 2
        center_offset = (length_cells - 1) * spacing / 2
        
        world_x = first_block_center_x + center_offset
        world_y = grid_y * spacing
        world_z = self.wall_height / 2

        half_extent = [
            total_length / 2,     # Length
            spacing / 2,          # Thickness (matches original block width)
            self.wall_height / 2  # Height
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

    def _create_floor(self):
        """Create a procedural floor matching the maze dimensions plus spawn area."""
        # Calculate full floor dimensions
        grid_width = self.maze_grid.shape[1]
        grid_height = self.maze_grid.shape[0]
        
        # Maze dimensions in meters
        maze_width_meters = grid_width * self.cell_size / 2
        maze_height_meters = grid_height * self.cell_size / 2
        
        # Extend floor backwards (negative Y) for the spawn area
        # Robots spawn at y = -cell_size. We add a buffer of 5 cells.
        spawn_buffer = self.cell_size * 5
        
        total_width = maze_width_meters
        total_height = maze_height_meters + spawn_buffer
        
        # Dark grey concrete color
        floor_color = [0.3, 0.3, 0.35, 1.0]
        
        # Thickness of floor (z-dimension)
        thickness = 0.1
        
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[total_width / 2, total_height / 2, thickness / 2]
        )

        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[total_width / 2, total_height / 2, thickness / 2],
            rgbaColor=floor_color
        )
        
        # Calculate center position
        center_x = maze_width_meters / 2 - (self.cell_size / 4) # Adjust for grid 0 start
        center_y = (maze_height_meters - spawn_buffer) / 2
        center_z = -thickness / 2
        
        floor_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[center_x, center_y, center_z]
        )
        
        # Set standard friction
        p.changeDynamics(floor_id, -1, lateralFriction=0.8)
        
        self.wall_ids.append(floor_id)

    # ========================================================================
    # Robot Spawning Methods
    # ========================================================================

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

    # ========================================================================
    # Utility Methods
    # ========================================================================

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

    # ========================================================================
    # Simulation Control Methods
    # ========================================================================

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


def main():
    """Main function demonstrating the maze environment with user input."""
    print("=" * 60)
    print("PyBullet Random Maze Environment Generator (Optimized)")
    print("=" * 60)

    # --- Input Configuration ---

    # 1. Maze Size
    maze_size_input = input("\nEnter maze size (e.g., '10' for 10x10, default=10): ").strip()
    maze_size = int(maze_size_input) if maze_size_input.isdigit() else 10

    # 2. Cell Size
    cell_size_input = input("Enter cell size in meters (default=2.0): ").strip()
    try:
        cell_size = float(cell_size_input)
    except ValueError:
        cell_size = 2.0

    # 3. Random Seed
    seed_input = input("Enter random seed (press Enter for random): ").strip()
    env_seed = int(seed_input) if seed_input.isdigit() else None

    # 4. Environment Type
    print("\nEnvironment types:")
    print("  1. Maze (complex maze with walls)")
    print("  2. Blank box (empty room with single wall in middle)")
    print("  3. Cave (organic cellular automata)")
    print("  4. Tunnel (long winding corridor)")
    print("  5. Rooms (dungeon with connected chambers)")
    env_type_input = input("Choose environment type (1-5, default=1): ").strip()

    env_types = {'2': 'blank_box', '3': 'cave', '4': 'tunnel', '5': 'rooms'}
    env_type = env_types.get(env_type_input, 'maze')

    # 5. GUI Toggle
    use_gui = input("Show PyBullet 3D window? (y/n, default=y): ").strip().lower() != 'n'

    # --- Environment Creation ---

    print(f"\nInitializing {maze_size}x{maze_size} {env_type}...")

    # Create maze environment
    env = ProceduralEnvironment(
        maze_size=(maze_size, maze_size),
        cell_size=cell_size,
        wall_height=2.5,
        gui=use_gui,
        seed=env_seed
    )

    # Generate and build the maze based on user choice
    print("\nGenerating layout...")
    env.generate_maze(env_type=env_type)

    # Print ASCII representation
    print("\nMaze layout (ASCII):")
    env.print_maze()

    # Build physical walls
    print("\nBuilding walls in PyBullet (Optimized Merging)...")
    env.build_walls()

    # Spawn multiple robots outside the entrance
    print("\nSpawning robots at entrance...")
    robot_ids = env.spawn_multi_robots(num_robots=3, spacing=1.5)
    print(f"Spawned {len(robot_ids)} robots with IDs: {robot_ids}")

    # Add exploration targets
    print("\nAdding exploration markers...")
    markers = env.add_exploration_markers(num_markers=5)
    print(f"Placed {len(markers)} markers")

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