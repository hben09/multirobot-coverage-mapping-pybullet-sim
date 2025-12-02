
import random
import time
from collections import deque

import numpy as np
import pybullet as p
import pybullet_data


class MapGenerator:
    """Generates a 2D grid representation of various environments."""

    def __init__(self, maze_size=(15, 15), seed=None):
        """
        Initialize the map generator.
        
        Args:
            maze_size: Tuple (width, height) of the maze in cells.
            seed: Random seed for reproducibility.
        """
        self.maze_width, self.maze_height = maze_size
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.maze_grid = None
        self.entrance_cell = None

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
        grid_width = self.maze_width * 2 + 1
        grid_height = self.maze_height * 2 + 1
        self.maze_grid = np.ones((grid_height, grid_width), dtype=int)

        if env_type == 'blank_box':
            self._generate_blank_box()
        elif env_type == 'cave':
            self._generate_cave()
        elif env_type == 'tunnel':
            self._generate_tunnel()
        elif env_type == 'rooms':
            self._generate_rooms()
        else:
            self._generate_recursive_maze()

        return self.maze_grid, self.entrance_cell

    def _generate_recursive_maze(self):
        grid_height, grid_width = self.maze_grid.shape
        self._carve_passages(1, 1)
        entrance_x = random.choice(range(1, grid_width - 1, 2))
        self.maze_grid[0, entrance_x] = 0
        self.entrance_cell = (entrance_x, 0)
        self.maze_grid[1, entrance_x] = 0
        self._add_loops(0.1)

    def _generate_blank_box(self):
        grid_height, grid_width = self.maze_grid.shape
        self.maze_grid[1:-1, 1:-1] = 0
        mid_x = grid_width // 2
        wall_start_y = grid_height // 4
        wall_end_y = 3 * grid_height // 4
        for y in range(wall_start_y, wall_end_y):
            self.maze_grid[y, mid_x] = 1
        entrance_x = grid_width // 2
        self.maze_grid[0, entrance_x] = 0
        self.entrance_cell = (entrance_x, 0)

    def _generate_cave(self):
        grid_height, grid_width = self.maze_grid.shape
        for y in range(1, grid_height - 1):
            for x in range(1, grid_width - 1):
                self.maze_grid[y, x] = 1 if random.random() < 0.45 else 0
        for _ in range(5):
            new_grid = self.maze_grid.copy()
            for y in range(1, grid_height - 1):
                for x in range(1, grid_width - 1):
                    neighbors = np.sum(self.maze_grid[y-1:y+2, x-1:x+2]) - self.maze_grid[y, x]
                    if neighbors > 4:
                        new_grid[y, x] = 1
                    elif neighbors < 4:
                        new_grid[y, x] = 0
            self.maze_grid = new_grid
        self._keep_largest_component(target_val=0)
        entrance_x = grid_width // 2
        lowest_y = 0
        found_connection = False
        for y in range(1, grid_height - 1):
            if found_connection: break
            for x in range(1, grid_width - 1):
                if self.maze_grid[y, x] == 0:
                    lowest_y = y
                    entrance_x = x
                    found_connection = True
                    break
        self.entrance_cell = (entrance_x, 0)
        for y in range(0, lowest_y + 1):
            self.maze_grid[y, entrance_x] = 0

    def _generate_tunnel(self):
        grid_height, grid_width = self.maze_grid.shape
        strip_height = 2
        wall_gap = 2
        num_strips = (grid_height - 2) // (strip_height + wall_gap)
        for i in range(num_strips):
            y_start = 1 + i * (strip_height + wall_gap)
            y_end = y_start + strip_height
            self.maze_grid[y_start:y_end, 1:-1] = 0
            if i < num_strips - 1:
                next_y_start = y_end
                next_y_end = y_end + wall_gap
                if i % 2 == 0:
                    self.maze_grid[next_y_start:next_y_end, -4:-2] = 0
                else:
                    self.maze_grid[next_y_start:next_y_end, 2:4] = 0
        entrance_x = grid_width // 2
        self.maze_grid[0, entrance_x] = 0
        for y in range(0, 2):
            self.maze_grid[y, entrance_x] = 0
        self.entrance_cell = (entrance_x, 0)

    def _generate_rooms(self):
        """
        Generates a dungeon-like environment using MST for connectivity.
        Features: Dynamic room count, guaranteed connectivity, and loops.
        """
        grid_height, grid_width = self.maze_grid.shape
        
        # 1. Reset grid to full walls
        self.maze_grid.fill(1)
        
        rooms = []
        # Heuristic: 1 room per ~50 cells of area, clamped between 5 and 40
        map_area = grid_width * grid_height
        target_rooms = int(map_area / 50)
        target_rooms = max(5, min(40, target_rooms))
        
        # 2. Place Rooms (Random attempts with buffer)
        max_attempts = 200
        for _ in range(max_attempts):
            if len(rooms) >= target_rooms:
                break
                
            w = random.randint(3, 8)
            h = random.randint(3, 8)
            # Align to odd coordinates to maintain grid parity
            x = random.randint(1, max(1, (grid_width - w - 2) // 2)) * 2 + 1
            y = random.randint(1, max(1, (grid_height - h - 2) // 2)) * 2 + 1
            
            new_room = {'x': x, 'y': y, 'w': w, 'h': h, 
                        'cx': x + w // 2, 'cy': y + h // 2}
            
            # Check overlap with 1-cell buffer
            failed = False
            # Boundary check
            if x + w >= grid_width - 1 or y + h >= grid_height - 1:
                failed = True
            
            if not failed:
                for other in rooms:
                    # Enforce 1 cell buffer (margin of 1 around existing room)
                    if (x - 1 < other['x'] + other['w'] + 1 and x + w + 1 > other['x'] - 1 and
                        y - 1 < other['y'] + other['h'] + 1 and y + h + 1 > other['y'] - 1):
                        failed = True
                        break
            
            if not failed:
                # Carve room immediately
                self.maze_grid[y:y+h, x:x+w] = 0
                rooms.append(new_room)

        if not rooms:
            self._generate_blank_box()
            return

        # 3. Connect Rooms (Prim's Algorithm / MST)
        # Start with the first room as 'connected'
        connected_indices = {0}
        unconnected_indices = set(range(1, len(rooms)))
        
        # Track active connections to avoid duplicating them later
        existing_connections = set()

        while unconnected_indices:
            best_dist = float('inf')
            best_pair = None # (connected_idx, unconnected_idx)

            # Find the closest connection between the two sets
            for u_idx in connected_indices:
                u = rooms[u_idx]
                for v_idx in unconnected_indices:
                    v = rooms[v_idx]
                    # Euclidean distance squared
                    dist = (u['cx'] - v['cx'])**2 + (u['cy'] - v['cy'])**2
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (u_idx, v_idx)
            
            if best_pair:
                u_idx, v_idx = best_pair
                self._connect_points(rooms[u_idx], rooms[v_idx])
                connected_indices.add(v_idx)
                unconnected_indices.remove(v_idx)
                
                # Normalize pair for set storage (low, high)
                pair_key = tuple(sorted((u_idx, v_idx)))
                existing_connections.add(pair_key)

        # 4. Add Loops (Re-introduce edges with low probability)
        # This prevents the map from being a perfect tree (single path)
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                if (i, j) in existing_connections:
                    continue
                
                r1, r2 = rooms[i], rooms[j]
                dist = (r1['cx'] - r2['cx'])**2 + (r1['cy'] - r2['cy'])**2
                
                # If rooms are reasonably close, 15% chance to add a tunnel
                if dist < (min(grid_width, grid_height) // 2) ** 2:
                    if random.random() < 0.15:
                        self._connect_points(r1, r2)
                        existing_connections.add((i, j))

        # 5. Set Entrance (First room)
        first_room = rooms[0]
        # Find a valid spot in the first room closest to top
        entrance_x = first_room['cx']
        # Clear path to top of map
        self.maze_grid[0, entrance_x] = 0
        for y in range(0, first_room['y']):
            self.maze_grid[y, entrance_x] = 0
        self.entrance_cell = (entrance_x, 0)

    def _connect_points(self, r1, r2):
        """Draws an L-shaped corridor between two room centers."""
        x1, y1 = r1['cx'], r1['cy']
        x2, y2 = r2['cx'], r2['cy']
        
        # Randomly choose horizontal or vertical first for variety
        if random.random() < 0.5:
            # Horizontal first, then Vertical
            self._carve_h_corridor(x1, x2, y1)
            self._carve_v_corridor(y1, y2, x2)
        else:
            # Vertical first, then Horizontal
            self._carve_v_corridor(y1, y2, x1)
            self._carve_h_corridor(x1, x2, y2)

    def _carve_h_corridor(self, x1, x2, y):
        grid_height, grid_width = self.maze_grid.shape
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if 0 < y < grid_height - 1 and 0 < x < grid_width - 1:
                self.maze_grid[y, x] = 0

    def _carve_v_corridor(self, y1, y2, x):
        grid_height, grid_width = self.maze_grid.shape
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if 0 < y < grid_height - 1 and 0 < x < grid_width - 1:
                self.maze_grid[y, x] = 0

    def _keep_largest_component(self, target_val=0):
        rows, cols = self.maze_grid.shape
        visited = np.zeros_like(self.maze_grid, dtype=bool)
        components = []
        for r in range(rows):
            for c in range(cols):
                if self.maze_grid[r, c] == target_val and not visited[r, c]:
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
        if not components: return
        components.sort(key=len, reverse=True)
        for comp in components[1:]:
            for r, c in comp:
                self.maze_grid[r, c] = 1

    def _carve_passages(self, cx, cy):
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if (0 < nx < self.maze_grid.shape[1] - 1 and
                0 < ny < self.maze_grid.shape[0] - 1):
                if self.maze_grid[ny, nx] == 1:
                    self.maze_grid[cy + dy // 2, cx + dx // 2] = 0
                    self.maze_grid[ny, nx] = 0
                    self._carve_passages(nx, ny)

    def _add_loops(self, probability):
        for y in range(1, self.maze_grid.shape[0] - 1):
            for x in range(1, self.maze_grid.shape[1] - 1):
                if self.maze_grid[y, x] == 1:
                    neighbors = 0
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        if self.maze_grid[y + dy, x + dx] == 0:
                            neighbors += 1
                    if neighbors >= 2 and random.random() < probability:
                        self.maze_grid[y, x] = 0


class PybulletRenderer:
    """Manages a procedural environment in PyBullet."""

    def __init__(self, maze_grid, entrance_cell, cell_size=2.0, wall_height=2.0, gui=True):
        """
        Initialize the PyBullet environment from a maze grid.
        """
        self.maze_grid = maze_grid
        self.entrance_cell = entrance_cell
        self.cell_size = cell_size
        self.wall_height = wall_height
        self.gui = gui
        
        self.wall_ids = []
        self.robot_ids = []
        self.entrance_position = None

        self._init_pybullet()

    def _init_pybullet(self):
        if self.gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)

        if self.gui:
            maze_width_m = self.maze_grid.shape[1] * self.cell_size / 2
            maze_height_m = self.maze_grid.shape[0] * self.cell_size / 2
            p.resetDebugVisualizerCamera(
                cameraDistance=max(maze_width_m, maze_height_m) * 0.8,
                cameraYaw=45,
                cameraPitch=-60,
                cameraTargetPosition=[maze_width_m/2, maze_height_m/2, 0]
            )

    def build_walls(self):
        if self.maze_grid is None:
            raise ValueError("maze_grid cannot be None.")

        for wall_id in self.wall_ids:
            p.removeBody(wall_id)
        self.wall_ids = []

        wall_color = [0.4, 0.35, 0.3, 1.0]
        rows, cols = self.maze_grid.shape

        for y in range(rows):
            x = 0
            while x < cols:
                if self.maze_grid[y, x] == 1:
                    start_x = x
                    length = 1
                    while (x + 1 < cols) and (self.maze_grid[y, x + 1] == 1):
                        x += 1
                        length += 1
                    wall_id = self._create_wall_strip(start_x, y, length, wall_color)
                    self.wall_ids.append(wall_id)
                x += 1
        self._create_floor()
        return self.wall_ids

    def _create_wall_strip(self, start_grid_x, grid_y, length_cells, color):
        spacing = self.cell_size / 2
        total_length = length_cells * spacing
        
        first_block_center_x = start_grid_x * spacing
        center_offset = (length_cells - 1) * spacing / 2
        
        world_x = first_block_center_x + center_offset
        world_y = grid_y * spacing
        world_z = self.wall_height / 2

        half_extent = [total_length / 2, spacing / 2, self.wall_height / 2]

        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extent)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extent, rgbaColor=color)

        wall_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[world_x, world_y, world_z]
        )
        return wall_id

    def _create_floor(self):
        grid_width = self.maze_grid.shape[1]
        grid_height = self.maze_grid.shape[0]
        
        maze_width_meters = grid_width * self.cell_size / 2
        maze_height_meters = grid_height * self.cell_size / 2
        
        spawn_buffer = self.cell_size * 5
        total_width = maze_width_meters
        total_height = maze_height_meters + spawn_buffer
        
        floor_color = [0.3, 0.3, 0.35, 1.0]
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
        
        center_x = maze_width_meters / 2 - (self.cell_size / 4)
        center_y = (maze_height_meters - spawn_buffer) / 2
        center_z = -thickness / 2
        
        floor_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[center_x, center_y, center_z]
        )
        p.changeDynamics(floor_id, -1, lateralFriction=0.8)
        self.wall_ids.append(floor_id)

    def get_spawn_position(self):
        if self.entrance_cell is None:
            raise ValueError("Entrance cell not set.")

        entrance_x, _ = self.entrance_cell
        world_x = entrance_x * self.cell_size / 2
        world_y = -self.cell_size
        world_z = 0.1

        self.entrance_position = (world_x, world_y, world_z)
        return self.entrance_position

    def spawn_robot(self, robot_urdf="r2d2.urdf", position=None, orientation=None):
        if position is None:
            position = self.get_spawn_position()
        if orientation is None:
            orientation = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        try:
            robot_id = p.loadURDF(robot_urdf, position, orientation)
        except Exception as e:
            print(f"Could not load {robot_urdf}, creating simple robot: {e}")
            robot_id = self._create_simple_robot(position, orientation)
        self.robot_ids.append(robot_id)
        return robot_id

    def _create_simple_robot(self, position, orientation):
        robot_size = [0.3, 0.2, 0.15]
        robot_color = [0.2, 0.6, 0.8, 1.0]
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=robot_size)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=robot_size, rgbaColor=robot_color)
        robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        return robot_id

    def spawn_multi_robots(self, num_robots, robot_urdf="r2d2.urdf", spacing=1.0):
        base_position = self.get_spawn_position()
        robot_ids = []
        for i in range(num_robots):
            offset_x = (i - num_robots / 2) * spacing
            position = (
                base_position[0] + offset_x,
                base_position[1] - i * spacing * 0.5,
                base_position[2]
            )
            robot_id = self.spawn_robot(robot_urdf, position)
            robot_ids.append(robot_id)
        return robot_ids

    def get_free_cells(self):
        free_cells = []
        for y in range(self.maze_grid.shape[0]):
            for x in range(self.maze_grid.shape[1]):
                if self.maze_grid[y, x] == 0:
                    world_x = x * self.cell_size / 2
                    world_y = y * self.cell_size / 2
                    free_cells.append((world_x, world_y))
        return free_cells

    def is_position_valid(self, x, y):
        grid_x = int(x / (self.cell_size / 2))
        grid_y = int(y / (self.cell_size / 2))
        if (0 <= grid_x < self.maze_grid.shape[1] and
            0 <= grid_y < self.maze_grid.shape[0]):
            return self.maze_grid[grid_y, grid_x] == 0
        return False

    def add_exploration_markers(self, num_markers=5):
        free_cells = self.get_free_cells()
        entrance_x = self.entrance_cell[0] * self.cell_size / 2
        valid_cells = [
            cell for cell in free_cells
            if np.sqrt((cell[0] - entrance_x)**2 + cell[1]**2) > self.cell_size * 3
        ]
        if len(valid_cells) < num_markers:
            valid_cells = free_cells
        marker_positions = random.sample(valid_cells, min(num_markers, len(valid_cells)))
        for i, (mx, my) in enumerate(marker_positions):
            marker_color = [random.uniform(0.5, 1.0), random.uniform(0.5, 1.0), random.uniform(0.2, 0.5), 1.0]
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.15, rgbaColor=marker_color)
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape, basePosition=[mx, my, 0.15])
        return marker_positions

    def print_maze(self):
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

    def step(self, time_step=1/240):
        p.stepSimulation()
        if self.gui:
            time.sleep(time_step)

    def run_simulation(self, duration=10.0, time_step=1/240):
        steps = int(duration / time_step)
        for _ in range(steps):
            self.step(time_step)

    def close(self):
        p.disconnect()


def get_user_input():
    """Get user input for maze configuration."""
    print("=" * 60)
    print("PyBullet Random Maze Environment Generator (Optimized)")
    print("=" * 60)

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

    return {
        "maze_size": maze_size,
        "cell_size": cell_size,
        "seed": env_seed,
        "env_type": env_type,
        "gui": use_gui,
    }


def setup_environment(config):
    """Initialize and build the environment based on config."""
    print(f"\nInitializing {config['maze_size']}x{config['maze_size']} {config['env_type']}...")

    # Generate the maze grid first
    map_generator = MapGenerator(
        maze_size=(config['maze_size'], config['maze_size']),
        seed=config['seed']
    )
    print("\nGenerating layout...")
    maze_grid, entrance_cell = map_generator.generate_maze(env_type=config['env_type'])

    # Create PyBullet environment from the grid
    env = PybulletRenderer(
        maze_grid=maze_grid,
        entrance_cell=entrance_cell,
        cell_size=config['cell_size'],
        wall_height=2.5,
        gui=config['gui']
    )

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

    return env


def run_simulation_loop(env):
    """Run the main simulation loop."""
    print("\nRunning simulation (press Ctrl+C to exit)...")
    print("Use mouse to rotate view, scroll to zoom")

    try:
        while True:
            env.step()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        env.close()


def main():
    """Main function to run the maze environment generator."""
    config = get_user_input()
    env = setup_environment(config)
    run_simulation_loop(env)


if __name__ == "__main__":
    main()