"""
Physics Engine Module

PyBullet interface for rendering environments in 3D physics simulation.
Handles wall construction, floor creation, and simulation control.
"""

import random
import time

import numpy as np
import pybullet as p
import pybullet_data


class PybulletRenderer:
    """Manages a procedural environment in PyBullet."""

    def __init__(self, maze_grid, entrance_cell, cell_size=2.0, wall_height=2.0, gui=True):
        self.maze_grid = maze_grid
        self.entrance_cell = entrance_cell
        self.cell_size = cell_size
        self.wall_height = wall_height
        self.gui = gui

        self.wall_ids = []
        self.entrance_position = None

        self._init_pybullet()

    def _init_pybullet(self):
        # 1. Connect to physics engine
        if self.gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # 2. Configure physics
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)

        # 3. Set camera position for GUI
        if self.gui:
            maze_width_m = self.maze_grid.shape[1] * self.cell_size / 2
            maze_height_m = self.maze_grid.shape[0] * self.cell_size / 2
            p.resetDebugVisualizerCamera(
                cameraDistance=max(maze_width_m, maze_height_m) * 0.8,
                cameraYaw=45,
                cameraPitch=-60,
                cameraTargetPosition=[maze_width_m/2, maze_height_m/2, 0]
            )

    # === WALL CONSTRUCTION ===

    def build_walls(self):
        if self.maze_grid is None:
            raise ValueError("maze_grid cannot be None.")

        # 1. Clear existing walls
        for wall_id in self.wall_ids:
            p.removeBody(wall_id)
        self.wall_ids = []

        wall_color = [0.4, 0.35, 0.3, 1.0]
        rows, cols = self.maze_grid.shape

        # 2. Merge adjacent wall cells into strips
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

        # 3. Create floor
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

    # === UTILITY METHODS ===

    def get_spawn_position(self):
        if self.entrance_cell is None:
            raise ValueError("Entrance cell not set.")

        entrance_x, _ = self.entrance_cell
        world_x = entrance_x * self.cell_size / 2
        world_y = -self.cell_size
        world_z = 0.1

        self.entrance_position = (world_x, world_y, world_z)
        return self.entrance_position

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
        for mx, my in marker_positions:
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

    # === SIMULATION CONTROL ===

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


# === MAIN SETUP FUNCTIONS ===

def get_user_input():
    """Get user input for maze configuration."""
    print("=" * 60)
    print("PyBullet Random Maze Environment Generator (Optimized)")
    print("=" * 60)

    # 1. Maze size
    maze_size_input = input("\nEnter maze size (e.g., '10' for 10x10, default=10): ").strip()
    maze_size = int(maze_size_input) if maze_size_input.isdigit() else 10

    # 2. Cell size
    cell_size_input = input("Enter cell size in meters (default=2.0): ").strip()
    try:
        cell_size = float(cell_size_input)
    except ValueError:
        cell_size = 2.0

    # 3. Random seed
    seed_input = input("Enter random seed (press Enter for random): ").strip()
    env_seed = int(seed_input) if seed_input.isdigit() else None

    # 4. Environment type
    print("\nEnvironment types:")
    print("  1. Maze (complex maze with walls)")
    print("  2. Blank box (empty room with single wall in middle)")
    print("  3. Cave (organic cellular automata)")
    print("  4. Tunnel (long winding corridor)")
    print("  5. Rooms (dungeon with connected chambers)")
    print("  6. Sewer (grid of interconnected pipes)")
    print("  7. Corridor Rooms (Central hall with attached rooms)")
    env_type_input = input("Choose environment type (1-7, default=1): ").strip()

    env_types = {
        '2': 'blank_box',
        '3': 'cave',
        '4': 'tunnel',
        '5': 'rooms',
        '6': 'sewer',
        '7': 'corridor_rooms'
    }
    env_type = env_types.get(env_type_input, 'maze')

    # 5. GUI toggle
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
    from simulation.level_generator import MapGenerator

    print(f"\nInitializing {config['maze_size']}x{config['maze_size']} {config['env_type']}...")

    # 1. Generate maze grid
    map_generator = MapGenerator(
        maze_size=(config['maze_size'], config['maze_size']),
        seed=config['seed']
    )
    print("\nGenerating layout...")
    maze_grid, entrance_cell = map_generator.generate_maze(env_type=config['env_type'])

    # 2. Create PyBullet environment
    env = PybulletRenderer(
        maze_grid=maze_grid,
        entrance_cell=entrance_cell,
        cell_size=config['cell_size'],
        wall_height=2.5,
        gui=config['gui']
    )

    # 3. Print ASCII representation
    print("\nMaze layout (ASCII):")
    env.print_maze()

    # 4. Build physical walls
    print("\nBuilding walls in PyBullet (Optimized Merging)...")
    env.build_walls()

    # 5. Add exploration targets
    print("\nAdding exploration markers...")
    markers = env.add_exploration_markers(num_markers=5)
    print(f"Placed {len(markers)} markers")

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
