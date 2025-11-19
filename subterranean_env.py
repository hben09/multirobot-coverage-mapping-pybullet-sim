import pybullet as p
import numpy as np
import random


class SubterraneanEnvironment:
    """Generator for randomized subterranean/cave environments"""

    def __init__(self, seed=None):
        """
        Initialize the subterranean environment generator

        Args:
            seed: Random seed for reproducible environments
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.obstacles = []  # Store obstacle IDs for cleanup

    def create_tunnel_network(self, num_tunnels=5, tunnel_length_range=(10, 20),
                             tunnel_width_range=(3, 6)):
        """
        Create a network of interconnected tunnels

        Args:
            num_tunnels: Number of main tunnel segments
            tunnel_length_range: (min, max) length of each tunnel
            tunnel_width_range: (min, max) width of tunnels
        """
        print(f"Generating tunnel network with {num_tunnels} tunnels...")

        # Ground plane
        p.loadURDF("plane.urdf")

        # Create main tunnels radiating from center
        angles = np.linspace(0, 2 * np.pi, num_tunnels, endpoint=False)

        for i, angle in enumerate(angles):
            tunnel_length = random.uniform(*tunnel_length_range)
            tunnel_width = random.uniform(*tunnel_width_range)

            # Create tunnel walls
            self._create_tunnel_walls(
                start_pos=[0, 0],
                angle=angle,
                length=tunnel_length,
                width=tunnel_width
            )

        # Add some connecting passages
        num_connections = max(2, num_tunnels // 2)
        for _ in range(num_connections):
            self._create_random_passage()

        # Add random rock formations/pillars
        self._add_rock_formations(num_rocks=random.randint(10, 20))

        print(f"Tunnel network created with {len(self.obstacles)} obstacles")

    def create_cave_system(self, num_chambers=4, chamber_size_range=(8, 15)):
        """
        Create a cave system with interconnected chambers

        Args:
            num_chambers: Number of cave chambers
            chamber_size_range: (min, max) size of chambers
        """
        print(f"Generating cave system with {num_chambers} chambers...")

        # Ground plane
        p.loadURDF("plane.urdf")

        # Generate chamber positions in a rough circle
        chamber_positions = []
        radius = 15

        for i in range(num_chambers):
            angle = (2 * np.pi * i) / num_chambers + random.uniform(-0.3, 0.3)
            x = radius * np.cos(angle) + random.uniform(-3, 3)
            y = radius * np.sin(angle) + random.uniform(-3, 3)
            chamber_positions.append([x, y])

        # Add central chamber
        chamber_positions.insert(0, [0, 0])

        # Create chambers (clear areas surrounded by walls)
        for i, pos in enumerate(chamber_positions):
            chamber_size = random.uniform(*chamber_size_range)
            self._create_chamber_walls(pos, chamber_size)

        # Connect chambers with passages
        for i in range(len(chamber_positions)):
            if i < len(chamber_positions) - 1:
                self._create_passage_between_points(
                    chamber_positions[i],
                    chamber_positions[i + 1],
                    width=random.uniform(3, 5)
                )

        # Add stalactites/stalagmites (represented as columns)
        self._add_cave_formations(num_formations=random.randint(15, 30))

        print(f"Cave system created with {len(self.obstacles)} obstacles")

    def create_maze_like_mine(self, grid_size=5, cell_size=6, wall_probability=0.3):
        """
        Create a maze-like mine shaft environment

        Args:
            grid_size: Number of cells in each direction
            cell_size: Size of each cell in meters
            wall_probability: Probability of a wall segment appearing
        """
        print(f"Generating maze-like mine ({grid_size}x{grid_size})...")

        # Ground plane
        p.loadURDF("plane.urdf")

        # Create outer boundary
        boundary_size = grid_size * cell_size
        self._create_boundary_walls(boundary_size)

        # Create internal maze walls
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i - grid_size // 2) * cell_size
                y = (j - grid_size // 2) * cell_size

                # Skip center area to ensure start location is clear
                if abs(i - grid_size // 2) <= 1 and abs(j - grid_size // 2) <= 1:
                    continue

                # Randomly add vertical walls
                if random.random() < wall_probability:
                    self._create_wall_segment(
                        [x + cell_size / 2, y],
                        length=cell_size,
                        orientation='vertical'
                    )

                # Randomly add horizontal walls
                if random.random() < wall_probability:
                    self._create_wall_segment(
                        [x, y + cell_size / 2],
                        length=cell_size,
                        orientation='horizontal'
                    )

        # Add mining equipment obstacles
        self._add_mining_obstacles(num_obstacles=random.randint(8, 15))

        print(f"Mine environment created with {len(self.obstacles)} obstacles")

    def _create_tunnel_walls(self, start_pos, angle, length, width):
        """Create walls for a tunnel segment"""
        # Left wall
        wall_pos_left = [
            start_pos[0] + (length / 2) * np.cos(angle) - (width / 2) * np.sin(angle),
            start_pos[1] + (length / 2) * np.sin(angle) + (width / 2) * np.cos(angle),
            1
        ]

        # Right wall
        wall_pos_right = [
            start_pos[0] + (length / 2) * np.cos(angle) + (width / 2) * np.sin(angle),
            start_pos[1] + (length / 2) * np.sin(angle) - (width / 2) * np.cos(angle),
            1
        ]

        wall_half_extents = [length / 2, 0.3, 1]

        # Create left wall
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_half_extents)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=wall_half_extents,
                                          rgbaColor=[0.4, 0.3, 0.2, 1])

        orientation = p.getQuaternionFromEuler([0, 0, angle])

        wall_left = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                                     baseVisualShapeIndex=visual_shape,
                                     basePosition=wall_pos_left,
                                     baseOrientation=orientation)
        self.obstacles.append(wall_left)

        # Create right wall
        wall_right = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                                      baseVisualShapeIndex=visual_shape,
                                      basePosition=wall_pos_right,
                                      baseOrientation=orientation)
        self.obstacles.append(wall_right)

    def _create_random_passage(self):
        """Create a random connecting passage"""
        start_x = random.uniform(-15, 15)
        start_y = random.uniform(-15, 15)
        angle = random.uniform(0, 2 * np.pi)
        length = random.uniform(5, 12)
        width = random.uniform(3, 5)

        self._create_tunnel_walls([start_x, start_y], angle, length, width)

    def _create_chamber_walls(self, center_pos, size):
        """Create walls around a chamber perimeter"""
        # Create octagonal chamber walls
        num_sides = 8
        wall_length = size / 3

        for i in range(num_sides):
            angle = (2 * np.pi * i) / num_sides

            # Position wall segment on perimeter
            wall_x = center_pos[0] + (size / 2) * np.cos(angle)
            wall_y = center_pos[1] + (size / 2) * np.sin(angle)

            # Create wall
            collision_shape = p.createCollisionShape(p.GEOM_BOX,
                                                    halfExtents=[wall_length / 2, 0.3, 1.5])
            visual_shape = p.createVisualShape(p.GEOM_BOX,
                                              halfExtents=[wall_length / 2, 0.3, 1.5],
                                              rgbaColor=[0.3, 0.3, 0.3, 1])

            orientation = p.getQuaternionFromEuler([0, 0, angle + np.pi / 2])

            wall = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                                    baseVisualShapeIndex=visual_shape,
                                    basePosition=[wall_x, wall_y, 1.5],
                                    baseOrientation=orientation)
            self.obstacles.append(wall)

    def _create_passage_between_points(self, pos1, pos2, width=4):
        """Create a passage connecting two points"""
        # Calculate angle and distance
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        angle = np.arctan2(dy, dx)
        length = np.sqrt(dx**2 + dy**2)

        # Create passage walls
        self._create_tunnel_walls(pos1, angle, length, width)

    def _create_boundary_walls(self, size):
        """Create outer boundary walls"""
        wall_height = 2
        wall_thickness = 0.5

        positions = [
            [0, size / 2, wall_height],   # North
            [0, -size / 2, wall_height],  # South
            [size / 2, 0, wall_height],   # East
            [-size / 2, 0, wall_height]   # West
        ]

        for i, pos in enumerate(positions):
            if i < 2:  # North/South walls
                half_extents = [size / 2, wall_thickness, wall_height]
                orientation = p.getQuaternionFromEuler([0, 0, 0])
            else:  # East/West walls
                half_extents = [wall_thickness, size / 2, wall_height]
                orientation = p.getQuaternionFromEuler([0, 0, 0])

            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                              rgbaColor=[0.2, 0.2, 0.2, 1])

            wall = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                                    baseVisualShapeIndex=visual_shape,
                                    basePosition=pos, baseOrientation=orientation)
            self.obstacles.append(wall)

    def _create_wall_segment(self, position, length, orientation='horizontal'):
        """Create a single wall segment for maze"""
        wall_height = 1.5
        wall_thickness = 0.3

        if orientation == 'horizontal':
            half_extents = [length / 2, wall_thickness, wall_height]
            angle = 0
        else:  # vertical
            half_extents = [wall_thickness, length / 2, wall_height]
            angle = 0

        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                          rgbaColor=[0.35, 0.3, 0.25, 1])

        wall = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                                baseVisualShapeIndex=visual_shape,
                                basePosition=[position[0], position[1], wall_height])
        self.obstacles.append(wall)

    def _add_rock_formations(self, num_rocks):
        """Add random rock formations/pillars"""
        for _ in range(num_rocks):
            x = random.uniform(-20, 20)
            y = random.uniform(-20, 20)

            # Skip if too close to center
            if np.sqrt(x**2 + y**2) < 3:
                continue

            # Random rock size
            radius = random.uniform(0.5, 1.5)
            height = random.uniform(0.8, 2.5)

            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER,
                                                    radius=radius, height=height)
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER,
                                              radius=radius, length=height,
                                              rgbaColor=[0.5, 0.4, 0.3, 1])

            rock = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                                    baseVisualShapeIndex=visual_shape,
                                    basePosition=[x, y, height / 2])
            self.obstacles.append(rock)

    def _add_cave_formations(self, num_formations):
        """Add stalactites/stalagmites"""
        for _ in range(num_formations):
            x = random.uniform(-25, 25)
            y = random.uniform(-25, 25)

            # Random formation size
            radius = random.uniform(0.3, 0.8)
            height = random.uniform(1.0, 3.0)

            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER,
                                                    radius=radius, height=height)
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER,
                                              radius=radius, length=height,
                                              rgbaColor=[0.4, 0.4, 0.45, 1])

            formation = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                                         baseVisualShapeIndex=visual_shape,
                                         basePosition=[x, y, height / 2])
            self.obstacles.append(formation)

    def _add_mining_obstacles(self, num_obstacles):
        """Add mining equipment and debris"""
        obstacle_types = ['box', 'cylinder', 'box']  # More boxes for crates/equipment

        for _ in range(num_obstacles):
            x = random.uniform(-20, 20)
            y = random.uniform(-20, 20)

            # Skip if too close to center
            if np.sqrt(x**2 + y**2) < 4:
                continue

            obstacle_type = random.choice(obstacle_types)

            if obstacle_type == 'box':
                # Crates or equipment boxes
                size = random.uniform(0.8, 2.0)
                half_extents = [size / 2, size / 2, size / 2]

                collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
                visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                                  rgbaColor=[0.6, 0.5, 0.3, 1])

                obstacle = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                                            baseVisualShapeIndex=visual_shape,
                                            basePosition=[x, y, size / 2])
            else:
                # Cylindrical equipment (barrels, support beams)
                radius = random.uniform(0.4, 0.8)
                height = random.uniform(1.0, 2.0)

                collision_shape = p.createCollisionShape(p.GEOM_CYLINDER,
                                                        radius=radius, height=height)
                visual_shape = p.createVisualShape(p.GEOM_CYLINDER,
                                                  radius=radius, length=height,
                                                  rgbaColor=[0.5, 0.5, 0.5, 1])

                obstacle = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                                            baseVisualShapeIndex=visual_shape,
                                            basePosition=[x, y, height / 2])

            self.obstacles.append(obstacle)

    def get_environment_bounds(self):
        """Return suggested map bounds for the environment"""
        # Calculate bounds based on obstacle positions
        if not self.obstacles:
            return {'x_min': -15, 'x_max': 15, 'y_min': -15, 'y_max': 15}

        # Default generous bounds for subterranean environments
        return {'x_min': -30, 'x_max': 30, 'y_min': -30, 'y_max': 30}

    def cleanup(self):
        """Remove all obstacles"""
        for obstacle_id in self.obstacles:
            p.removeBody(obstacle_id)
        self.obstacles.clear()


def demo_environments():
    """Demo function to visualize different environment types"""
    import pybullet_data

    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)

    # Create environment generator
    env = SubterraneanEnvironment(seed=42)

    # Choose environment type
    print("\nAvailable environments:")
    print("1. Tunnel Network")
    print("2. Cave System")
    print("3. Maze-like Mine")

    choice = input("\nSelect environment (1-3): ")

    if choice == "1":
        env.create_tunnel_network(num_tunnels=6)
    elif choice == "2":
        env.create_cave_system(num_chambers=5)
    elif choice == "3":
        env.create_maze_like_mine(grid_size=6, wall_probability=0.35)
    else:
        print("Invalid choice, creating tunnel network by default")
        env.create_tunnel_network()

    print("\nEnvironment created! Bounds:", env.get_environment_bounds())
    print("Press Ctrl+C to exit...")

    # Keep simulation running
    try:
        while True:
            p.stepSimulation()
    except KeyboardInterrupt:
        print("\nCleaning up...")
        env.cleanup()
        p.disconnect()


if __name__ == "__main__":
    demo_environments()
