"""Procedural map generation utilities for grid-based environments.

This module provides the MapGenerator class, which utilizes various algorithms
(Cellular Automata, Recursive Backtracking, Room Placement) to generate 
2D binary grid maps.
"""

import random
from collections import deque

import numpy as np


class MapGenerator:
    """Generates procedural 2D grid maps based on specified environment types.

    Attributes:
        maze_width (int): Width of the logical maze (excluding walls).
        maze_height (int): Height of the logical maze (excluding walls).
        maze_grid (np.ndarray): The generated binary grid (0 for floor, 1 for wall).
        entrance_cell (tuple): Coordinates (x, y) of the map entrance.
    """

    def __init__(self, maze_size=(15, 15), seed=None):
        """Initializes the generator with dimensions and an optional seed.

        Args:
            maze_size (tuple[int, int]): Tuple of (width, height).
            seed (int, optional): Random seed for reproducibility.
        """
        self.maze_width, self.maze_height = maze_size
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.maze_grid = None
        self.entrance_cell = None

    def generate_maze(self, env_type='maze'):
        """Main factory method to generate a map based on environment type.

        Args:
            env_type (str): The style of map to generate. Options include:
                'blank_box', 'cave', 'tunnel', 'rooms', 'sewer', 
                'corridor_rooms', or default 'maze'.

        Returns:
            tuple: (np.ndarray, tuple) representing the grid and entrance (x, y).
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
        elif env_type == 'sewer':
            self._generate_sewer()
        elif env_type == 'corridor_rooms':
            self._generate_corridor_rooms()
        else:
            self._generate_recursive_maze()

        return self.maze_grid, self.entrance_cell

    # -------------------------------------------------------------------------
    # Map Generation Algorithms
    # -------------------------------------------------------------------------

    def _generate_recursive_maze(self):
        """Generates a perfect maze using recursive backtracking, then adds loops."""
        grid_height, grid_width = self.maze_grid.shape

        # Start carving from (1, 1)
        self._carve_passages(1, 1)

        # Create entrance
        entrance_x = random.choice(range(1, grid_width - 1, 2))
        self.maze_grid[0, entrance_x] = 0
        self.maze_grid[1, entrance_x] = 0
        self.entrance_cell = (entrance_x, 0)

        self._add_loops(0.1)

    def _generate_blank_box(self):
        """Generates an open room with a central vertical partition."""
        grid_height, grid_width = self.maze_grid.shape

        self.maze_grid[1:-1, 1:-1] = 0

        # Create central divider
        mid_x = grid_width // 2
        wall_start_y = grid_height // 4
        wall_end_y = 3 * grid_height // 4
        for y in range(wall_start_y, wall_end_y):
            self.maze_grid[y, mid_x] = 1

        entrance_x = grid_width // 2
        self.maze_grid[0, entrance_x] = 0
        self.entrance_cell = (entrance_x, 0)

    def _generate_cave(self):
        """Generates a cave system using cellular automata smoothing."""
        grid_height, grid_width = self.maze_grid.shape

        # Random noise initialization
        for y in range(1, grid_height - 1):
            for x in range(1, grid_width - 1):
                self.maze_grid[y, x] = 1 if random.random() < 0.45 else 0

        # Cellular automata smoothing steps
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

        # Find valid entrance point at the lowest available Y
        entrance_x = grid_width // 2
        lowest_y = 0
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

        self.entrance_cell = (entrance_x, 0)
        for y in range(0, lowest_y + 1):
            self.maze_grid[y, entrance_x] = 0

    def _generate_tunnel(self):
        """Generates horizontal strip tunnels connected at alternating ends."""
        grid_height, grid_width = self.maze_grid.shape

        strip_height = 2
        wall_gap = 2
        num_strips = (grid_height - 2) // (strip_height + wall_gap)

        for i in range(num_strips):
            y_start = 1 + i * (strip_height + wall_gap)
            y_end = y_start + strip_height
            self.maze_grid[y_start:y_end, 1:-1] = 0

            # Connect strips
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
        """Places non-overlapping rooms and connects them using a minimal spanning tree."""
        grid_height, grid_width = self.maze_grid.shape
        self.maze_grid.fill(1)

        map_area = grid_width * grid_height
        target_rooms = max(5, min(40, int(map_area / 50)))

        rooms = []
        max_attempts = 200

        # Attempt to place random non-overlapping rooms
        for _ in range(max_attempts):
            if len(rooms) >= target_rooms:
                break

            w = random.randint(3, 8)
            h = random.randint(3, 8)
            x = random.randint(1, max(1, (grid_width - w - 2) // 2)) * 2 + 1
            y = random.randint(1, max(1, (grid_height - h - 2) // 2)) * 2 + 1

            new_room = {'x': x, 'y': y, 'w': w, 'h': h,
                        'cx': x + w // 2, 'cy': y + h // 2}

            # Check boundaries
            failed = False
            if x + w >= grid_width - 1 or y + h >= grid_height - 1:
                failed = True

            # Check overlap
            if not failed:
                for other in rooms:
                    if (x - 1 < other['x'] + other['w'] + 1 and x + w + 1 > other['x'] - 1 and
                            y - 1 < other['y'] + other['h'] + 1 and y + h + 1 > other['y'] - 1):
                        failed = True
                        break

            if not failed:
                self.maze_grid[y:y+h, x:x+w] = 0
                rooms.append(new_room)

        if not rooms:
            self._generate_blank_box()
            return

        # Connect rooms (Prim's algorithm variation for MST)
        connected_indices = {0}
        unconnected_indices = set(range(1, len(rooms)))
        existing_connections = set()

        while unconnected_indices:
            best_dist = float('inf')
            best_pair = None

            for u_idx in connected_indices:
                u = rooms[u_idx]
                for v_idx in unconnected_indices:
                    v = rooms[v_idx]
                    dist = (u['cx'] - v['cx'])**2 + (u['cy'] - v['cy'])**2

                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (u_idx, v_idx)

            if best_pair:
                u_idx, v_idx = best_pair
                self._connect_points(rooms[u_idx], rooms[v_idx])
                connected_indices.add(v_idx)
                unconnected_indices.remove(v_idx)
                existing_connections.add(tuple(sorted((u_idx, v_idx))))

        # Add random extra connections for loops
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                if (i, j) in existing_connections:
                    continue

                r1, r2 = rooms[i], rooms[j]
                dist = (r1['cx'] - r2['cx'])**2 + (r1['cy'] - r2['cy'])**2

                if dist < (min(grid_width, grid_height) // 2) ** 2:
                    if random.random() < 0.15:
                        self._connect_points(r1, r2)
                        existing_connections.add((i, j))

        first_room = rooms[0]
        entrance_x = first_room['cx']
        self.maze_grid[0, entrance_x] = 0
        for y in range(0, first_room['y']):
            self.maze_grid[y, entrance_x] = 0
        self.entrance_cell = (entrance_x, 0)

    def _generate_sewer(self):
        """Generates a sewer-like grid structure with noise."""
        grid_height, grid_width = self.maze_grid.shape
        self.maze_grid.fill(1)

        spacing_y = 4
        spacing_x = 4

        # Horizontal pipes
        for y in range(2, grid_height - 2, spacing_y):
            if random.random() < 0.80:
                self.maze_grid[y, 1:-1] = 0
                if random.random() < 0.3 and y + 1 < grid_height - 1:
                    self.maze_grid[y + 1, 1:-1] = 0

        # Vertical pipes
        for x in range(2, grid_width - 2, spacing_x):
            if random.random() < 0.70:
                self.maze_grid[1:-1, x] = 0

        mid_x = grid_width // 2
        mid_y = grid_height // 2
        self.maze_grid[1:-1, mid_x] = 0
        self.maze_grid[mid_y, 1:-1] = 0

        # Add noise/rubble
        for y in range(1, grid_height - 1):
            for x in range(1, grid_width - 1):
                if self.maze_grid[y, x] == 0:
                    if abs(x - mid_x) < 3 or y < 3:
                        continue
                    if random.random() < 0.05:
                        self.maze_grid[y, x] = 1

        self._keep_largest_component(target_val=0)

        self.entrance_cell = (mid_x, 0)
        self.maze_grid[0:3, mid_x] = 0

    def _generate_corridor_rooms(self):
        """Generates a central vertical corridor with rooms branching off."""
        grid_height, grid_width = self.maze_grid.shape
        self.maze_grid.fill(1)

        mid_x = grid_width // 2
        self.maze_grid[1:-1, mid_x-1:mid_x+2] = 0

        def place_rooms_on_side(is_left_side):
            current_y = 2

            if is_left_side:
                max_room_w = mid_x - 4
            else:
                max_room_w = grid_width - (mid_x + 3) - 2

            if max_room_w < 4:
                return

            while current_y < grid_height - 5:
                room_h = random.randint(4, 8)
                room_w = random.randint(4, max(4, max_room_w))

                if current_y + room_h >= grid_height - 1:
                    break

                if random.random() < 0.8:
                    if is_left_side:
                        room_x_end = mid_x - 2
                        room_x_start = max(1, room_x_end - room_w)
                        self.maze_grid[current_y:current_y+room_h, room_x_start:room_x_end] = 0
                        door_y = current_y + room_h // 2
                        self.maze_grid[door_y, mid_x-2] = 0
                    else:
                        room_x_start = mid_x + 3
                        room_x_end = min(grid_width - 1, room_x_start + room_w)
                        self.maze_grid[current_y:current_y+room_h, room_x_start:room_x_end] = 0
                        door_y = current_y + room_h // 2
                        self.maze_grid[door_y, mid_x+2] = 0

                current_y += room_h + 2

        place_rooms_on_side(is_left_side=True)
        place_rooms_on_side(is_left_side=False)

        self.entrance_cell = (mid_x, 0)
        self.maze_grid[0:2, mid_x] = 0

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------

    def _connect_points(self, r1, r2):
        """Connects two points (dicts with 'cx','cy') via an L-shaped corridor."""
        x1, y1 = r1['cx'], r1['cy']
        x2, y2 = r2['cx'], r2['cy']

        if random.random() < 0.5:
            self._carve_h_corridor(x1, x2, y1)
            self._carve_v_corridor(y1, y2, x2)
        else:
            self._carve_v_corridor(y1, y2, x1)
            self._carve_h_corridor(x1, x2, y2)

    def _carve_h_corridor(self, x1, x2, y):
        """Carves a horizontal corridor between x1 and x2 at height y."""
        grid_height, grid_width = self.maze_grid.shape
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if 0 < y < grid_height - 1 and 0 < x < grid_width - 1:
                self.maze_grid[y, x] = 0

    def _carve_v_corridor(self, y1, y2, x):
        """Carves a vertical corridor between y1 and y2 at column x."""
        grid_height, grid_width = self.maze_grid.shape
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if 0 < y < grid_height - 1 and 0 < x < grid_width - 1:
                self.maze_grid[y, x] = 0

    def _keep_largest_component(self, target_val=0):
        """Identifies connected components using BFS and fills all but the largest."""
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

        if not components:
            return
        
        # Keep largest, fill others
        components.sort(key=len, reverse=True)
        for comp in components[1:]:
            for r, c in comp:
                self.maze_grid[r, c] = 1

    def _carve_passages(self, cx, cy):
        """Recursive backtracking implementation."""
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
        """Randomly removes walls between passages to create loops."""
        for y in range(1, self.maze_grid.shape[0] - 1):
            for x in range(1, self.maze_grid.shape[1] - 1):
                if self.maze_grid[y, x] == 1:
                    neighbors = 0
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        if self.maze_grid[y + dy, x + dx] == 0:
                            neighbors += 1
                    if neighbors >= 2 and random.random() < probability:
                        self.maze_grid[y, x] = 0