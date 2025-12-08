import pybullet as p
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq
from collections import defaultdict
from environment import MapGenerator, PybulletRenderer
from sim_logger import SimulationLogger


# =============================================================================
# Numba JIT-compiled A* Pathfinding (6-18x faster than pure Python)
# =============================================================================
from numba import njit
print("✓ Numba detected - A* pathfinding will be JIT-compiled for maximum speed")

# =============================================================================
# Numba-accelerated Occupancy Grid (for fast LIDAR processing)
# =============================================================================
from numba_occupancy import NumbaOccupancyGrid
print("✓ Numba occupancy grid loaded - LIDAR processing will be accelerated")

@njit(cache=True)
def _numba_heuristic(ax, ay, bx, by):
    """Octile distance heuristic - exact for 8-way movement."""
    dx = abs(bx - ax)
    dy = abs(by - ay)
    if dx > dy:
        return dx + 0.41421356 * dy
    else:
        return dy + 0.41421356 * dx

@njit(cache=True)
def _numba_heap_push(heap, heap_size, f, x, y):
    """Push item onto min-heap."""
    idx = heap_size
    heap[idx, 0] = f
    heap[idx, 1] = x
    heap[idx, 2] = y
    
    while idx > 0:
        parent = (idx - 1) // 2
        if heap[parent, 0] > heap[idx, 0]:
            heap[parent, 0], heap[idx, 0] = heap[idx, 0], heap[parent, 0]
            heap[parent, 1], heap[idx, 1] = heap[idx, 1], heap[parent, 1]
            heap[parent, 2], heap[idx, 2] = heap[idx, 2], heap[parent, 2]
            idx = parent
        else:
            break
    
    return heap_size + 1

@njit(cache=True)
def _numba_heap_pop(heap, heap_size):
    """Pop minimum item from heap."""
    if heap_size == 0:
        return 0.0, -1, -1, 0
    
    f = heap[0, 0]
    x = int(heap[0, 1])
    y = int(heap[0, 2])
    
    heap_size -= 1
    if heap_size > 0:
        heap[0, 0] = heap[heap_size, 0]
        heap[0, 1] = heap[heap_size, 1]
        heap[0, 2] = heap[heap_size, 2]
        
        idx = 0
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            smallest = idx
            
            if left < heap_size and heap[left, 0] < heap[smallest, 0]:
                smallest = left
            if right < heap_size and heap[right, 0] < heap[smallest, 0]:
                smallest = right
            
            if smallest != idx:
                heap[idx, 0], heap[smallest, 0] = heap[smallest, 0], heap[idx, 0]
                heap[idx, 1], heap[smallest, 1] = heap[smallest, 1], heap[idx, 1]
                heap[idx, 2], heap[smallest, 2] = heap[smallest, 2], heap[idx, 2]
                idx = smallest
            else:
                break
    
    return f, x, y, heap_size

@njit(cache=True)
def _numba_astar_core(
    grid, start_x, start_y, goal_x, goal_y,
    grid_offset_x, grid_offset_y, use_inflation, max_iterations=50000
):
    """
    Numba JIT-compiled A* core algorithm.
    Grid values: 0=unknown, 1=free, 2=obstacle, 3=inflated
    """
    dx_arr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
    dy_arr = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
    costs = np.array([1.41421356, 1.0, 1.41421356, 1.0, 1.0, 1.41421356, 1.0, 1.41421356], dtype=np.float64)
    
    height, width = grid.shape
    
    sx = start_x - grid_offset_x
    sy = start_y - grid_offset_y
    gx = goal_x - grid_offset_x
    gy = goal_y - grid_offset_y
    
    if sx < 0 or sx >= width or sy < 0 or sy >= height:
        return np.empty((0, 2), dtype=np.int32), False
    if gx < 0 or gx >= width or gy < 0 or gy >= height:
        return np.empty((0, 2), dtype=np.int32), False
    
    g_score = np.full((height, width), np.inf, dtype=np.float64)
    g_score[sy, sx] = 0.0
    
    parent_dir = np.full((height, width), -1, dtype=np.int8)
    closed = np.zeros((height, width), dtype=np.bool_)
    
    max_heap_size = height * width
    heap = np.zeros((max_heap_size, 3), dtype=np.float64)
    heap_size = 0
    
    f_start = _numba_heuristic(sx, sy, gx, gy)
    heap_size = _numba_heap_push(heap, heap_size, f_start, float(sx), float(sy))
    
    iterations = 0
    found = False
    
    while iterations < max_iterations and heap_size > 0:
        iterations += 1
        
        _, cx, cy, heap_size = _numba_heap_pop(heap, heap_size)
        
        if cx == -1:
            break
        
        if closed[cy, cx]:
            continue
        
        closed[cy, cx] = True
        
        if cx == gx and cy == gy:
            found = True
            break
        
        current_g = g_score[cy, cx]
        
        for i in range(8):
            nx = cx + dx_arr[i]
            ny = cy + dy_arr[i]
            
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            
            if closed[ny, nx]:
                continue
            
            cell_val = grid[ny, nx]
            
            if cell_val == 2:  # Obstacle
                continue
            
            if cell_val == 0:  # Unknown
                continue
            
            if use_inflation and cell_val == 3:
                if not (nx == sx and ny == sy) and not (nx == gx and ny == gy):
                    continue
            
            # Diagonal corner check
            if dx_arr[i] != 0 and dy_arr[i] != 0:
                card1_x = cx + dx_arr[i]
                card1_y = cy
                card2_x = cx
                card2_y = cy + dy_arr[i]
                
                if 0 <= card1_x < width and 0 <= card1_y < height:
                    if grid[card1_y, card1_x] == 2:
                        continue
                if 0 <= card2_x < width and 0 <= card2_y < height:
                    if grid[card2_y, card2_x] == 2:
                        continue
            
            new_g = current_g + costs[i]
            
            if new_g < g_score[ny, nx]:
                g_score[ny, nx] = new_g
                f = new_g + _numba_heuristic(nx, ny, gx, gy)
                parent_dir[ny, nx] = i
                heap_size = _numba_heap_push(heap, heap_size, f, float(nx), float(ny))
    
    if not found:
        return np.empty((0, 2), dtype=np.int32), False
    
    # Count path length
    path_len = 0
    px, py = gx, gy
    while not (px == sx and py == sy):
        path_len += 1
        d = parent_dir[py, px]
        if d == -1:
            return np.empty((0, 2), dtype=np.int32), False
        px = px - dx_arr[d]
        py = py - dy_arr[d]
    
    # Build path
    path = np.empty((path_len, 2), dtype=np.int32)
    px, py = gx, gy
    for i in range(path_len - 1, -1, -1):
        path[i, 0] = px + grid_offset_x
        path[i, 1] = py + grid_offset_y
        d = parent_dir[py, px]
        px = px - dx_arr[d]
        py = py - dy_arr[d]
    
    return path, True


class NumbaAStarHelper:
    """Helper class to manage Numba A* grid conversion and caching."""
    
    def __init__(self):
        self.grid = None
        self.grid_offset_x = 0
        self.grid_offset_y = 0
        self._last_explored_count = 0
        self._last_obstacle_count = 0
        self._warmed_up = False
    
    def _warmup(self):
        """Pre-compile Numba functions."""
        if self._warmed_up:
            return
        test_grid = np.ones((10, 10), dtype=np.int8)
        _numba_astar_core(test_grid, 0, 0, 5, 5, 0, 0, True, 100)
        self._warmed_up = True
    
    def update_grid(self, occupancy_grid, obstacle_cells, safety_margin=1):
        """Convert dict-based occupancy grid to numpy array for Numba."""
        if not occupancy_grid:
            self.grid = None
            return
        
        # Check if rebuild needed
        if (len(occupancy_grid) == self._last_explored_count and 
            len(obstacle_cells) == self._last_obstacle_count and
            self.grid is not None):
            return
        
        self._last_explored_count = len(occupancy_grid)
        self._last_obstacle_count = len(obstacle_cells)
        
        all_cells = list(occupancy_grid.keys())
        xs = [c[0] for c in all_cells]
        ys = [c[1] for c in all_cells]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        padding = safety_margin + 2
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        
        self.grid_offset_x = min_x
        self.grid_offset_y = min_y
        
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        
        self.grid = np.zeros((height, width), dtype=np.int8)
        
        for (x, y), val in occupancy_grid.items():
            gx = x - min_x
            gy = y - min_y
            if 0 <= gx < width and 0 <= gy < height:
                self.grid[gy, gx] = val
        
        for (x, y) in obstacle_cells:
            gx = x - min_x
            gy = y - min_y
            if 0 <= gx < width and 0 <= gy < height:
                self.grid[gy, gx] = 2
        
        if safety_margin > 0:
            inflated_mask = np.zeros_like(self.grid, dtype=np.bool_)
            for (x, y) in obstacle_cells:
                gx = x - min_x
                gy = y - min_y
                for dx in range(-safety_margin, safety_margin + 1):
                    for dy in range(-safety_margin, safety_margin + 1):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if self.grid[ny, nx] == 1:
                                inflated_mask[ny, nx] = True
            self.grid[inflated_mask] = 3
    
    def plan_path(self, start_grid, goal_grid, use_inflation=True):
        """Plan path using Numba-accelerated A*."""
        self._warmup()
        
        if self.grid is None:
            return []
        
        path_array, found = _numba_astar_core(
            self.grid,
            start_grid[0], start_grid[1],
            goal_grid[0], goal_grid[1],
            self.grid_offset_x, self.grid_offset_y,
            use_inflation
        )
        
        if not found and use_inflation:
            path_array, found = _numba_astar_core(
                self.grid,
                start_grid[0], start_grid[1],
                goal_grid[0], goal_grid[1],
                self.grid_offset_x, self.grid_offset_y,
                False
            )
        
        if not found or len(path_array) == 0:
            return []
        
        path = [(int(path_array[i, 0]), int(path_array[i, 1])) for i in range(len(path_array))]
        
        if len(path) > 8:
            path = path[::2]
            if path[-1] != goal_grid:
                path.append(goal_grid)
        
        return path


def decompose_grid_to_rectangles(occupancy_grid, max_rects=None):
    """
    Decomposes free space into rectangles and optionally returns only the top N largest.
    """
    if not occupancy_grid:
        return []
    
    # Filter only free cells to find bounds
    free_cells = [k for k, v in occupancy_grid.items() if v == 1]
    if not free_cells:
        return []
        
    xs = [c[0] for c in free_cells]
    ys = [c[1] for c in free_cells]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    
    # Build dense boolean grid (True = Free)
    grid = np.zeros((h, w), dtype=bool)
    for (gx, gy) in free_cells:
        grid[gy - min_y, gx - min_x] = True
            
    # Decompose
    rects = []
    remaining = grid
    
    while True:
        # Find first True cell
        coords = np.argwhere(remaining)
        if len(coords) == 0:
            break
        r, c = coords[0]
        
        # Expand Width
        current_w = 1
        while c + current_w < w and remaining[r, c + current_w]:
            current_w += 1
            
        # Expand Height
        current_h = 1
        while r + current_h < h:
            # Check if whole row segment is free
            if not np.all(remaining[r + current_h, c : c + current_w]):
                break
            current_h += 1
            
        # Store rect (gx, gy, w, h)
        rects.append((c + min_x, r + min_y, current_w, current_h))
        
        # Mark as visited
        remaining[r : r + current_h, c : c + current_w] = False

    # --- NEW LOGIC: Sort and Limit ---
    if max_rects is not None:
        # Sort by Area (Width * Height) in descending order
        rects.sort(key=lambda r: r[2] * r[3], reverse=True)
        return rects[:max_rects]
        
    return rects


class Robot:
    # Class constants for bounded data structures
    MAX_TRAJECTORY_LENGTH = 200  # Rolling window for trajectory
    TRAJECTORY_TRIM_SIZE = 150   # Trim to this size when exceeded
    
    def __init__(self, robot_id, position, color):
        self.id = robot_id
        self.position = position
        self.color = color
        self.lidar_data = []  # Now replaced each scan, not extended
        self.trajectory = []
        
        # Autonomy State
        self.goal = None
        self.path = []  # List of grid waypoints from A*
        self.manual_control = False
        self.mode = 'IDLE'  # IDLE, EXPLORING, RETURNING_HOME
        
        # Blacklist for unreachable goals (grid_pos -> expiration_step)
        self.blacklisted_goals = {}
        
        # Direction tracking for exploration bias
        self.exploration_direction = np.array([1.0, 0.0])  # Initial heading (east)
        self.direction_history = []  # Track recent movement directions
        self.direction_smoothing_window = 10  # Number of samples to average
        
        # Stuck detection
        self.stuck_counter = 0
        self.last_position = np.array(position[:2])
        self.goal_attempts = 0
        self.max_goal_attempts = 3  # Give up after 3 failed attempts to reach a goal
        
        # Global graph for navigation - OPTIMIZED with spatial hashing
        self.home_position = np.array(position[:2])  # Remember start position
        self.global_graph_nodes = [tuple(position[:2])]  # List of (x, y) waypoints
        self.global_graph_edges = set()  # SET for O(1) membership checks (was list)
        self.last_graph_node_idx = 0  # Index of last node added to graph
        self.waypoint_spacing = 3.0  # Meters between waypoints
        
        # Spatial hash for O(1) average nearest-neighbor lookup
        self._node_grid = defaultdict(list)  # (grid_x, grid_y) -> [node_indices]
        self._node_grid_cell_size = self.waypoint_spacing  # Grid cell = waypoint spacing
        # Add initial node to spatial hash
        self._add_node_to_spatial_hash(0, position[:2])
    
    def _get_spatial_hash_cell(self, pos):
        """Get the spatial hash grid cell for a position."""
        return (int(pos[0] / self._node_grid_cell_size),
                int(pos[1] / self._node_grid_cell_size))
    
    def _add_node_to_spatial_hash(self, node_idx, pos):
        """Add a node to the spatial hash."""
        cell = self._get_spatial_hash_cell(pos)
        self._node_grid[cell].append(node_idx)
    
    def _find_nearest_node_fast(self, pos):
        """Find nearest node using spatial hash - O(1) average case."""
        cell = self._get_spatial_hash_cell(pos)
        
        # Check current cell and all 8 neighbors
        min_dist_sq = float('inf')
        nearest_idx = -1
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_cell = (cell[0] + dx, cell[1] + dy)
                for node_idx in self._node_grid.get(check_cell, []):
                    node = self.global_graph_nodes[node_idx]
                    dist_sq = (pos[0] - node[0])**2 + (pos[1] - node[1])**2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        nearest_idx = node_idx
        
        return nearest_idx, np.sqrt(min_dist_sq)

    def update_exploration_direction(self):
        """
        Update the robot's exploration direction based on recent movement.
        Uses a low-pass filter over recent trajectory points (like GBPlanner).
        """
        if len(self.trajectory) < 2:
            return
        
        # Get recent trajectory segment
        recent_points = self.trajectory[-min(20, len(self.trajectory)):]
        
        if len(recent_points) >= 2:
            # Calculate direction from older point to newer point
            start = np.array(recent_points[0])
            end = np.array(recent_points[-1])
            direction = end - start
            
            norm = np.linalg.norm(direction)
            if norm > 0.5:  # Only update if we've moved meaningfully
                direction = direction / norm
                
                # Smooth with existing direction (low-pass filter)
                alpha = 0.3  # Smoothing factor
                self.exploration_direction = (
                    alpha * direction + 
                    (1 - alpha) * self.exploration_direction
                )
                # Renormalize
                self.exploration_direction /= np.linalg.norm(self.exploration_direction)

    def get_heading_vector(self):
        """Get the robot's current facing direction from physics."""
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        return np.array([np.cos(yaw), np.sin(yaw)])

    def check_if_stuck(self, threshold=0.3, stuck_limit=200):
        """
        Check if robot is stuck (not moving despite having a goal).
        Increased limit to 200 to prevent false positives on slow turns.
        """
        pos, _ = p.getBasePositionAndOrientation(self.id)
        current_pos = np.array([pos[0], pos[1]])
        
        distance_moved = np.linalg.norm(current_pos - self.last_position)
        
        if distance_moved < threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_position = current_pos
        
        return self.stuck_counter > stuck_limit

    def reset_stuck_state(self):
        """Reset stuck detection when getting a new goal."""
        self.stuck_counter = 0
        pos, _ = p.getBasePositionAndOrientation(self.id)
        self.last_position = np.array([pos[0], pos[1]])
        
    def cleanup_blacklist(self, current_step):
        """Remove expired entries from the goal blacklist."""
        expired = [pos for pos, expiry in self.blacklisted_goals.items() if current_step >= expiry]
        for pos in expired:
            del self.blacklisted_goals[pos]

    def update_global_graph(self):
        """
        Add current position to global graph if far enough from existing nodes.
        Creates edges to connect new nodes to the graph.
        OPTIMIZED: Uses spatial hashing for O(1) nearest-neighbor lookup.
        """
        pos, _ = p.getBasePositionAndOrientation(self.id)
        current_pos = (pos[0], pos[1])
        
        # Use spatial hash for O(1) average nearest-neighbor lookup
        nearest_node_idx, min_dist = self._find_nearest_node_fast(current_pos)
        
        # Only add new node if far enough from all existing nodes
        if min_dist >= self.waypoint_spacing:
            new_node_idx = len(self.global_graph_nodes)
            self.global_graph_nodes.append(current_pos)
            
            # Add to spatial hash
            self._add_node_to_spatial_hash(new_node_idx, current_pos)
            
            # Connect to the last node we were at (maintains path continuity)
            self.global_graph_edges.add((self.last_graph_node_idx, new_node_idx))
            
            # Also connect to nearest node if different (creates shortcuts)
            if nearest_node_idx != -1 and nearest_node_idx != self.last_graph_node_idx:
                # Normalize edge representation (smaller index first) for consistent hashing
                edge = (min(nearest_node_idx, new_node_idx), max(nearest_node_idx, new_node_idx))
                self.global_graph_edges.add(edge)  # O(1) with set
            
            self.last_graph_node_idx = new_node_idx
        else:
            # Update last node to nearest if we're close to an existing node
            self.last_graph_node_idx = nearest_node_idx

    def plan_path_on_global_graph(self, target_pos):
        """
        Plan a path on the global graph to reach target_pos.
        Uses Dijkstra's algorithm on the graph.
        OPTIMIZED: Uses spatial hashing for nearest-neighbor lookups.
        
        Returns: List of (x, y) waypoints to follow
        """
        if len(self.global_graph_nodes) < 2:
            return [target_pos]
        
        # Find nearest node to current position using spatial hash
        pos, _ = p.getBasePositionAndOrientation(self.id)
        current_pos = (pos[0], pos[1])
        start_node_idx, _ = self._find_nearest_node_fast(current_pos)
        
        # Find nearest node to target using spatial hash
        goal_node_idx, _ = self._find_nearest_node_fast(target_pos)
        
        # Build adjacency list
        adjacency = {i: [] for i in range(len(self.global_graph_nodes))}
        for edge in self.global_graph_edges:
            n1, n2 = edge
            # Calculate edge weight (distance)
            p1 = self.global_graph_nodes[n1]
            p2 = self.global_graph_nodes[n2]
            weight = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            adjacency[n1].append((n2, weight))
            adjacency[n2].append((n1, weight))  # Undirected graph
        
        # Dijkstra's algorithm
        distances = {i: float('inf') for i in range(len(self.global_graph_nodes))}
        distances[start_node_idx] = 0
        came_from = {start_node_idx: None}
        
        frontier = [(0, start_node_idx)]
        visited = set()
        
        while frontier:
            current_dist, current_node = heapq.heappop(frontier)
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            if current_node == goal_node_idx:
                break
            
            for neighbor, weight in adjacency[current_node]:
                if neighbor in visited:
                    continue
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    came_from[neighbor] = current_node
                    heapq.heappush(frontier, (new_dist, neighbor))
        
        # Reconstruct path
        if goal_node_idx not in came_from:
            return [target_pos]  # No path found, just go directly
        
        path = []
        current = goal_node_idx
        while current is not None:
            path.append(self.global_graph_nodes[current])
            current = came_from.get(current)
        path.reverse()
        
        # Add final target position if not at a node
        if path and np.sqrt((path[-1][0] - target_pos[0])**2 + (path[-1][1] - target_pos[1])**2) > 0.5:
            path.append(target_pos)
        
        return path

    def get_lidar_scan(self, num_rays=360, max_range=10):
        """Simulate lidar by casting rays in 360 degrees - OPTIMIZED with numpy"""
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        # Mount Lidar Higher to avoid self-collision
        lidar_height_offset = 0.3 
        lidar_z = pos[2] + lidar_height_offset

        # OPTIMIZED: Use numpy for vectorized angle calculations
        angles = yaw + np.linspace(0, 2.0 * np.pi, num_rays, endpoint=False)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        
        # Build ray arrays
        ray_from = [[pos[0], pos[1], lidar_z]] * num_rays
        ray_to = [
            [pos[0] + max_range * cos_angles[i],
             pos[1] + max_range * sin_angles[i],
             lidar_z]
            for i in range(num_rays)
        ]

        results = p.rayTestBatch(ray_from, ray_to)

        scan_points = []
        for i, result in enumerate(results):
            hit_object_id = result[0]
            hit_fraction = result[2]
            
            # Check if hit something
            if hit_fraction < 1.0 and hit_object_id != self.id:
                hit_position = result[3]
                # Store (x, y, is_hit=True)
                scan_points.append((hit_position[0], hit_position[1], True))
            else:
                # Ray maxed out - hit nothing
                # Store (x, y, is_hit=False) at max range
                scan_points.append((ray_to[i][0], ray_to[i][1], False))

        # OPTIMIZATION: Replace lidar_data entirely instead of extending
        # The occupancy grid only uses the current scan anyway
        self.lidar_data = scan_points
        
        # OPTIMIZATION: Bound trajectory to prevent unbounded growth
        self.trajectory.append((pos[0], pos[1]))
        if len(self.trajectory) > Robot.MAX_TRAJECTORY_LENGTH:
            self.trajectory = self.trajectory[-Robot.TRAJECTORY_TRIM_SIZE:]
        
        # NEW: Update exploration direction after recording trajectory
        self.update_exploration_direction()
        
        return scan_points

    def move(self, linear_vel, angular_vel):
        """Move robot using physics velocity control."""
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        vx = linear_vel * np.cos(yaw)
        vy = linear_vel * np.sin(yaw)

        p.resetBaseVelocity(self.id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, angular_vel])

    def follow_path(self, mapper):
        """
        Follow the A* path. 
        If path is empty but goal exists, drive to goal.
        """
        if not self.path and self.goal is None:
            return 0.0, 0.0

        target_pos = self.goal
        
        # If we have a path, aim for the next waypoint
        if self.path:
            # Get next waypoint (grid coords)
            next_wp = self.path[0]
            # Convert to world coords
            wx, wy = mapper.grid_to_world(next_wp[0], next_wp[1])
            target_pos = (wx, wy)
            
            # Check if we reached this waypoint
            pos, _ = p.getBasePositionAndOrientation(self.id)
            dist = np.sqrt((wx - pos[0])**2 + (wy - pos[1])**2)
            
            # Waypoint acceptance radius
            if dist < 0.6:
                self.path.pop(0)  # Waypoint reached
                if self.path:  # If more points, recurse to get next target
                    return self.follow_path(mapper)

        # Drive to target_pos (Pure Pursuit / Proportional Control)
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        dx = target_pos[0] - pos[0]
        dy = target_pos[1] - pos[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Stop if reached final goal
        if distance < 0.5 and not self.path:
            self.goal = None 
            return 0.0, 0.0

        desired_angle = np.arctan2(dy, dx)
        angle_diff = desired_angle - yaw
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        # Control Limits (Increased for faster exploration)
        linear_vel = min(8.0, distance * 2.0)  # Max 8.0 m/s (was 4.0)
        angular_vel = np.clip(angle_diff * 4.0, -4.0, 4.0)

        # Slow down if turning sharply (reduced penalty for faster turns)
        if abs(angle_diff) > 0.5:
            linear_vel *= 0.3  # 30% speed when turning (was 10%)

        return linear_vel, angular_vel


class CoverageMapper:
    """Multi-robot coverage mapper for procedurally generated environments"""

    def __init__(self, use_gui=True, maze_size=(10, 10), cell_size=2.0, env_seed=None, env_type='maze', num_robots=3, show_partitions=False):
        self.num_robots = num_robots
        self.show_partitions = show_partitions  # Flag for rectangular decomposition
        
        # Generate the maze grid first
        map_generator = MapGenerator(
            maze_size=maze_size,
            seed=env_seed
        )
        maze_grid, entrance_cell = map_generator.generate_maze(env_type=env_type)

        # Create PyBullet environment from the grid
        self.env = PybulletRenderer(
            maze_grid=maze_grid,
            entrance_cell=entrance_cell,
            cell_size=cell_size,
            wall_height=2.5,
            gui=use_gui
        )

        self.env.build_walls()
        self.physics_client = self.env.physics_client

        self.robots = []
        self.create_robots()

        self.grid_resolution = 0.5
        self.occupancy_grid = {}  # 1=Free, 2=Obstacle
        self.explored_cells = set()
        self.obstacle_cells = set()

        block_physical_size = self.env.cell_size / 2.0
        half_block = block_physical_size / 2.0
        
        maze_world_width = self.env.maze_grid.shape[1] * block_physical_size
        maze_world_height = self.env.maze_grid.shape[0] * block_physical_size
        
        self.map_bounds = {
            'x_min': -half_block,
            'x_max': maze_world_width - half_block,
            'y_min': -half_block,
            'y_max': maze_world_height - half_block
        }

        self.scale_factor = block_physical_size / self.grid_resolution
        ground_truth_zeros = np.sum(self.env.maze_grid == 0)
        self.total_free_cells = ground_truth_zeros * (self.scale_factor ** 2)
        
        self.coverage_history = []
        self.realtime_fig = None
        self.realtime_axes = None
        self.claimed_goals = {}
        
        # Zoom/View Control
        self.current_xlim = None
        self.current_ylim = None
        
        # Utility function weights
        self.direction_bias_weight = 2.5  # How much to reward forward motion
        self.size_weight = 0.3            # Weight for frontier size
        self.distance_weight = 1.0        # Weight for distance cost
        
        # Coordination / Crowding weights (NEW)
        self.crowding_penalty_weight = 100.0  # Heavy penalty for targeting same area
        self.crowding_radius = 8.0           # Radius (meters) to discourage other robots
        
        # Return-to-home settings
        self.return_home_coverage = 100.0  # Trigger return at this coverage %
        self.returning_home = False       # Global state: are we returning home?
        self.robots_home = set()          # Track which robots have arrived home
        
        # Safety margin
        self.safety_margin = 1
        
        # Numba A* helper
        self._numba_astar = NumbaAStarHelper()
        
        # Performance optimization caches
        self._cached_coverage = 0.0
        self._coverage_cache_valid = False
        self._coverage_explored_count = 0 
        self._cached_frontiers = None
        
        # INCREMENTAL COVERAGE TRACKING
        self._incremental_valid_count = 0
        self._processed_explored_cells = set()
        
        # Pre-compute maze lookup
        self._maze_block_size = self.env.cell_size / 2.0
        self._maze_half_block = self._maze_block_size / 2.0
        self._maze_inv_block_size = 1.0 / self._maze_block_size
        
        # Pre-computed grid bounds
        self._grid_bounds = None 
        
        # INCREMENTAL FRONTIER DETECTION
        self._frontier_candidates = set()
        
        # Numba-accelerated occupancy grid
        self._numba_occupancy = NumbaOccupancyGrid(self.map_bounds, self.grid_resolution)
        
        # Logging
        self.logger = None
        self.env_config = {
            'maze_size': maze_size,
            'cell_size': cell_size,
            'env_seed': env_seed,
            'env_type': env_type,
            'num_robots': num_robots,
        }

    def create_robots(self):
        spawn_pos = self.env.get_spawn_position()
        
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
        
        start_positions = []
        spacing = 1.5
        for i in range(self.num_robots):
            offset = (i - (self.num_robots - 1) / 2) * spacing
            start_positions.append([spawn_pos[0] + offset, spawn_pos[1], 0.25])
        
        colors = [all_colors[i % len(all_colors)] for i in range(self.num_robots)]

        for i, (pos, color) in enumerate(zip(start_positions, colors)):
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.25)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.25, rgbaColor=color)

            robot_id = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=pos
            )

            p.changeDynamics(robot_id, -1, localInertiaDiagonal=[0, 0, 1])
            p.changeDynamics(robot_id, -1, lateralFriction=1.0, spinningFriction=0.1, rollingFriction=0.0)

            robot = Robot(robot_id, pos, color)
            self.robots.append(robot)

    def world_to_grid(self, x, y):
        grid_x = int((x - self.map_bounds['x_min']) / self.grid_resolution)
        grid_y = int((y - self.map_bounds['y_min']) / self.grid_resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        x = self.map_bounds['x_min'] + (grid_x + 0.5) * self.grid_resolution
        y = self.map_bounds['y_min'] + (grid_y + 0.5) * self.grid_resolution
        return (x, y)

    def update_occupancy_grid(self, robot):
        """Update occupancy grid from robot LIDAR data - NUMBA OPTIMIZED."""
        if not robot.lidar_data:
            return

        pos, _ = p.getBasePositionAndOrientation(robot.id)
        
        old_explored_count = len(self.explored_cells)
        
        self._numba_occupancy.update_from_lidar(
            (pos[0], pos[1]),
            robot.lidar_data,
            self.occupancy_grid,
            self.explored_cells,
            self.obstacle_cells
        )
        
        if len(self.explored_cells) > old_explored_count:
            robot_cell = self.world_to_grid(pos[0], pos[1])
            lidar_range_cells = int(15.0 / self.grid_resolution) + 1 
            
            for dx in range(-lidar_range_cells, lidar_range_cells + 1):
                for dy in range(-lidar_range_cells, lidar_range_cells + 1):
                    cell = (robot_cell[0] + dx, robot_cell[1] + dy)
                    if cell in self.explored_cells:
                        self._frontier_candidates.add(cell)

    def detect_frontiers(self, use_cache=False):
        """
        Detect and cluster frontier cells - OPTIMIZED with candidate tracking.
        """
        if use_cache and self._cached_frontiers is not None and len(self._cached_frontiers) > 0:
            return self._cached_frontiers
        
        if self._grid_bounds is None:
            self._grid_bounds = {
                'min_gx': int((self.map_bounds['x_min'] - self.map_bounds['x_min']) / self.grid_resolution),
                'max_gx': int((self.map_bounds['x_max'] - self.map_bounds['x_min']) / self.grid_resolution),
                'min_gy': int((self.map_bounds['y_min'] - self.map_bounds['y_min']) / self.grid_resolution),
                'max_gy': int((self.map_bounds['y_max'] - self.map_bounds['y_min']) / self.grid_resolution),
            }
        
        gb = self._grid_bounds
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        if self._frontier_candidates:
            cells_to_check = self._frontier_candidates
        else:
            cells_to_check = self.explored_cells
        
        frontiers = set()
        occupancy = self.occupancy_grid 
        non_frontier_candidates = set()
        
        for cell in cells_to_check:
            if cell not in self.explored_cells:
                non_frontier_candidates.add(cell)
                continue
            if occupancy.get(cell) != 1:
                non_frontier_candidates.add(cell)
                continue
                
            x, y = cell
            is_frontier = False
            has_all_neighbors_known = True
            
            for dx, dy in neighbor_offsets:
                nx, ny = x + dx, y + dy
                if not (gb['min_gx'] <= nx <= gb['max_gx'] and gb['min_gy'] <= ny <= gb['max_gy']):
                    continue
                if (nx, ny) not in occupancy:
                    is_frontier = True
                    has_all_neighbors_known = False
                    break
            
            if is_frontier:
                frontiers.add(cell)
            elif has_all_neighbors_known:
                non_frontier_candidates.add(cell)
        
        self._frontier_candidates -= non_frontier_candidates
        
        if not frontiers:
            self._cached_frontiers = []
            return []

        from collections import deque
        frontier_list = frontiers
        visited = set()
        clusters = []

        for f in frontier_list:
            if f in visited:
                continue
            cluster = []
            queue = deque([f])
            visited.add(f)
            
            while queue:
                current = queue.popleft()
                cluster.append(current)
                cx, cy = current
                for dx, dy in neighbor_offsets:
                    nbr = (cx + dx, cy + dy)
                    if nbr in frontier_list and nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            
            if len(cluster) > 6:
                avg_x = sum(c[0] for c in cluster) / len(cluster)
                avg_y = sum(c[1] for c in cluster) / len(cluster)
                best_point = min(cluster, key=lambda p: (p[0] - avg_x)**2 + (p[1] - avg_y)**2)
                wx, wy = self.grid_to_world(best_point[0], best_point[1])
                clusters.append({'pos': (wx, wy), 'grid_pos': best_point, 'size': len(cluster)})

        self._cached_frontiers = clusters
        return clusters

    def plan_path_astar(self, start_grid, goal_grid):
        """
        A* Pathfinding on the occupancy grid.
        Uses Numba JIT compilation for 6-18x speedup.
        """
        self._numba_astar.update_grid(
            self.occupancy_grid,
            self.obstacle_cells,
            self.safety_margin
        )
        path = self._numba_astar.plan_path(start_grid, goal_grid, use_inflation=True)
        return path
    
    def calculate_utility(self, robot, frontier):
        """
        Utility Function with Direction Bias and Volumetric Gain
        """
        pos, orn = p.getBasePositionAndOrientation(robot.id)
        robot_pos = np.array([pos[0], pos[1]])
        frontier_pos = np.array(frontier['pos'])
        
        dist = np.linalg.norm(frontier_pos - robot_pos)
        distance_cost = dist * self.distance_weight
        size_gain = frontier['size'] * self.size_weight
        
        to_frontier = frontier_pos - robot_pos
        to_frontier_norm = np.linalg.norm(to_frontier)
        
        if to_frontier_norm > 0.1:
            to_frontier_unit = to_frontier / to_frontier_norm
            alignment = np.dot(robot.exploration_direction, to_frontier_unit)
            alignment_normalized = (alignment + 1.0) / 2.0
            direction_bonus = alignment_normalized * self.direction_bias_weight
        else:
            alignment = 0.0
            direction_bonus = 0.0
        
        utility = size_gain - distance_cost + direction_bonus
        return utility, {}

    def assign_global_goals(self, step):
        """
        Market-based Coordination loop (Improved: Global Best-First).
        """
        frontiers = self.detect_frontiers()
        if not frontiers:
            return

        active_robots = [r for r in self.robots if not r.manual_control]
        for robot in active_robots:
             robot.mode = 'GLOBAL_RELOCATE'
             robot.cleanup_blacklist(step)

        current_round_goals = []
        unassigned_robots = active_robots.copy()

        while unassigned_robots:
            best_global_utility = -float('inf')
            best_pair = None 

            for robot in unassigned_robots:
                current_goal_pos = None
                if robot.goal:
                    current_goal_pos = np.array(robot.goal)

                for f in frontiers:
                    if f['grid_pos'] in robot.blacklisted_goals:
                        continue
                    
                    target_pos = np.array(f['pos'])
                    util, _ = self.calculate_utility(robot, f)
                    
                    crowding_penalty = 0.0
                    for assigned_goal in current_round_goals:
                        dist_to_assigned = np.linalg.norm(target_pos - np.array(assigned_goal))
                        if dist_to_assigned < self.crowding_radius:
                            factor = 1.0 - (dist_to_assigned / self.crowding_radius)
                            crowding_penalty += factor * self.crowding_penalty_weight
                    
                    persistence_bias = 5.0
                    if current_goal_pos is not None:
                        dist_to_current = np.linalg.norm(target_pos - current_goal_pos)
                        if dist_to_current < 2.0:
                            util += persistence_bias

                    final_utility = util - crowding_penalty

                    if final_utility > best_global_utility:
                        best_global_utility = final_utility
                        best_pair = (robot, f)

            if best_pair:
                winner_robot, winning_frontier = best_pair
                target_pos = winning_frontier['pos']
                
                old_goal = winner_robot.goal
                winner_robot.goal = target_pos
                
                should_replan = True
                if old_goal is not None:
                    dist_change = np.linalg.norm(np.array(target_pos) - np.array(old_goal))
                    if dist_change < 2.0:
                        should_replan = False
                
                if should_replan:
                    winner_robot.reset_stuck_state()
                    winner_robot.goal_attempts = 0
                    
                    pos, _ = p.getBasePositionAndOrientation(winner_robot.id)
                    start_grid = self.world_to_grid(pos[0], pos[1])
                    path = self.plan_path_astar(start_grid, winning_frontier['grid_pos'])
                    
                    if path:
                        winner_robot.path = path
                        print(f"Robot {winner_robot.id}: Assigned goal (Utility: {best_global_utility:.1f})")
                    else:
                        winner_robot.goal = None
                        winner_robot.blacklisted_goals[winning_frontier['grid_pos']] = step + 500
                        print(f"Robot {winner_robot.id}: Path failed to {winning_frontier['grid_pos']}, blacklisted.")

                if winner_robot.goal is not None:
                    current_round_goals.append(target_pos)
                    unassigned_robots.remove(winner_robot)
                
            else:
                break
        
        for loser_robot in unassigned_robots:
            if loser_robot.goal is not None:
                loser_robot.goal = None
                loser_robot.path = []

    def exploration_logic(self, robot, step):
        if robot.manual_control:
            l, a = robot.follow_path(self) 
            robot.move(l, a)
            return
        
        if robot.mode == 'HOME':
            robot.move(0.0, 0.0)
            return

        if robot.goal:
            if robot.check_if_stuck(threshold=0.2, stuck_limit=200):
                print(f"Robot {robot.id} is stuck! Abandoning goal and replanning...")
                robot.goal = None
                robot.path = []
                robot.reset_stuck_state()
                robot.goal_attempts += 1
                
                if robot.goal_attempts > robot.max_goal_attempts:
                    robot.move(0.0, 2.0) 
                    robot.goal_attempts = 0
                return
            
            l, a = robot.follow_path(self)
            robot.move(l, a)
        else:
            if robot.mode == 'RETURNING_HOME':
                self.plan_return_path_for_robot(robot)
                if robot.goal:
                    l, a = robot.follow_path(self)
                    robot.move(l, a)
                else:
                    robot.move(0.0, 0.5)
            else:
                robot.move(0.0, 0.5)

    def plan_return_path_for_robot(self, robot):
        """Plan a path home for a single robot using A*."""
        pos, _ = p.getBasePositionAndOrientation(robot.id)
        start_grid = self.world_to_grid(pos[0], pos[1])
        home_grid = self.world_to_grid(robot.home_position[0], robot.home_position[1])
        
        robot.path = self.plan_path_astar(start_grid, home_grid)
        if robot.path:
            robot.goal = tuple(robot.home_position)
            robot.reset_stuck_state()

    def trigger_return_home(self):
        """Trigger all robots to return to their home positions using A*."""
        for robot in self.robots:
            robot.mode = 'RETURNING_HOME'
            self.plan_return_path_for_robot(robot)
            print(f"Robot {robot.id}: Returning home with {len(robot.path)} waypoints")

    def calculate_coverage(self, use_cache=True):
        """
        Calculate coverage percentage - INCREMENTAL VERSION.
        """
        if self.total_free_cells == 0:
            return 0.0

        current_count = len(self.explored_cells)
        if use_cache and current_count == len(self._processed_explored_cells):
            return self._cached_coverage

        new_cells = self.explored_cells - self._processed_explored_cells
        
        if not new_cells:
            return self._cached_coverage
        
        maze_h, maze_w = self.env.maze_grid.shape
        maze_grid = self.env.maze_grid
        x_min = self.map_bounds['x_min']
        y_min = self.map_bounds['y_min']
        grid_res = self.grid_resolution
        half_block = self._maze_half_block
        inv_block_size = self._maze_inv_block_size
        
        new_valid_count = 0
        for cell in new_cells:
            wx = x_min + (cell[0] + 0.5) * grid_res
            wy = y_min + (cell[1] + 0.5) * grid_res
            
            mx = int((wx + half_block) * inv_block_size)
            my = int((wy + half_block) * inv_block_size)
            
            if 0 <= mx < maze_w and 0 <= my < maze_h:
                if maze_grid[my, mx] == 0:
                    new_valid_count += 1
        
        self._incremental_valid_count += new_valid_count
        self._processed_explored_cells.update(new_cells)
        
        coverage_percent = min(100.0, (self._incremental_valid_count / self.total_free_cells) * 100)
        
        self._cached_coverage = coverage_percent
        self._coverage_cache_valid = True
        self._coverage_explored_count = current_count
        
        return coverage_percent
    
    def invalidate_coverage_cache(self):
        self._coverage_cache_valid = False

    def setup_realtime_visualization(self):
        plt.ion()
        self.realtime_fig = plt.figure(figsize=(18, 14))
        gs = self.realtime_fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.25, wspace=0.2)

        self.realtime_axes = {
            'grid': self.realtime_fig.add_subplot(gs[0, 0]),
            'frontier': self.realtime_fig.add_subplot(gs[0, 1]),
            'coverage': self.realtime_fig.add_subplot(gs[1, :])
        }

        title = 'Multi-Robot Coverage Mapping\n(Scroll to Zoom, Click for manual control, "P" to toggle decomposition)'
        self.realtime_fig.suptitle(title, fontsize=14, fontweight='bold')
        self.realtime_fig.canvas.mpl_connect('button_press_event', self.on_map_click)
        self.realtime_fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.realtime_fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.show(block=False)

    def on_key_press(self, event):
        if event.key == 'p' or event.key == 'P':
            self.show_partitions = not self.show_partitions
            print(f"\n[Viz] Rectangular Decomposition: {'ON' if self.show_partitions else 'OFF'}")

    def on_scroll(self, event):
        """Handle mouse scroll for zooming in/out."""
        if event.inaxes not in [self.realtime_axes['grid'], self.realtime_axes['frontier']]:
            return

        cur_xlim = event.inaxes.get_xlim()
        cur_ylim = event.inaxes.get_ylim()
        
        xdata = event.xdata 
        ydata = event.ydata
        
        if xdata is None or ydata is None:
            return

        base_scale = 1.2
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.current_xlim = [xdata - new_width * (1 - relx), xdata + new_width * relx]
        self.current_ylim = [ydata - new_height * (1 - rely), ydata + new_height * rely]

    def on_map_click(self, event):
        if event.inaxes == self.realtime_axes['grid']:
            x, y = event.xdata, event.ydata
            if (self.map_bounds['x_min'] <= x <= self.map_bounds['x_max'] and
                self.map_bounds['y_min'] <= y <= self.map_bounds['y_max']):
                
                self.robots[0].goal = (x, y)
                self.robots[0].manual_control = True
                
                pos, _ = p.getBasePositionAndOrientation(self.robots[0].id)
                start = self.world_to_grid(pos[0], pos[1])
                end = self.world_to_grid(x, y)
                
                print(f"Planning path to ({x:.2f}, {y:.2f})...")
                self.robots[0].path = self.plan_path_astar(start, end)
                print(f"Path found: {len(self.robots[0].path)} steps")

    def update_realtime_visualization(self, step):
        if self.realtime_fig is None:
            return

        for ax in self.realtime_axes.values():
            ax.clear()

        # Update occupancy grid
        ax_grid = self.realtime_axes['grid']
        grid_x = int((self.map_bounds['x_max'] - self.map_bounds['x_min']) / self.grid_resolution)
        grid_y = int((self.map_bounds['y_max'] - self.map_bounds['y_min']) / self.grid_resolution)
        grid_image = np.ones((grid_y, grid_x, 3)) * 0.7

        for cell, value in self.occupancy_grid.items():
            gx, gy = cell
            if 0 <= gx < grid_x and 0 <= gy < grid_y:
                if value == 1:
                    grid_image[gy, gx] = [1, 1, 1]
                elif value == 2:
                    grid_image[gy, gx] = [0, 0, 0]

        extent = [self.map_bounds['x_min'], self.map_bounds['x_max'],
                 self.map_bounds['y_min'], self.map_bounds['y_max']]
        ax_grid.imshow(grid_image, origin='lower', extent=extent, interpolation='nearest')

        # === RECTANGULAR DECOMPOSITION VISUALIZATION ===
        if self.show_partitions:
            # Pass max_rects=5 here to limit the visualization
            rects = decompose_grid_to_rectangles(self.occupancy_grid, max_rects=5)
            
            for r in rects:
                gx, gy, w, h = r
                # ... (rest of the drawing logic stays the same)
                # Convert to world coordinates
                wx, wy = self.grid_to_world(gx, gy)
                # Correction: grid_to_world returns center. Rect needs bottom-left.
                # Since w, h are in grid cells, convert size to meters
                rect_w = w * self.grid_resolution
                rect_h = h * self.grid_resolution
                # grid_to_world(gx, gy) returns center of (gx, gy)
                # Bottom-left of (gx, gy) is center - res/2
                rect_x = wx - self.grid_resolution/2
                rect_y = wy - self.grid_resolution/2
                
                # Random pastel color
                color = np.random.rand(3) * 0.5 + 0.5
                rect_patch = patches.Rectangle(
                    (rect_x, rect_y), rect_w, rect_h,
                    linewidth=1, edgecolor='black', facecolor=(*color, 0.3)
                )
                ax_grid.add_patch(rect_patch)


        color_names = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple',
                       'gray', 'pink', 'darkgreen', 'brown', 'navy', 'lime', 'salmon', 'teal']
        
        for i, robot in enumerate(self.robots):
            color = color_names[i % len(color_names)]
            
            for edge in robot.global_graph_edges:
                n1, n2 = edge
                if n1 < len(robot.global_graph_nodes) and n2 < len(robot.global_graph_nodes):
                    p1 = robot.global_graph_nodes[n1]
                    p2 = robot.global_graph_nodes[n2]
                    ax_grid.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                               c=color, linewidth=1, alpha=0.3, linestyle='-')
            
            if robot.global_graph_nodes:
                nodes_arr = np.array(robot.global_graph_nodes)
                ax_grid.scatter(nodes_arr[:, 0], nodes_arr[:, 1], 
                              c=color, s=15, alpha=0.5, marker='o', zorder=3)
                
                ax_grid.scatter(robot.home_position[0], robot.home_position[1],
                              c=color, s=100, marker='s', edgecolors='white', 
                              linewidths=2, zorder=4, label=f'R{i} Home' if i == 0 else '')
        
        for i, robot in enumerate(self.robots):
            color = color_names[i % len(color_names)]
            
            if robot.trajectory:
                traj = np.array(robot.trajectory)
                ax_grid.plot(traj[:, 0], traj[:, 1], c=color, linewidth=1.5, alpha=0.6)
            
            if robot.path:
                path_world = [self.grid_to_world(p[0], p[1]) for p in robot.path]
                path_arr = np.array(path_world)
                ax_grid.plot(path_arr[:, 0], path_arr[:, 1], c=color, linestyle=':', linewidth=2)

            pos, _ = p.getBasePositionAndOrientation(robot.id)
            ax_grid.scatter(pos[0], pos[1], c=color, s=100, marker='^',
                          edgecolors='black', linewidths=1.5, zorder=5)
            
            arrow_len = 1.5
            ax_grid.arrow(pos[0], pos[1], 
                         robot.exploration_direction[0] * arrow_len,
                         robot.exploration_direction[1] * arrow_len,
                         head_width=0.3, head_length=0.2, fc=color, ec='black',
                         alpha=0.7, zorder=6)

            if robot.goal is not None:
                ax_grid.scatter(robot.goal[0], robot.goal[1], c=color, s=150,
                              marker='X', edgecolors='white', linewidths=2, zorder=6)

        if self.current_xlim is None or self.current_ylim is None:
            bounds_margin = 5
            self.current_xlim = [self.map_bounds['x_min'] - bounds_margin, self.map_bounds['x_max'] + bounds_margin]
            self.current_ylim = [self.map_bounds['y_min'] - bounds_margin, self.map_bounds['y_max'] + bounds_margin]

        ax_grid.set_xlim(self.current_xlim)
        ax_grid.set_ylim(self.current_ylim)
        
        ax_grid.set_aspect('equal')
        ax_grid.grid(True, alpha=0.3)
        
        status = "RETURNING HOME" if self.returning_home else "EXPLORING"
        decomp_status = " | [P]artitions: ON" if self.show_partitions else ""
        ax_grid.set_title(f'Occupancy Grid + Global Graph | Status: {status}{decomp_status}')

        coverage = self.calculate_coverage()
        total_nodes = sum(len(r.global_graph_nodes) for r in self.robots)
        ax_grid.text(0.02, 0.98, f'Coverage: {coverage:.1f}%\nGraph nodes: {total_nodes}',
                    transform=ax_grid.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        ax_frontier = self.realtime_axes['frontier']
        frontiers_data = self.detect_frontiers()
        
        explored_points = []
        for cell in self.explored_cells:
            if self.occupancy_grid.get(cell) == 1:
                x, y = self.grid_to_world(cell[0], cell[1])
                explored_points.append([x, y])

        if explored_points:
            explored_array = np.array(explored_points)
            ax_frontier.scatter(explored_array[:, 0], explored_array[:, 1],
                              c='lightblue', s=3, alpha=0.4, marker='s')

        obstacle_points = []
        for cell in self.obstacle_cells:
            x, y = self.grid_to_world(cell[0], cell[1])
            obstacle_points.append([x, y])

        if obstacle_points:
            obstacle_array = np.array(obstacle_points)
            ax_frontier.scatter(obstacle_array[:, 0], obstacle_array[:, 1],
                              c='black', s=3, marker='s')

        if frontiers_data:
            frontier_points = [f['pos'] for f in frontiers_data]
            frontier_array = np.array(frontier_points)
            ax_frontier.scatter(frontier_array[:, 0], frontier_array[:, 1],
                              c='yellow', s=50, marker='o', edgecolors='red',
                              linewidths=2, label='Frontier Targets', zorder=10)

        for i, robot in enumerate(self.robots):
            color = color_names[i % len(color_names)]
            pos, _ = p.getBasePositionAndOrientation(robot.id)
            ax_frontier.scatter(pos[0], pos[1], c=color, s=150, marker='^',
                              edgecolors='black', linewidths=2, zorder=6)

        ax_frontier.set_xlim(self.current_xlim)
        ax_frontier.set_ylim(self.current_ylim)
        
        ax_frontier.set_aspect('equal')
        ax_frontier.grid(True, alpha=0.3)
        ax_frontier.set_title(f'Frontier Detection\n({len(frontiers_data)} targets)')
        ax_frontier.set_xlabel('X (meters)')
        ax_frontier.set_ylabel('Y (meters)')

        ax_coverage = self.realtime_axes['coverage']
        if self.coverage_history:
            steps, coverage_values = zip(*self.coverage_history)
            ax_coverage.plot(steps, coverage_values, linewidth=2, color='blue')
            ax_coverage.fill_between(steps, coverage_values, alpha=0.3, color='blue')
            ax_coverage.axhline(y=coverage, color='red', linestyle='--', alpha=0.5)

        ax_coverage.set_xlabel('Simulation Step')
        ax_coverage.set_ylabel('Coverage (%)')
        ax_coverage.set_title('Coverage Progress (Direction Bias Enabled)')
        ax_coverage.grid(True, alpha=0.3)
        ax_coverage.set_ylim(0, 100)
        ax_coverage.set_xlim(0, max(2000, step))

        self.realtime_fig.canvas.draw()
        self.realtime_fig.canvas.flush_events()
        plt.pause(0.001)

    def run_simulation(self, steps=5000, scan_interval=10, use_gui=True, realtime_viz=True, 
                       viz_update_interval=50, viz_mode='realtime', log_path='./logs'):
        """
        Run the simulation with configurable visualization and logging.
        """
        print("Starting multi-robot coverage mapping simulation...")
        print("*** IMPROVED LOGIC: SMART SCHEDULER + BLACKLISTING ***")
        print("*** PERFORMANCE MODE: INCREMENTAL FRONTIERS + OCTILE HEURISTIC ***")
        print("*** DIRECTION BIAS + CROWDING PENALTY ENABLED ***")
        print(f"  - Direction weight: {self.direction_bias_weight}")
        print(f"  - Crowding Penalty: {self.crowding_penalty_weight} (Radius: {self.crowding_radius}m)")
        
        if steps is None:
            print("Running unlimited steps (press Ctrl+C to stop)...")
        else:
            print(f"Running for {steps} steps...")

        do_realtime = viz_mode in ['realtime', 'both']
        do_logging = viz_mode in ['logging', 'both']
        
        fast_mode = viz_mode == 'logging' and not use_gui
        if fast_mode:
            print("*** FAST MODE ENABLED - Maximum simulation speed ***")
        
        print(f"Visualization mode: {viz_mode}")
        if do_realtime:
            print("  - Real-time visualization: ENABLED")
        if do_logging:
            print(f"  - Logging: ENABLED (saving to {log_path})")

        if do_realtime:
            print("Setting up real-time visualization...")
            self.setup_realtime_visualization()
            
        if do_logging:
            self.logger = SimulationLogger(log_dir=log_path)
            self.logger.initialize(self, self.env_config)

        step = 0
        
        robots = self.robots
        num_robots = len(robots)
        returning_home = self.returning_home
        robots_home = self.robots_home
        return_home_coverage = self.return_home_coverage
        
        start_time = time.perf_counter()
        last_report_time = start_time
        last_report_step = 0
        report_interval_seconds = 3.0
        
        perf_stats = defaultdict(float)
        
        while True:
            if steps is not None and step >= steps:
                break

            if returning_home and len(robots_home) == num_robots:
                print("\n*** ALL ROBOTS RETURNED HOME ***")
                break

            t0 = time.perf_counter()
            
            active_robots = [r for r in robots if not r.manual_control]
            any_idle = any(r.goal is None for r in active_robots)
            
            should_plan = (step % 50 == 0) or (any_idle and step % 5 == 0)
            
            if should_plan:
                coverage = self.calculate_coverage(use_cache=False)
                
                if not returning_home and coverage >= return_home_coverage:
                    print(f"\n*** COVERAGE {coverage:.1f}% >= {return_home_coverage}% - RETURNING HOME ***")
                    returning_home = True
                    self.returning_home = True
                    self.trigger_return_home()
                elif not returning_home:
                    self.assign_global_goals(step)
            
            perf_stats['global_planning'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            for robot in robots:
                self.exploration_logic(robot, step)
                
                if returning_home and robot.mode == 'RETURNING_HOME':
                    pos, _ = p.getBasePositionAndOrientation(robot.id)
                    dx = pos[0] - robot.home_position[0]
                    dy = pos[1] - robot.home_position[1]
                    if dx*dx + dy*dy < 1.0: 
                        robot.mode = 'HOME'
                        robot.goal = None
                        robot.path = []
                        robots_home.add(robot.id)
                        print(f"Robot {robot.id} arrived home!")
            perf_stats['local_planning'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            if step % scan_interval == 0:
                for robot in robots:
                    robot.get_lidar_scan(num_rays=90, max_range=15)
                    self.update_occupancy_grid(robot)
                    
                    if robot.mode != 'HOME':
                        robot.update_global_graph()

                self._coverage_cache_valid = False
                self._cached_frontiers = None
                
                coverage = self.calculate_coverage()
                self.coverage_history.append((step, coverage))
                
            perf_stats['sensing'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            if step % viz_update_interval == 0:
                if do_realtime:
                    self.update_realtime_visualization(step)
                if do_logging:
                    self.logger.log_frame(step, self)
            perf_stats['visualization'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            p.stepSimulation()
            perf_stats['physics'] += time.perf_counter() - t0

            if use_gui:
                time.sleep(1./240.)

            current_time = time.perf_counter()
            elapsed_since_report = current_time - last_report_time
            
            if elapsed_since_report >= report_interval_seconds:
                steps_done = step - last_report_step
                sps = steps_done / elapsed_since_report if elapsed_since_report > 0 else 0
                coverage = self.calculate_coverage()
                status = "RETURNING HOME" if returning_home else "EXPLORING"
                
                print(f"Step {step} | Coverage: {coverage:.1f}% | {sps:.0f} steps/sec | Status: {status}")
                
                total_time = sum(perf_stats.values())
                if total_time > 0:
                    print("  [Performance Breakdown]:")
                    print(f"   - Sensing (LIDAR):   {100*perf_stats['sensing']/total_time:.1f}%")
                    print(f"   - Global Planning:   {100*perf_stats['global_planning']/total_time:.1f}%")
                    print(f"   - Local Planning:    {100*perf_stats['local_planning']/total_time:.1f}%")
                    print(f"   - Visualization:     {100*perf_stats['visualization']/total_time:.1f}%")
                    print(f"   - Physics Engine:    {100*perf_stats['physics']/total_time:.1f}%")
                
                perf_stats.clear()
                
                last_report_time = current_time
                last_report_step = step

            step += 1

        total_time = time.perf_counter() - start_time
        print(f"\n*** SIMULATION COMPLETE ***")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Total steps: {step}")
        print(f"  Average speed: {step/total_time:.0f} steps/second")

        if do_realtime:
            self.update_realtime_visualization(step)
            print("\nReal-time visualization complete.")
            
        if do_logging:
            self.logger.log_frame(step, self)
            log_filepath = self.logger.save()

            print(f"\nTo replay this simulation interactively, run:")
            print(f"  python playback.py {log_filepath}")

            return log_filepath

        print("\nSimulation complete!")
        final_coverage = self.calculate_coverage()
        print(f"Final Coverage: {final_coverage:.2f}%")
        print(f"Explored Free Cells: {int(final_coverage/100 * self.total_free_cells)}/{int(self.total_free_cells)}")

        return None

    def cleanup(self):
        self.env.close()


def main():
    print("=" * 60)
    print("Multi-Robot Coverage Mapping")
    print("=" * 60)

    maze_size_input = input("\nEnter maze size (e.g., '10' for 10x10, default=10): ").strip()
    maze_size = int(maze_size_input) if maze_size_input.isdigit() else 10

    cell_size_input = input("Enter cell size in meters (default=10.0): ").strip()
    try:
        cell_size = float(cell_size_input)
    except:
        cell_size = 10.0

    seed_input = input("Enter random seed (press Enter for random): ").strip()
    env_seed = int(seed_input) if seed_input.isdigit() else None

    print("\nEnvironment types:")
    print("  1. Maze (complex maze with walls)")
    print("  2. Blank box (empty room with single wall in middle)")
    print("  3. Cave (organic cellular automata)")
    print("  4. Tunnel (long winding corridor)")
    print("  5. Rooms (dungeon with connected chambers)")
    print("  6. Sewer (grid of interconnected pipes)")
    print("  7. Corridor Rooms (Central hall with attached rooms)")
    env_type_input = input("Choose environment type (1-7, default=1): ").strip()

    if env_type_input == '2':
        env_type = 'blank_box'
    elif env_type_input == '3':
        env_type = 'cave'
    elif env_type_input == '4':
        env_type = 'tunnel'
    elif env_type_input == '5':
        env_type = 'rooms'
    elif env_type_input == '6':
        env_type = 'sewer'
    elif env_type_input == '7':
        env_type = 'corridor_rooms'
    else:
        env_type = 'maze'

    gui_input = input("Show PyBullet 3D window? (y/n, default=n): ").strip().lower()
    use_gui = gui_input == 'y'

    # NEW: Toggle for partition visualization
    part_input = input("Show Rectangular Decomposition visualization? (y/n, default=n): ").strip().lower()
    show_partitions = part_input == 'y'

    num_robots_input = input("Number of robots (1-16, default=3): ").strip()
    if num_robots_input.isdigit():
        num_robots = max(1, min(16, int(num_robots_input)))
    else:
        num_robots = 3

    steps_input = input("Number of simulation steps (press Enter for unlimited): ").strip()
    if steps_input.isdigit():
        max_steps = int(steps_input)
    else:
        max_steps = None

    print("\nVisualization modes:")
    print("  1. realtime - Live matplotlib window (default)")
    print("  2. logging  - Log to file for offline playback (faster)")
    print("  3. both     - Live visualization AND logging")
    print("  4. none     - No visualization (fastest)")
    viz_mode_input = input("Choose visualization mode (1-4, default=1): ").strip()
    
    if viz_mode_input == '2':
        viz_mode = 'logging'
    elif viz_mode_input == '3':
        viz_mode = 'both'
    elif viz_mode_input == '4':
        viz_mode = 'none'
    else:
        viz_mode = 'realtime'

    print(f"\nCreating {maze_size}x{maze_size} {env_type} with {cell_size}m cells and {num_robots} robots...")

    mapper = CoverageMapper(
        use_gui=use_gui,
        maze_size=(maze_size, maze_size),
        cell_size=cell_size,
        env_seed=env_seed,
        env_type=env_type,
        num_robots=num_robots,
        show_partitions=show_partitions
    )

    log_filepath = None

    try:
        log_filepath = mapper.run_simulation(
            steps=max_steps,
            scan_interval=10,
            use_gui=use_gui,
            viz_mode=viz_mode,
            viz_update_interval=50,
            log_path='./logs'
        )

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        if mapper.logger is not None and len(mapper.logger.frames) > 0:
            print("Saving partial log...")
            log_filepath = mapper.logger.save()
            print(f"\nTo replay this simulation interactively, run:")
            print(f"  python playback.py {log_filepath}")
    finally:
        if mapper.realtime_fig is not None:
            plt.close(mapper.realtime_fig)
        mapper.cleanup()
        print("PyBullet disconnected")

    if log_filepath is not None:
        render_input = input("\nRender video from log? (y/n, default=n): ").strip().lower()
        if render_input == 'y':
            try:
                from video_renderer import render_video_from_log
                print("\nRendering video with OpenCV (fast parallel renderer)...")
                render_video_from_log(log_filepath)
            except ImportError:
                print("Warning: video_renderer_opencv.py not found, skipping video rendering")
            except Exception as e:
                print(f"Error rendering video: {e}")


if __name__ == "__main__":
    main()