"""
OccupancyGridManager - Manages occupancy grid, frontier detection, and coordinate transformations.

This class extracts all grid-related functionality from SimulationManager to improve
separation of concerns and maintainability.
"""

import numpy as np
from collections import deque
from mapping.numba_accelerator import NumbaOccupancyGrid


class OccupancyGridManager:
    """Manages occupancy grid, frontier detection, and coordinate transformations."""

    def __init__(self, map_bounds, grid_resolution, env):
        """
        Initialize the occupancy grid manager.

        Args:
            map_bounds: Dict with 'x_min', 'x_max', 'y_min', 'y_max'
            grid_resolution: Size of each grid cell in meters
            env: PybulletRenderer environment (for coverage calculation)
        """
        self.map_bounds = map_bounds
        self.grid_resolution = grid_resolution
        self.env = env

        # Numba-accelerated occupancy grid (SINGLE SOURCE OF TRUTH)
        self._numba_occupancy = NumbaOccupancyGrid(map_bounds, grid_resolution)

        # DEPRECATED: Legacy dict/set kept temporarily for compatibility
        # TODO: Remove these once all code uses Numpy grid directly
        self.occupancy_grid = {}  # 1=Free, 2=Obstacle
        self.explored_cells = set()
        self.obstacle_cells = set()

        # Pre-compute maze lookup for coverage calculation
        block_physical_size = self.env.cell_size / 2.0
        self._maze_block_size = block_physical_size
        self._maze_half_block = block_physical_size / 2.0
        self._maze_inv_block_size = 1.0 / block_physical_size

        # Calculate total free cells for coverage
        scale_factor = block_physical_size / grid_resolution
        ground_truth_zeros = np.sum(self.env.maze_grid == 0)
        self.total_free_cells = ground_truth_zeros * (scale_factor ** 2)

        # Pre-computed grid bounds for frontier detection
        self._grid_bounds = None

        # INCREMENTAL FRONTIER DETECTION
        self._frontier_candidates = set()
        self._cached_frontiers = None

        # INCREMENTAL COVERAGE TRACKING
        self._incremental_valid_count = 0
        self._processed_explored_cells = set()
        self._cached_coverage = 0.0
        self._coverage_cache_valid = False
        self._coverage_explored_count = 0

    def get_numpy_grid(self):
        """
        Get direct reference to the Numpy occupancy grid.

        Returns:
            Numpy array reference (not a copy!)
        """
        return self._numba_occupancy.grid

    def get_grid_offset(self):
        """
        Get the grid offset coordinates.

        Returns:
            Tuple of (offset_x, offset_y)
        """
        return (self._numba_occupancy.grid_offset_x, self._numba_occupancy.grid_offset_y)

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates."""
        grid_x = int((x - self.map_bounds['x_min']) / self.grid_resolution)
        grid_y = int((y - self.map_bounds['y_min']) / self.grid_resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates."""
        x = self.map_bounds['x_min'] + (grid_x + 0.5) * self.grid_resolution
        y = self.map_bounds['y_min'] + (grid_y + 0.5) * self.grid_resolution
        return (x, y)

    def update_occupancy_grid(self, robot_pos, lidar_data):
        """
        Update occupancy grid from robot LIDAR data - NUMBA OPTIMIZED.

        Args:
            robot_pos: Tuple of (x, y) robot position
            lidar_data: List of LIDAR scan points from robot

        Returns:
            bool: True if new cells were explored
        """
        if not lidar_data:
            return False

        old_explored_count = len(self.explored_cells)

        # Update Numpy grid (single source of truth)
        num_rays, new_free_cells, new_obstacle_cells = self._numba_occupancy.update_from_lidar(
            robot_pos,
            lidar_data
        )

        # Sync to legacy dicts/sets for backward compatibility
        # TODO: Remove this once visualization code uses Numpy grid directly
        for cell in new_free_cells:
            self.occupancy_grid[cell] = 1
            self.explored_cells.add(cell)

        for cell in new_obstacle_cells:
            self.occupancy_grid[cell] = 2
            self.obstacle_cells.add(cell)

        # Track frontier candidates near newly explored areas
        if len(self.explored_cells) > old_explored_count:
            robot_cell = self.world_to_grid(robot_pos[0], robot_pos[1])
            lidar_range_cells = int(15.0 / self.grid_resolution) + 1

            for dx in range(-lidar_range_cells, lidar_range_cells + 1):
                for dy in range(-lidar_range_cells, lidar_range_cells + 1):
                    cell = (robot_cell[0] + dx, robot_cell[1] + dy)
                    if cell in self.explored_cells:
                        self._frontier_candidates.add(cell)

            return True

        return False

    def detect_frontiers(self, use_cache=False):
        """
        Detect and cluster frontier cells - OPTIMIZED with candidate tracking.

        Args:
            use_cache: If True, return cached frontiers if available

        Returns:
            List of frontier clusters, each with 'pos', 'grid_pos', and 'size'
        """
        if use_cache and self._cached_frontiers is not None and len(self._cached_frontiers) > 0:
            return self._cached_frontiers

        # Initialize grid bounds on first call
        if self._grid_bounds is None:
            self._grid_bounds = {
                'min_gx': int((self.map_bounds['x_min'] - self.map_bounds['x_min']) / self.grid_resolution),
                'max_gx': int((self.map_bounds['x_max'] - self.map_bounds['x_min']) / self.grid_resolution),
                'min_gy': int((self.map_bounds['y_min'] - self.map_bounds['y_min']) / self.grid_resolution),
                'max_gy': int((self.map_bounds['y_max'] - self.map_bounds['y_min']) / self.grid_resolution),
            }

        gb = self._grid_bounds
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # Use frontier candidates if available, otherwise check all explored cells
        if self._frontier_candidates:
            cells_to_check = self._frontier_candidates
        else:
            cells_to_check = self.explored_cells

        frontiers = set()
        occupancy = self.occupancy_grid
        non_frontier_candidates = set()

        # Find frontier cells (free cells adjacent to unknown cells)
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

        # Remove non-frontier cells from candidates
        self._frontier_candidates -= non_frontier_candidates

        if not frontiers:
            self._cached_frontiers = []
            return []

        # Cluster frontiers using flood fill
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

            # Only keep clusters with sufficient size
            if len(cluster) > 6:
                avg_x = sum(c[0] for c in cluster) / len(cluster)
                avg_y = sum(c[1] for c in cluster) / len(cluster)
                best_point = min(cluster, key=lambda p: (p[0] - avg_x)**2 + (p[1] - avg_y)**2)
                wx, wy = self.grid_to_world(best_point[0], best_point[1])
                clusters.append({
                    'pos': (wx, wy),
                    'grid_pos': best_point,
                    'size': len(cluster)
                })

        self._cached_frontiers = clusters
        return clusters

    def calculate_coverage(self, use_cache=True):
        """
        Calculate coverage percentage - INCREMENTAL VERSION.

        Args:
            use_cache: If True, use cached coverage if valid

        Returns:
            float: Coverage percentage (0-100)
        """
        if self.total_free_cells == 0:
            return 0.0

        current_count = len(self.explored_cells)
        if use_cache and current_count == len(self._processed_explored_cells):
            return self._cached_coverage

        new_cells = self.explored_cells - self._processed_explored_cells

        if not new_cells:
            return self._cached_coverage

        # Validate new cells against ground truth maze
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
        """Invalidate the coverage cache."""
        self._coverage_cache_valid = False

    def invalidate_frontier_cache(self):
        """Invalidate the frontier cache."""
        self._cached_frontiers = None

    def get_grid_stats(self):
        """Get statistics about the grid state."""
        return {
            'explored_cells': len(self.explored_cells),
            'obstacle_cells': len(self.obstacle_cells),
            'frontier_candidates': len(self._frontier_candidates),
            'total_free_cells': self.total_free_cells,
            'coverage': self._cached_coverage
        }
