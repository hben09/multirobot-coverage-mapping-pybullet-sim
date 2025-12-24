"""
OccupancyGridManager - Manages occupancy grid, frontier detection, and coordinate transformations.

This class extracts all grid-related functionality from SimulationManager to improve
separation of concerns and maintainability.
"""

import numpy as np
from collections import deque
from mapping.numba_accelerator import NumbaOccupancyGrid, detect_frontiers_numpy


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

        # Pre-compute maze lookup for coverage calculation
        block_physical_size = self.env.cell_size / 2.0
        self._maze_block_size = block_physical_size
        self._maze_half_block = block_physical_size / 2.0
        self._maze_inv_block_size = 1.0 / block_physical_size

        # Calculate total free cells for coverage using exact count
        self.total_free_cells = self._calculate_exact_free_cell_count()

        # FRONTIER DETECTION
        self._cached_frontiers = None

        # INCREMENTAL COVERAGE TRACKING (using Numpy grid snapshot)
        self._last_coverage_grid = None
        self._incremental_valid_count = 0
        self._cached_coverage = 0.0
        self._coverage_cache_valid = False

    def _calculate_exact_free_cell_count(self):
        """
        Calculate the EXACT number of grid cells that map to free maze cells.

        This replaces the approximation (ground_truth_zeros * scale_factor^2)
        with an exact count to avoid the 97% coverage ceiling.

        Returns:
            int: Exact count of grid cells mapping to free maze cells
        """
        # Get grid dimensions
        height, width = self._numba_occupancy.grid.shape
        offset_x = self._numba_occupancy.grid_offset_x
        offset_y = self._numba_occupancy.grid_offset_y

        # Generate all possible grid coordinates within map bounds
        min_gx = int((self.map_bounds['x_min'] - self.map_bounds['x_min']) / self.grid_resolution) - offset_x
        max_gx = int((self.map_bounds['x_max'] - self.map_bounds['x_min']) / self.grid_resolution) - offset_x
        min_gy = int((self.map_bounds['y_min'] - self.map_bounds['y_min']) / self.grid_resolution) - offset_y
        max_gy = int((self.map_bounds['y_max'] - self.map_bounds['y_min']) / self.grid_resolution) - offset_y

        # Clamp to grid dimensions
        min_gx = max(0, min_gx)
        max_gx = min(width, max_gx)
        min_gy = max(0, min_gy)
        max_gy = min(height, max_gy)

        # Create meshgrid of all cell indices
        gy_range = np.arange(min_gy, max_gy)
        gx_range = np.arange(min_gx, max_gx)
        gx_grid, gy_grid = np.meshgrid(gx_range, gy_range)

        # Flatten to 1D arrays
        gx_flat = gx_grid.flatten()
        gy_flat = gy_grid.flatten()

        # Convert to absolute grid coordinates
        abs_gx = gx_flat + offset_x
        abs_gy = gy_flat + offset_y

        # Convert to world coordinates
        x_min = self.map_bounds['x_min']
        y_min = self.map_bounds['y_min']
        wx = x_min + (abs_gx + 0.5) * self.grid_resolution
        wy = y_min + (abs_gy + 0.5) * self.grid_resolution

        # Convert to maze coordinates
        half_block = self._maze_half_block
        inv_block_size = self._maze_inv_block_size
        mx = ((wx + half_block) * inv_block_size).astype(int)
        my = ((wy + half_block) * inv_block_size).astype(int)

        # Check which cells map to free maze cells
        maze_h, maze_w = self.env.maze_grid.shape
        maze_grid = self.env.maze_grid

        in_bounds = (mx >= 0) & (mx < maze_w) & (my >= 0) & (my < maze_h)
        valid_idx = np.where(in_bounds)[0]

        if len(valid_idx) == 0:
            return 0

        mx_valid = mx[valid_idx]
        my_valid = my[valid_idx]
        is_free = (maze_grid[my_valid, mx_valid] == 0)

        total_free = np.sum(is_free)
        return int(total_free)

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

        # Update Numpy grid (single source of truth)
        num_rays, new_free_cells, new_obstacle_cells = self._numba_occupancy.update_from_lidar(
            robot_pos,
            lidar_data
        )

        # Track if we explored new cells
        has_new_cells = len(new_free_cells) > 0 or len(new_obstacle_cells) > 0

        # Invalidate frontier cache if new cells were explored
        if has_new_cells:
            self.invalidate_frontier_cache()

        return has_new_cells

    def detect_frontiers(self, use_cache=False):
        """
        Detect and cluster frontier cells using NUMPY-OPTIMIZED operations.

        Args:
            use_cache: If True, return cached frontiers if available

        Returns:
            List of frontier clusters, each with 'pos', 'grid_pos', and 'size'
        """
        if use_cache and self._cached_frontiers is not None and len(self._cached_frontiers) > 0:
            return self._cached_frontiers

        # Use vectorized Numpy frontier detection (FAST PATH)
        numpy_grid = self._numba_occupancy.grid
        offset_x, offset_y = self._numba_occupancy.grid_offset_x, self._numba_occupancy.grid_offset_y

        clusters = detect_frontiers_numpy(
            numpy_grid,
            offset_x,
            offset_y,
            self.grid_resolution,
            self.map_bounds
        )

        self._cached_frontiers = clusters
        return clusters

    def calculate_coverage(self, use_cache=True):
        """
        Calculate coverage percentage - INCREMENTAL VERSION using Numpy grid.

        Args:
            use_cache: If True, use cached coverage if valid

        Returns:
            float: Coverage percentage (0-100)
        """
        if self.total_free_cells == 0:
            return 0.0

        # Get current grid
        current_grid = self._numba_occupancy.grid
        offset_x, offset_y = self._numba_occupancy.grid_offset_x, self._numba_occupancy.grid_offset_y

        # Initialize cache on first call
        if self._last_coverage_grid is None:
            self._last_coverage_grid = np.zeros_like(current_grid)

        # Find cells that changed from 0 (unknown) to 1 (free)
        new_free_mask = (self._last_coverage_grid == 0) & (current_grid == 1)

        # If nothing changed and cache is valid, return cached value
        if not np.any(new_free_mask) and use_cache:
            return self._cached_coverage

        # Get coordinates of new free cells
        new_coords = np.argwhere(new_free_mask)

        if len(new_coords) == 0 and use_cache:
            return self._cached_coverage

        # Validate new cells against ground truth maze
        maze_h, maze_w = self.env.maze_grid.shape
        maze_grid = self.env.maze_grid
        x_min = self.map_bounds['x_min']
        y_min = self.map_bounds['y_min']
        grid_res = self.grid_resolution
        half_block = self._maze_half_block
        inv_block_size = self._maze_inv_block_size

        # Vectorized conversion to world coordinates
        abs_gx = new_coords[:, 1] + offset_x  # column = x
        abs_gy = new_coords[:, 0] + offset_y  # row = y
        wx = x_min + (abs_gx + 0.5) * grid_res
        wy = y_min + (abs_gy + 0.5) * grid_res

        # Vectorized conversion to maze coordinates
        mx = ((wx + half_block) * inv_block_size).astype(int)
        my = ((wy + half_block) * inv_block_size).astype(int)

        # Check bounds
        valid_mask = (mx >= 0) & (mx < maze_w) & (my >= 0) & (my < maze_h)

        # Count cells that are in bounds AND correspond to free cells in ground truth
        # Use advanced indexing to check maze values in one operation
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            mx_valid = mx[valid_indices]
            my_valid = my[valid_indices]
            # Check which of these valid cells are free (0) in ground truth
            is_free_in_maze = (maze_grid[my_valid, mx_valid] == 0)
            new_valid_count = np.sum(is_free_in_maze)
        else:
            new_valid_count = 0

        self._incremental_valid_count += new_valid_count

        # Update cache
        self._last_coverage_grid = current_grid.copy()

        coverage_percent = min(100.0, (self._incremental_valid_count / self.total_free_cells) * 100)

        self._cached_coverage = coverage_percent
        self._coverage_cache_valid = True

        return coverage_percent

    def invalidate_coverage_cache(self):
        """Invalidate the coverage cache."""
        self._coverage_cache_valid = False

    def invalidate_frontier_cache(self):
        """Invalidate the frontier cache."""
        self._cached_frontiers = None

    def get_grid_stats(self):
        """Get statistics about the grid state."""
        numpy_grid = self._numba_occupancy.grid
        explored_count = np.sum(numpy_grid == 1)
        obstacle_count = np.sum(numpy_grid == 2)

        return {
            'explored_cells': int(explored_count),
            'obstacle_cells': int(obstacle_count),
            'total_free_cells': self.total_free_cells,
            'coverage': self._cached_coverage,
            'valid_explored_count': self._incremental_valid_count
        }

    def debug_coverage_discrepancy(self):
        """
        Debug helper to identify why coverage might not reach 100%.

        Returns:
            Dict with diagnostic information
        """
        # Get all free cells from the grid
        numpy_grid = self._numba_occupancy.grid
        offset_x, offset_y = self._numba_occupancy.grid_offset_x, self._numba_occupancy.grid_offset_y

        free_coords = np.argwhere(numpy_grid == 1)
        total_explored = len(free_coords)

        if total_explored == 0:
            return {'message': 'No cells explored yet'}

        # Convert all to absolute coordinates and check against maze
        abs_gx = free_coords[:, 1] + offset_x
        abs_gy = free_coords[:, 0] + offset_y

        x_min = self.map_bounds['x_min']
        y_min = self.map_bounds['y_min']
        grid_res = self.grid_resolution
        half_block = self._maze_half_block
        inv_block_size = self._maze_inv_block_size

        wx = x_min + (abs_gx + 0.5) * grid_res
        wy = y_min + (abs_gy + 0.5) * grid_res

        mx = ((wx + half_block) * inv_block_size).astype(int)
        my = ((wy + half_block) * inv_block_size).astype(int)

        maze_h, maze_w = self.env.maze_grid.shape
        maze_grid = self.env.maze_grid

        # Count different categories
        in_bounds = (mx >= 0) & (mx < maze_w) & (my >= 0) & (my < maze_h)
        out_of_bounds_count = np.sum(~in_bounds)

        in_bounds_idx = np.where(in_bounds)[0]
        if len(in_bounds_idx) > 0:
            mx_valid = mx[in_bounds_idx]
            my_valid = my[in_bounds_idx]
            is_free = (maze_grid[my_valid, mx_valid] == 0)
            is_obstacle = (maze_grid[my_valid, mx_valid] == 1)

            valid_free_count = np.sum(is_free)
            mapped_to_obstacle_count = np.sum(is_obstacle)
        else:
            valid_free_count = 0
            mapped_to_obstacle_count = 0

        return {
            'total_explored_cells': total_explored,
            'out_of_bounds': out_of_bounds_count,
            'mapped_to_obstacles': mapped_to_obstacle_count,
            'valid_free_cells': valid_free_count,
            'expected_free_cells': self.total_free_cells,
            'incremental_count': self._incremental_valid_count,
            'discrepancy': self.total_free_cells - valid_free_count,
            'coverage_from_count': (valid_free_count / self.total_free_cells * 100) if self.total_free_cells > 0 else 0
        }
