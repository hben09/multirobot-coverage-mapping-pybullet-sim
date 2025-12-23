"""
Numba-optimized occupancy grid updates for LIDAR processing.

This module accelerates the Bresenham line tracing and grid updates
that were previously the bottleneck in update_occupancy_grid().

Grid cell values:
  0 = Unknown
  1 = Free (explored)
  2 = Obstacle
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _bresenham_single_ray(grid, x0, y0, x1, y1, mark_endpoint_obstacle):
    """
    Trace a single ray using Bresenham's algorithm.
    Marks all cells along the ray as free (1), except obstacles stay as obstacles.
    If mark_endpoint_obstacle is True, marks the endpoint as obstacle (2).
    
    Args:
        grid: 2D numpy array (modified in-place)
        x0, y0: Start point (robot position in grid coords, already offset)
        x1, y1: End point (hit/max-range point in grid coords, already offset)
        mark_endpoint_obstacle: If True, mark (x1, y1) as obstacle
    """
    height, width = grid.shape
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    
    while True:
        # Bounds check
        if 0 <= x < width and 0 <= y < height:
            # Only mark as free if not already an obstacle
            if grid[y, x] != 2:
                grid[y, x] = 1
        
        # Reached endpoint
        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    # Mark endpoint as obstacle if it was a hit
    if mark_endpoint_obstacle:
        if 0 <= x1 < width and 0 <= y1 < height:
            grid[y1, x1] = 2


@njit(cache=True)
def _process_all_rays(grid, robot_gx, robot_gy, ray_endpoints, ray_is_hit):
    """
    Process all LIDAR rays in a single compiled function.
    
    Args:
        grid: 2D numpy array (height, width), modified in-place
        robot_gx, robot_gy: Robot position in grid coordinates (already offset)
        ray_endpoints: Nx2 array of (gx, gy) endpoints (already offset)
        ray_is_hit: N-length boolean array, True if ray hit an obstacle
    """
    num_rays = ray_endpoints.shape[0]
    
    for i in range(num_rays):
        end_gx = ray_endpoints[i, 0]
        end_gy = ray_endpoints[i, 1]
        is_hit = ray_is_hit[i]
        
        _bresenham_single_ray(grid, robot_gx, robot_gy, end_gx, end_gy, is_hit)


@njit(cache=True)
def _process_rays_collect_new(grid, robot_gx, robot_gy, ray_endpoints_offset, 
                               ray_endpoints_raw, ray_is_hit, 
                               out_free_cells, out_obs_cells,
                               grid_offset_x, grid_offset_y):
    """
    Process all rays and collect ONLY cells that change from unknown (0).
    This dramatically reduces the number of cells to sync to Python dicts.
    
    Returns: (num_new_free_cells, num_new_obs_cells)
    """
    height, width = grid.shape
    num_rays = ray_endpoints_offset.shape[0]
    
    free_idx = 0
    obs_idx = 0
    max_free = out_free_cells.shape[0]
    max_obs = out_obs_cells.shape[0]
    
    for i in range(num_rays):
        end_gx = ray_endpoints_offset[i, 0]
        end_gy = ray_endpoints_offset[i, 1]
        end_raw_gx = ray_endpoints_raw[i, 0]
        end_raw_gy = ray_endpoints_raw[i, 1]
        is_hit = ray_is_hit[i]
        
        x0, y0 = robot_gx, robot_gy
        x1, y1 = end_gx, end_gy
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            if 0 <= x < width and 0 <= y < height:
                current_val = grid[y, x]
                
                # Only process if cell is UNKNOWN (0) or we're marking obstacle
                if current_val == 0:
                    # New free cell
                    grid[y, x] = 1
                    if free_idx < max_free:
                        # Convert back to raw coords for dict key
                        out_free_cells[free_idx, 0] = x + grid_offset_x
                        out_free_cells[free_idx, 1] = y + grid_offset_y
                        free_idx += 1
                # If already free (1) or obstacle (2), skip
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        # Mark endpoint as obstacle if hit
        if is_hit:
            if 0 <= x1 < width and 0 <= y1 < height:
                # Only record if this is a NEW obstacle
                if grid[y1, x1] != 2:
                    grid[y1, x1] = 2
                    if obs_idx < max_obs:
                        out_obs_cells[obs_idx, 0] = end_raw_gx
                        out_obs_cells[obs_idx, 1] = end_raw_gy
                        obs_idx += 1
    
    return free_idx, obs_idx


@njit(cache=True, parallel=False)  # parallel=False for now, can test parallel=True later
def _batch_world_to_grid(points_world, x_min, y_min, inv_resolution):
    """
    Convert array of world coordinates to grid coordinates.
    
    Args:
        points_world: Nx2 array of (x, y) world coordinates
        x_min, y_min: Grid origin in world coordinates
        inv_resolution: 1.0 / grid_resolution
        
    Returns:
        Nx2 array of (gx, gy) grid coordinates as int32
    """
    n = points_world.shape[0]
    grid_coords = np.empty((n, 2), dtype=np.int32)
    
    for i in range(n):
        grid_coords[i, 0] = int((points_world[i, 0] - x_min) * inv_resolution)
        grid_coords[i, 1] = int((points_world[i, 1] - y_min) * inv_resolution)
    
    return grid_coords


class NumbaOccupancyGrid:
    """
    Manages a NumPy-based occupancy grid with Numba-accelerated updates.
    Keeps dict/set synchronized for compatibility with existing code.
    """
    
    def __init__(self, map_bounds, grid_resolution):
        """
        Initialize the occupancy grid.
        
        Args:
            map_bounds: Dict with 'x_min', 'x_max', 'y_min', 'y_max'
            grid_resolution: Size of each grid cell in meters
        """
        self.map_bounds = map_bounds
        self.grid_resolution = grid_resolution
        self.inv_resolution = 1.0 / grid_resolution
        
        # Calculate grid dimensions with padding for safety
        padding = 10  # Extra cells around the edges
        self.grid_offset_x = int((map_bounds['x_min']) * self.inv_resolution) - padding
        self.grid_offset_y = int((map_bounds['y_min']) * self.inv_resolution) - padding
        
        width = int((map_bounds['x_max'] - map_bounds['x_min']) * self.inv_resolution) + 2 * padding
        height = int((map_bounds['y_max'] - map_bounds['y_min']) * self.inv_resolution) + 2 * padding
        
        # Ensure minimum size
        width = max(width, 100)
        height = max(height, 100)
        
        self.width = width
        self.height = height
        
        # Main grid: 0=unknown, 1=free, 2=obstacle
        self.grid = np.zeros((height, width), dtype=np.int8)
        
        # Track if we've warmed up Numba
        self._warmed_up = False
        
        print(f"NumbaOccupancyGrid initialized: {width}x{height} cells "
              f"({width * height / 1e6:.2f}M cells), "
              f"offset=({self.grid_offset_x}, {self.grid_offset_y})")
    
    def _warmup(self):
        """Pre-compile Numba functions with small test data."""
        if self._warmed_up:
            return
        
        # Small test to trigger JIT compilation
        test_grid = np.zeros((10, 10), dtype=np.int8)
        test_endpoints = np.array([[5, 5]], dtype=np.int32)
        test_hits = np.array([True], dtype=np.bool_)
        _process_all_rays(test_grid, 0, 0, test_endpoints, test_hits)
        
        test_points = np.array([[0.0, 0.0]], dtype=np.float64)
        _batch_world_to_grid(test_points, 0.0, 0.0, 1.0)
        
        self._warmed_up = True
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices."""
        gx = int((x - self.map_bounds['x_min']) * self.inv_resolution) - self.grid_offset_x
        gy = int((y - self.map_bounds['y_min']) * self.inv_resolution) - self.grid_offset_y
        return gx, gy
    
    def world_to_grid_raw(self, x, y):
        """Convert world coordinates to raw grid indices (without offset, for dict compatibility)."""
        gx = int((x - self.map_bounds['x_min']) * self.inv_resolution)
        gy = int((y - self.map_bounds['y_min']) * self.inv_resolution)
        return gx, gy
    
    def update_from_lidar(self, robot_pos, scan_points, occupancy_dict, explored_set, obstacle_set):
        """
        Update the grid from LIDAR scan data.
        
        OPTIMIZATION: Only sync cells that actually changed from unknown (0) to 
        free (1) or obstacle (2). Skip cells that were already known.
        """
        self._warmup()
        
        if not scan_points:
            return 0
        
        # Convert robot position to grid coords
        robot_gx_offset, robot_gy_offset = self.world_to_grid(robot_pos[0], robot_pos[1])
        robot_gx_raw, robot_gy_raw = self.world_to_grid_raw(robot_pos[0], robot_pos[1])
        
        # Mark robot position
        if 0 <= robot_gx_offset < self.width and 0 <= robot_gy_offset < self.height:
            if self.grid[robot_gy_offset, robot_gx_offset] == 0:
                self.grid[robot_gy_offset, robot_gx_offset] = 1
                robot_raw = (robot_gx_raw, robot_gy_raw)
                occupancy_dict[robot_raw] = 1
                explored_set.add(robot_raw)
        
        # Convert scan points to numpy arrays
        n_points = len(scan_points)
        endpoints_world = np.empty((n_points, 2), dtype=np.float64)
        is_hit = np.empty(n_points, dtype=np.bool_)
        
        for i, (x, y, hit) in enumerate(scan_points):
            endpoints_world[i, 0] = x
            endpoints_world[i, 1] = y
            is_hit[i] = hit
        
        # Batch convert to grid coordinates
        endpoints_grid_raw = _batch_world_to_grid(
            endpoints_world,
            self.map_bounds['x_min'],
            self.map_bounds['y_min'],
            self.inv_resolution
        )
        
        # Apply offset for array indexing
        endpoints_grid_offset = endpoints_grid_raw.copy()
        endpoints_grid_offset[:, 0] -= self.grid_offset_x
        endpoints_grid_offset[:, 1] -= self.grid_offset_y

        # Allocate output buffers for NEW cells only
        max_cells_per_ray = 100
        max_free_cells = n_points * max_cells_per_ray
        max_obs_cells = n_points

        out_free_cells = np.empty((max_free_cells, 2), dtype=np.int32)
        out_obs_cells = np.empty((max_obs_cells, 2), dtype=np.int32)

        # Combined ray tracing + new cell collection
        num_free, num_obs = _process_rays_collect_new(
            self.grid,
            robot_gx_offset, robot_gy_offset,
            endpoints_grid_offset,
            endpoints_grid_raw,
            is_hit,
            out_free_cells,
            out_obs_cells,
            self.grid_offset_x,
            self.grid_offset_y
        )

        # Sync ONLY new cells to dict/set
        for i in range(num_free):
            cell = (int(out_free_cells[i, 0]), int(out_free_cells[i, 1]))
            occupancy_dict[cell] = 1
            explored_set.add(cell)

        for i in range(num_obs):
            cell = (int(out_obs_cells[i, 0]), int(out_obs_cells[i, 1]))
            occupancy_dict[cell] = 2
            obstacle_set.add(cell)

        return n_points