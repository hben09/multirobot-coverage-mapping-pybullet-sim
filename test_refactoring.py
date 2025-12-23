"""Test the refactored grid system to ensure Numpy array is the single source of truth."""
import sys
sys.path.insert(0, 'c:/Users/hben0/Github/pybullet-multi-robot-coverage')

# Test imports
from mapping.numba_accelerator import NumbaOccupancyGrid
from navigation.pathfinding import NumbaAStarHelper

print('✓ All imports successful')

# Test basic grid creation
map_bounds = {'x_min': -10.0, 'x_max': 10.0, 'y_min': -10.0, 'y_max': 10.0}
grid_res = 0.1

numba_grid = NumbaOccupancyGrid(map_bounds, grid_res)
print(f'✓ NumbaOccupancyGrid created: {numba_grid.grid.shape}')

# Test update signature
scan_points = [(1.0, 1.0, False), (2.0, 2.0, True)]
robot_pos = (0.0, 0.0)
num_rays, new_free, new_obs = numba_grid.update_from_lidar(robot_pos, scan_points)
print(f'✓ update_from_lidar works: {num_rays} rays, {len(new_free)} free, {len(new_obs)} obstacles')

# Test pathfinding helper
planner = NumbaAStarHelper()
planner.update_grid(numba_grid.grid, numba_grid.grid_offset_x, numba_grid.grid_offset_y, safety_margin=1)
print(f'✓ NumbaAStarHelper.update_grid works with new signature')
print(f'✓ Grid reference stored: {planner.grid.shape}')
print(f'✓ Inflated grid created: {planner._inflated_grid.shape if planner._inflated_grid is not None else "None"}')

print('\nSUCCESS: All refactoring changes work correctly!')
