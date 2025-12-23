import numpy as np
from numba import njit

print("✓ Numba detected - A* pathfinding will be JIT-compiled for maximum speed")


# === NUMBA JIT-COMPILED HELPER FUNCTIONS ===

@njit(cache=True)
def _numba_heuristic(ax, ay, bx, by):
    """Octile distance heuristic for 8-way movement."""
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

    # 1. Bubble up to maintain heap property
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
        # 1. Move last element to root
        heap[0, 0] = heap[heap_size, 0]
        heap[0, 1] = heap[heap_size, 1]
        heap[0, 2] = heap[heap_size, 2]

        # 2. Bubble down to restore heap property
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


# === A* CORE ALGORITHM ===

@njit(cache=True)
def _numba_astar_core(
    grid, start_x, start_y, goal_x, goal_y,
    grid_offset_x, grid_offset_y, use_inflation, max_iterations=50000
):
    """
    Numba JIT-compiled A* pathfinding.
    Grid values: 0=unknown, 1=free, 2=obstacle, 3=inflated
    """
    # 1. Define 8-way movement with costs
    dx_arr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
    dy_arr = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
    costs = np.array([1.41421356, 1.0, 1.41421356, 1.0, 1.0, 1.41421356, 1.0, 1.41421356], dtype=np.float64)

    height, width = grid.shape

    # 2. Convert to grid coordinates
    sx = start_x - grid_offset_x
    sy = start_y - grid_offset_y
    gx = goal_x - grid_offset_x
    gy = goal_y - grid_offset_y

    # 3. Validate start and goal
    if sx < 0 or sx >= width or sy < 0 or sy >= height:
        return np.empty((0, 2), dtype=np.int32), False
    if gx < 0 or gx >= width or gy < 0 or gy >= height:
        return np.empty((0, 2), dtype=np.int32), False

    # 4. Initialize data structures
    g_score = np.full((height, width), np.inf, dtype=np.float64)
    g_score[sy, sx] = 0.0

    parent_dir = np.full((height, width), -1, dtype=np.int8)
    closed = np.zeros((height, width), dtype=np.bool_)

    max_heap_size = height * width
    heap = np.zeros((max_heap_size, 3), dtype=np.float64)
    heap_size = 0

    # 5. Initialize heap with start node
    f_start = _numba_heuristic(sx, sy, gx, gy)
    heap_size = _numba_heap_push(heap, heap_size, f_start, float(sx), float(sy))

    iterations = 0
    found = False

    # 6. Main A* loop
    while iterations < max_iterations and heap_size > 0:
        iterations += 1

        _, cx, cy, heap_size = _numba_heap_pop(heap, heap_size)

        if cx == -1:
            break

        if closed[cy, cx]:
            continue

        closed[cy, cx] = True

        # 7. Check if goal reached
        if cx == gx and cy == gy:
            found = True
            break

        current_g = g_score[cy, cx]

        # 8. Explore neighbors
        for i in range(8):
            nx = cx + dx_arr[i]
            ny = cy + dy_arr[i]

            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue

            if closed[ny, nx]:
                continue

            cell_val = grid[ny, nx]

            if cell_val == 2:
                continue

            if cell_val == 0:
                continue

            if use_inflation and cell_val == 3:
                if not (nx == sx and ny == sy) and not (nx == gx and ny == gy):
                    continue

            # 9. Check diagonal corner cutting
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

            # 10. Update neighbor if better path found
            new_g = current_g + costs[i]

            if new_g < g_score[ny, nx]:
                g_score[ny, nx] = new_g
                f = new_g + _numba_heuristic(nx, ny, gx, gy)
                parent_dir[ny, nx] = i
                heap_size = _numba_heap_push(heap, heap_size, f, float(nx), float(ny))

    if not found:
        return np.empty((0, 2), dtype=np.int32), False

    # 11. Count path length by backtracking
    path_len = 0
    px, py = gx, gy
    while not (px == sx and py == sy):
        path_len += 1
        d = parent_dir[py, px]
        if d == -1:
            return np.empty((0, 2), dtype=np.int32), False
        px = px - dx_arr[d]
        py = py - dy_arr[d]

    # 12. Build path array
    path = np.empty((path_len, 2), dtype=np.int32)
    px, py = gx, gy
    for i in range(path_len - 1, -1, -1):
        path[i, 0] = px + grid_offset_x
        path[i, 1] = py + grid_offset_y
        d = parent_dir[py, px]
        px = px - dx_arr[d]
        py = py - dy_arr[d]

    return path, True


# === A* HELPER CLASS ===

class NumbaAStarHelper:
    """Manages grid conversion and caching for Numba A* pathfinding."""

    def __init__(self):
        self.grid = None
        self.grid_offset_x = 0
        self.grid_offset_y = 0
        self._last_explored_count = 0
        self._last_obstacle_count = 0
        self._warmed_up = False

    def _warmup(self):
        """Pre-compile Numba functions on first use."""
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

        # 1. Check if rebuild needed
        if (len(occupancy_grid) == self._last_explored_count and
            len(obstacle_cells) == self._last_obstacle_count and
            self.grid is not None):
            return

        self._last_explored_count = len(occupancy_grid)
        self._last_obstacle_count = len(obstacle_cells)

        # 2. Find grid bounds
        all_cells = list(occupancy_grid.keys())
        xs = [c[0] for c in all_cells]
        ys = [c[1] for c in all_cells]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # 3. Add padding for safety margin
        padding = safety_margin + 2
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding

        self.grid_offset_x = min_x
        self.grid_offset_y = min_y

        width = max_x - min_x + 1
        height = max_y - min_y + 1

        # 4. Initialize grid
        self.grid = np.zeros((height, width), dtype=np.int8)

        # 5. Fill occupancy values
        for (x, y), val in occupancy_grid.items():
            gx = x - min_x
            gy = y - min_y
            if 0 <= gx < width and 0 <= gy < height:
                self.grid[gy, gx] = val

        # 6. Mark obstacles
        for (x, y) in obstacle_cells:
            gx = x - min_x
            gy = y - min_y
            if 0 <= gx < width and 0 <= gy < height:
                self.grid[gy, gx] = 2

        # 7. Apply safety margin inflation
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

        # 1. Attempt path with inflation
        path_array, found = _numba_astar_core(
            self.grid,
            start_grid[0], start_grid[1],
            goal_grid[0], goal_grid[1],
            self.grid_offset_x, self.grid_offset_y,
            use_inflation
        )

        # 2. Retry without inflation if failed
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

        # 3. Convert to list of tuples
        path = [(int(path_array[i, 0]), int(path_array[i, 1])) for i in range(len(path_array))]

        # 4. Downsample long paths
        if len(path) > 8:
            path = path[::2]
            if path[-1] != goal_grid:
                path.append(goal_grid)

        return path
