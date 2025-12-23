"""
Spatial decomposition utilities for coverage mapping.
Provides methods for decomposing free space into rectangular regions.
"""

import numpy as np


def decompose_grid_to_rectangles(occupancy_grid, offset_x=0, offset_y=0, max_rects=None):
    """
    Decomposes free space into rectangles and optionally returns only the top N largest.

    This function takes an occupancy grid (Numpy array or dict) and partitions the free
    space into non-overlapping rectangles using a greedy maximal rectangle decomposition.

    Args:
        occupancy_grid (np.ndarray or dict): Either:
                                             - 2D Numpy array where value 1 = free space
                                             - Dictionary mapping (gx, gy) -> occupancy values
        offset_x (int): Grid offset in x direction (for Numpy arrays)
        offset_y (int): Grid offset in y direction (for Numpy arrays)
        max_rects (int, optional): If specified, only return the N largest rectangles
                                   sorted by area (width × height).

    Returns:
        list: List of rectangles as tuples (gx, gy, width, height) where:
              - gx, gy: Grid coordinates of bottom-left corner (in absolute coords)
              - width, height: Rectangle dimensions in grid cells

    Examples:
        >>> occupancy = {(0,0): 1, (1,0): 1, (0,1): 1, (1,1): 1}
        >>> rects = decompose_grid_to_rectangles(occupancy)
        >>> # Returns [(0, 0, 2, 2)] - one 2×2 rectangle

        >>> numpy_grid = np.array([[1, 1], [1, 1]])
        >>> rects = decompose_grid_to_rectangles(numpy_grid, offset_x=0, offset_y=0)
        >>> # Returns [(0, 0, 2, 2)]
    """
    # Handle Numpy array input (NEW FAST PATH)
    if isinstance(occupancy_grid, np.ndarray):
        # Extract boolean mask for free cells
        grid = (occupancy_grid == 1)

        # Check if there are any free cells
        if not np.any(grid):
            return []

        h, w = grid.shape
        min_x, min_y = 0, 0

    # Handle legacy dictionary input (BACKWARD COMPATIBILITY)
    else:
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

        # For dict input, offset is embedded in the coordinates
        offset_x = min_x
        offset_y = min_y
        min_x, min_y = 0, 0

    # Decompose using greedy maximal rectangles
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

        # Store rect (gx, gy, w, h) in absolute coordinates
        rects.append((c + min_x + offset_x, r + min_y + offset_y, current_w, current_h))

        # Mark as visited
        remaining[r : r + current_h, c : c + current_w] = False

    # Sort and limit if requested
    if max_rects is not None:
        # Sort by Area (Width * Height) in descending order
        rects.sort(key=lambda r: r[2] * r[3], reverse=True)
        return rects[:max_rects]

    return rects


def calculate_rectangle_area(rect):
    """
    Calculate the area of a rectangle.

    Args:
        rect (tuple): Rectangle as (gx, gy, width, height)

    Returns:
        int: Area in grid cells (width × height)
    """
    return rect[2] * rect[3]


def filter_rectangles_by_area(rects, min_area=None, max_area=None):
    """
    Filter rectangles by area constraints.

    Args:
        rects (list): List of rectangles as (gx, gy, width, height) tuples
        min_area (int, optional): Minimum area threshold (inclusive)
        max_area (int, optional): Maximum area threshold (inclusive)

    Returns:
        list: Filtered list of rectangles
    """
    filtered = rects

    if min_area is not None:
        filtered = [r for r in filtered if calculate_rectangle_area(r) >= min_area]

    if max_area is not None:
        filtered = [r for r in filtered if calculate_rectangle_area(r) <= max_area]

    return filtered


def get_rectangle_bounds(rect):
    """
    Get the bounding coordinates of a rectangle.

    Args:
        rect (tuple): Rectangle as (gx, gy, width, height)

    Returns:
        dict: Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max'
    """
    gx, gy, w, h = rect
    return {
        'x_min': gx,
        'x_max': gx + w - 1,
        'y_min': gy,
        'y_max': gy + h - 1
    }


def rectangles_overlap(rect1, rect2):
    """
    Check if two rectangles overlap.

    Args:
        rect1 (tuple): First rectangle as (gx, gy, width, height)
        rect2 (tuple): Second rectangle as (gx, gy, width, height)

    Returns:
        bool: True if rectangles overlap, False otherwise
    """
    b1 = get_rectangle_bounds(rect1)
    b2 = get_rectangle_bounds(rect2)

    # Check if one rectangle is completely to the left/right/above/below the other
    if b1['x_max'] < b2['x_min'] or b2['x_max'] < b1['x_min']:
        return False
    if b1['y_max'] < b2['y_min'] or b2['y_max'] < b1['y_min']:
        return False

    return True
