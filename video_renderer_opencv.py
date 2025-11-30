"""
OpenCV-based parallel video renderer for simulation logs.

This is a drop-in replacement for the matplotlib-based generate_video_from_log method.
It renders frames ~10-50x faster and supports parallel processing.

Usage:
    from video_renderer_opencv import render_video_from_log

    # Input: 'logs/sim_log_xxx.npz' -> Output: 'logs/sim_log_xxx.mp4' at 30fps
    render_video_from_log('logs/sim_log_xxx.npz')

    # Control parallelism (4 workers)
    render_video_from_log('logs/sim_log_xxx.npz', num_workers=4)

    # Sequential (no parallelism)
    render_video_from_log('logs/sim_log_xxx.npz', num_workers=1)

CLI Usage:
    python video_renderer_opencv.py logs/sim_log_xxx.npz
    python video_renderer_opencv.py logs/sim_log_xxx.npz -j 4

Output:
    - Videos are always saved to a 'logs' folder
    - If input is 'logs/file.npz', output is 'logs/file.mp4'
    - If input is 'file.npz', output is 'logs/file.mp4'
"""

import cv2
import numpy as np
import multiprocessing
from multiprocessing import Pool
import os
import sys
import time


# ============================================================================
# Color definitions (BGR format for OpenCV)
# ============================================================================
COLORS_BGR = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'magenta': (255, 0, 255),
    'cyan': (255, 255, 0),
    'orange': (0, 165, 255),
    'purple': (128, 0, 128),
    'gray': (128, 128, 128),
    'pink': (203, 192, 255),
    'darkgreen': (0, 100, 0),
    'brown': (42, 42, 165),
    'navy': (128, 0, 0),
    'lime': (0, 255, 0),
    'salmon': (114, 128, 250),
    'teal': (128, 128, 0),
}
COLOR_LIST = list(COLORS_BGR.keys())

# Grid colors
COLOR_UNEXPLORED = (180, 180, 180)  # Light gray
COLOR_FREE = (255, 255, 255)        # White
COLOR_OBSTACLE = (0, 0, 0)          # Black
COLOR_FRONTIER = (0, 255, 255)      # Yellow
COLOR_BACKGROUND = (240, 240, 240)  # Off-white


# ============================================================================
# Coordinate transform helpers
# ============================================================================
class CoordinateTransform:
    """Handles world-to-pixel coordinate transformations."""
    
    def __init__(self, bounds, resolution, image_size, margin=50):
        self.bounds = bounds
        self.resolution = resolution
        self.image_width, self.image_height = image_size
        self.margin = margin
        
        # Compute scale to fit world bounds into image (with margin)
        world_width = bounds['x_max'] - bounds['x_min']
        world_height = bounds['y_max'] - bounds['y_min']
        
        available_width = self.image_width - 2 * margin
        available_height = self.image_height - 2 * margin
        
        self.scale = min(available_width / world_width, available_height / world_height)
        
        # Center offset
        self.offset_x = margin + (available_width - world_width * self.scale) / 2
        self.offset_y = margin + (available_height - world_height * self.scale) / 2
        
    def world_to_pixel(self, x, y):
        """Convert world coordinates to pixel coordinates."""
        px = int(self.offset_x + (x - self.bounds['x_min']) * self.scale)
        # Flip y-axis (image origin is top-left)
        py = int(self.image_height - self.offset_y - (y - self.bounds['y_min']) * self.scale)
        return (px, py)
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid cell indices to world coordinates (cell center)."""
        x = self.bounds['x_min'] + (grid_x + 0.5) * self.resolution
        y = self.bounds['y_min'] + (grid_y + 0.5) * self.resolution
        return (x, y)
    
    def grid_to_pixel(self, grid_x, grid_y):
        """Convert grid cell to pixel coordinates."""
        wx, wy = self.grid_to_world(grid_x, grid_y)
        return self.world_to_pixel(wx, wy)
    
    def world_distance_to_pixels(self, distance):
        """Convert a world distance to pixels."""
        return int(distance * self.scale)


# ============================================================================
# Drawing primitives
# ============================================================================
def draw_triangle(img, center, size, angle, color, thickness=-1):
    """Draw a triangle (robot marker) at the given position and orientation."""
    cx, cy = center
    # Triangle points relative to center, pointing right (angle=0)
    pts = np.array([
        [size, 0],
        [-size * 0.6, -size * 0.5],
        [-size * 0.6, size * 0.5],
    ], dtype=np.float32)
    
    # Rotate
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = pts @ rot.T
    
    # Translate and convert to int
    pts = pts + np.array([cx, cy])
    pts = pts.astype(np.int32)
    
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, (0, 0, 0), max(1, thickness // 4))


def draw_arrow(img, start, direction, length, color, thickness=2):
    """Draw an arrow from start point in the given direction."""
    sx, sy = start
    dx, dy = direction
    # Normalize direction
    norm = np.sqrt(dx*dx + dy*dy)
    if norm < 1e-6:
        return
    dx, dy = dx / norm, dy / norm
    
    ex, ey = int(sx + dx * length), int(sy - dy * length)  # Flip y for image coords
    sx, sy = int(sx), int(sy)
    
    cv2.arrowedLine(img, (sx, sy), (ex, ey), color, thickness, tipLength=0.3)


def draw_x_marker(img, center, size, color, thickness=2):
    """Draw an X marker (for goals)."""
    cx, cy = center
    s = size
    cv2.line(img, (cx - s, cy - s), (cx + s, cy + s), color, thickness)
    cv2.line(img, (cx - s, cy + s), (cx + s, cy - s), color, thickness)
    # White outline
    cv2.line(img, (cx - s, cy - s), (cx + s, cy + s), (255, 255, 255), thickness + 2)
    cv2.line(img, (cx - s, cy + s), (cx + s, cy - s), (255, 255, 255), thickness + 2)
    cv2.line(img, (cx - s, cy - s), (cx + s, cy + s), color, thickness)
    cv2.line(img, (cx - s, cy + s), (cx + s, cy - s), color, thickness)


def draw_text_with_background(img, text, pos, font_scale=0.5, color=(0, 0, 0), 
                              bg_color=(255, 255, 255), thickness=1, padding=5):
    """Draw text with a background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = pos
    cv2.rectangle(img, 
                  (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + baseline + padding),
                  bg_color, -1)
    cv2.rectangle(img,
                  (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + baseline + padding),
                  (0, 0, 0), 1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


# ============================================================================
# Panel renderers
# ============================================================================
def render_occupancy_grid(frame, metadata, transform, panel_size):
    """Render the occupancy grid panel."""
    width, height = panel_size
    img = np.full((height, width, 3), COLOR_BACKGROUND, dtype=np.uint8)
    
    bounds = metadata['map_bounds']
    resolution = metadata['grid_resolution']
    
    # Calculate cell size in pixels
    cell_pixel_size = max(1, int(resolution * transform.scale))
    
    # Draw occupancy grid cells
    for cell_str, value in frame['occupancy_grid'].items():
        gx, gy = map(int, cell_str.split(','))
        px, py = transform.grid_to_pixel(gx, gy)
        
        half_size = cell_pixel_size // 2
        pt1 = (px - half_size, py - half_size)
        pt2 = (px + half_size, py + half_size)
        
        if value == 1:  # Free space
            cv2.rectangle(img, pt1, pt2, COLOR_FREE, -1)
        elif value == 2:  # Obstacle
            cv2.rectangle(img, pt1, pt2, COLOR_OBSTACLE, -1)
    
    # Draw robots
    for i, robot_state in enumerate(frame['robots']):
        color_name = COLOR_LIST[i % len(COLOR_LIST)]
        color = COLORS_BGR[color_name]
        pos = robot_state['position']
        
        # Trajectory
        if robot_state['trajectory'] and len(robot_state['trajectory']) > 1:
            traj_pixels = [transform.world_to_pixel(p[0], p[1]) for p in robot_state['trajectory']]
            traj_pixels = np.array(traj_pixels, dtype=np.int32)
            cv2.polylines(img, [traj_pixels], False, color, 1, cv2.LINE_AA)
        
        # Global graph edges
        nodes = robot_state['global_graph_nodes']
        for edge in robot_state['global_graph_edges']:
            n1, n2 = edge
            if n1 < len(nodes) and n2 < len(nodes):
                p1 = transform.world_to_pixel(nodes[n1][0], nodes[n1][1])
                p2 = transform.world_to_pixel(nodes[n2][0], nodes[n2][1])
                cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)
        
        # Planned path (dotted)
        if robot_state['path'] and len(robot_state['path']) > 1:
            for j in range(len(robot_state['path']) - 1):
                p1 = robot_state['path'][j]
                p2 = robot_state['path'][j + 1]
                px1, py1 = transform.grid_to_pixel(p1[0], p1[1])
                px2, py2 = transform.grid_to_pixel(p2[0], p2[1])
                # Draw dashed line
                cv2.line(img, (px1, py1), (px2, py2), color, 2, cv2.LINE_AA)
        
        # Robot marker (triangle)
        robot_px = transform.world_to_pixel(pos[0], pos[1])
        orientation = robot_state['orientation']
        draw_triangle(img, robot_px, 12, -orientation, color)  # Negative because y is flipped
        
        # Exploration direction arrow
        exp_dir = robot_state['exploration_direction']
        arrow_len = transform.world_distance_to_pixels(1.5)
        draw_arrow(img, robot_px, exp_dir, arrow_len, color, 2)
        
        # Goal marker
        if robot_state['goal']:
            goal = robot_state['goal']
            goal_px = transform.world_to_pixel(goal[0], goal[1])
            draw_x_marker(img, goal_px, 8, color, 2)
        
        # Home position (square)
        home = metadata['robot_home_positions'][i]
        home_px = transform.world_to_pixel(home[0], home[1])
        cv2.rectangle(img, 
                      (home_px[0] - 6, home_px[1] - 6),
                      (home_px[0] + 6, home_px[1] + 6),
                      color, -1)
        cv2.rectangle(img,
                      (home_px[0] - 6, home_px[1] - 6),
                      (home_px[0] + 6, home_px[1] + 6),
                      (255, 255, 255), 1)
    
    # Title and info
    status = "RETURNING HOME" if frame['returning_home'] else "EXPLORING"
    title = f"Occupancy Grid | Step {frame['step']} | {status}"
    cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Coverage info
    draw_text_with_background(img, f"Coverage: {frame['coverage']:.1f}%", 
                              (10, 50), font_scale=0.5, bg_color=(200, 220, 255))
    
    return img


def render_frontier_map(frame, metadata, transform, panel_size):
    """Render the frontier detection panel."""
    width, height = panel_size
    img = np.full((height, width, 3), COLOR_BACKGROUND, dtype=np.uint8)
    
    resolution = metadata['grid_resolution']
    cell_pixel_size = max(1, int(resolution * transform.scale))
    
    # Draw explored cells (light blue for free space)
    for cell in frame['explored_cells']:
        cell_key = f"{cell[0]},{cell[1]}"
        if frame['occupancy_grid'].get(cell_key) == 1:
            px, py = transform.grid_to_pixel(cell[0], cell[1])
            half_size = cell_pixel_size // 2
            cv2.rectangle(img, 
                          (px - half_size, py - half_size),
                          (px + half_size, py + half_size),
                          (255, 200, 150), -1)  # Light blue
    
    # Draw obstacles
    for cell in frame['obstacle_cells']:
        px, py = transform.grid_to_pixel(cell[0], cell[1])
        half_size = cell_pixel_size // 2
        cv2.rectangle(img,
                      (px - half_size, py - half_size),
                      (px + half_size, py + half_size),
                      COLOR_OBSTACLE, -1)
    
    # Draw frontiers
    if frame['frontiers']:
        for frontier in frame['frontiers']:
            pos = frontier['pos']
            px, py = transform.world_to_pixel(pos[0], pos[1])
            cv2.circle(img, (px, py), 8, COLOR_FRONTIER, -1)
            cv2.circle(img, (px, py), 8, (0, 0, 255), 2)  # Red outline
    
    # Draw robots
    for i, robot_state in enumerate(frame['robots']):
        color_name = COLOR_LIST[i % len(COLOR_LIST)]
        color = COLORS_BGR[color_name]
        pos = robot_state['position']
        robot_px = transform.world_to_pixel(pos[0], pos[1])
        draw_triangle(img, robot_px, 15, -robot_state['orientation'], color)
    
    # Title
    num_frontiers = len(frame['frontiers']) if frame['frontiers'] else 0
    title = f"Frontier Detection ({num_frontiers} targets)"
    cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    return img


def render_coverage_graph(frames, current_frame_idx, panel_size, max_steps=None):
    """Render the coverage progress graph."""
    width, height = panel_size
    img = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
    
    # Graph area with margins
    margin_left = 60
    margin_right = 20
    margin_top = 40
    margin_bottom = 40
    
    graph_width = width - margin_left - margin_right
    graph_height = height - margin_top - margin_bottom
    
    # Draw axes
    origin = (margin_left, height - margin_bottom)
    x_end = (width - margin_right, height - margin_bottom)
    y_end = (margin_left, margin_top)
    
    cv2.line(img, origin, x_end, (0, 0, 0), 2)  # X-axis
    cv2.line(img, origin, y_end, (0, 0, 0), 2)  # Y-axis
    
    # Get data up to current frame
    steps = [f['step'] for f in frames[:current_frame_idx + 1]]
    coverages = [f['coverage'] for f in frames[:current_frame_idx + 1]]
    
    if not steps:
        return img
    
    # Scale factors
    if max_steps is None:
        max_steps = max(2000, frames[-1]['step'])
    
    x_scale = graph_width / max_steps
    y_scale = graph_height / 100  # Coverage is 0-100%
    
    # Draw grid lines
    for pct in [25, 50, 75, 100]:
        y = int(origin[1] - pct * y_scale)
        cv2.line(img, (margin_left, y), (width - margin_right, y), (200, 200, 200), 1)
        cv2.putText(img, f"{pct}%", (5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Draw step markers
    step_interval = max(500, (max_steps // 4 // 500) * 500)
    for s in range(0, max_steps + 1, step_interval):
        x = int(margin_left + s * x_scale)
        cv2.line(img, (x, height - margin_bottom), (x, height - margin_bottom + 5), (0, 0, 0), 1)
        cv2.putText(img, str(s), (x - 15, height - margin_bottom + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
    # Plot coverage line and fill
    if len(steps) > 1:
        points = []
        for i in range(len(steps)):
            x = int(margin_left + steps[i] * x_scale)
            y = int(origin[1] - coverages[i] * y_scale)
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        # Fill area under curve
        fill_points = np.vstack([
            points,
            [[points[-1, 0], origin[1]], [points[0, 0], origin[1]]]
        ])
        cv2.fillPoly(img, [fill_points], (255, 200, 200))  # Light blue fill
        
        # Draw line
        cv2.polylines(img, [points], False, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Current value indicator
        current_coverage = coverages[-1]
        y_current = int(origin[1] - current_coverage * y_scale)
        cv2.line(img, (margin_left, y_current), (width - margin_right, y_current), 
                 (0, 0, 255), 1, cv2.LINE_AA)
    
    # Labels
    cv2.putText(img, "Coverage Progress", (width // 2 - 80, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Simulation Step", (width // 2 - 50, height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Rotate "Coverage (%)" for y-axis label - just put it at an angle
    cv2.putText(img, "Coverage", (5, height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return img


# ============================================================================
# Frame renderer (for parallel processing)
# ============================================================================
def render_single_frame(frame_idx, frames, metadata, output_size, max_steps):
    """
    Render a single frame to a numpy array.
    This function is designed to be called in parallel.
    """
    frame = frames[frame_idx]
    
    # Layout: 2x2 grid, bottom row is single panel
    # [Occupancy Grid] [Frontier Map]
    # [    Coverage Progress Graph    ]
    
    width, height = output_size
    panel_width = width // 2
    top_panel_height = int(height * 0.7)
    bottom_panel_height = height - top_panel_height
    
    # Create coordinate transforms for each map panel
    transform = CoordinateTransform(
        metadata['map_bounds'],
        metadata['grid_resolution'],
        (panel_width, top_panel_height),
        margin=40
    )
    
    # Render each panel
    occupancy_panel = render_occupancy_grid(frame, metadata, transform, (panel_width, top_panel_height))
    frontier_panel = render_frontier_map(frame, metadata, transform, (panel_width, top_panel_height))
    coverage_panel = render_coverage_graph(frames, frame_idx, (width, bottom_panel_height), max_steps)
    
    # Composite into final frame
    final_frame = np.full((height, width, 3), COLOR_BACKGROUND, dtype=np.uint8)
    
    # Top row
    final_frame[0:top_panel_height, 0:panel_width] = occupancy_panel
    final_frame[0:top_panel_height, panel_width:width] = frontier_panel
    
    # Bottom row
    final_frame[top_panel_height:height, 0:width] = coverage_panel
    
    # Draw panel borders
    cv2.line(final_frame, (panel_width, 0), (panel_width, top_panel_height), (100, 100, 100), 2)
    cv2.line(final_frame, (0, top_panel_height), (width, top_panel_height), (100, 100, 100), 2)
    
    # Title bar
    env_config = metadata['env_config']
    title = f"Simulation: {env_config['env_type']} {env_config['maze_size']} | {env_config['num_robots']} robots | Frame {frame_idx}/{len(frames)-1}"
    cv2.rectangle(final_frame, (0, 0), (width, 30), (50, 50, 50), -1)
    cv2.putText(final_frame, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return final_frame


# ============================================================================
# Main video rendering function
# ============================================================================
def _worker_init(log_filepath, shared_output_size, shared_max_steps):
    """Initialize worker process with shared data."""
    global _frames, _metadata, _output_size, _max_steps
    
    # Load data locally in the worker process
    # This prevents pickling huge objects across process boundaries
    data = np.load(log_filepath, allow_pickle=True)
    _metadata = data['metadata'][0]
    _frames = data['frames'].tolist() # Worker now owns this memory
    
    _output_size = shared_output_size
    _max_steps = shared_max_steps


def _worker_render(frame_idx):
    """Worker function that uses global shared data."""
    return render_single_frame(frame_idx, _frames, _metadata, _output_size, _max_steps)


def print_progress(current, total, start_time, bar_length=40):
    """Print a progress bar to the console."""
    elapsed = time.time() - start_time
    progress = current / total
    fps = current / elapsed if elapsed > 0 else 0
    eta = (elapsed / current) * (total - current) if current > 0 else 0
    
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    # \r returns to start of line, end='' prevents newline
    print(f'\r  [{bar}] {current}/{total} ({progress*100:.0f}%) | {fps:.1f} fps | ETA: {eta:.0f}s    ', end='', flush=True)


def render_video_from_log(log_filepath, num_workers=None):
    """
    Render a simulation log to an MP4 video using OpenCV.

    Args:
        log_filepath: Path to the .npz log file
        num_workers: Number of parallel workers (default: 4, use 1 for sequential)

    Returns:
        Path to the output video file
    """
    # Auto-generate output path - save to logs folder
    log_basename = os.path.basename(log_filepath).replace('.npz', '.mp4')
    log_dir = os.path.dirname(log_filepath)

    # If log is already in a 'logs' directory, use that; otherwise create 'logs' subdirectory
    if os.path.basename(log_dir) == 'logs':
        logs_dir = log_dir
    else:
        logs_dir = os.path.join(log_dir or '.', 'logs')

    # Create logs directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)

    output_path = os.path.join(logs_dir, log_basename)

    # Fixed parameters for consistency
    fps = 30
    output_size = (1280, 960)

    print(f"Loading log file: {log_filepath}")
    print(f"Output will be saved to: {output_path}")
    
    # Load log data (only for main process to get metadata/length)
    data = np.load(log_filepath, allow_pickle=True)
    metadata = data['metadata'][0]
    frames = data['frames'].tolist()
    num_frames = len(frames)
    
    print(f"  Loaded {num_frames} frames")
    print(f"  Environment: {metadata['env_config']['env_type']} {metadata['env_config']['maze_size']}")
    print(f"  Robots: {metadata['env_config']['num_robots']}")
    
    # Determine max steps for consistent x-axis scaling
    max_steps = max(2000, frames[-1]['step'])
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, output_size)
    
    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    
    start_time = time.time()

    if num_workers is None:
        num_workers = 4
    
    # On Windows, multiprocessing can be problematic - offer sequential fallback
    use_parallel = num_workers > 1
    
    if use_parallel:
        print(f"  Rendering with {num_workers} parallel workers...")
        print(f"  (If this hangs, try with -j 1 for sequential rendering)")
        
        try:
            # Use initializer to share data with workers (more Windows-friendly)
            # Process in batches for progress updates
            batch_size = min(50, num_frames)
            frames_written = 0
            
            with Pool(
                processes=num_workers,
                initializer=_worker_init,
                initargs=(log_filepath, output_size, max_steps) # Pass path, not huge data objects
            ) as pool:
                # Use imap for better progress tracking
                for frame_img in pool.imap(_worker_render, range(num_frames)):
                    video_writer.write(frame_img)
                    frames_written += 1
                    
                    if frames_written % 10 == 0 or frames_written == num_frames:
                        print_progress(frames_written, num_frames, start_time)
                        
        except Exception as e:
            print(f"\n  Parallel rendering failed: {e}")
            print("  Falling back to sequential rendering...")
            video_writer.release()
            # Recreate video writer
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, output_size)
            use_parallel = False
    
    if not use_parallel:
        print(f"  Rendering sequentially...")
        for i in range(num_frames):
            frame_img = render_single_frame(i, frames, metadata, output_size, max_steps)
            video_writer.write(frame_img)
            
            if i % 10 == 0 or i == num_frames - 1:
                print_progress(i + 1, num_frames, start_time)
    
    video_writer.release()
    print()  # Newline after progress bar
    
    elapsed = time.time() - start_time
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\n  Video saved: {output_path}")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Render time: {elapsed:.1f}s ({num_frames/elapsed:.1f} fps)")
    
    return output_path


# ============================================================================
# Convenience function for drop-in replacement
# ============================================================================
def generate_video_from_log_opencv(log_filepath, video_path=None, fps=30, dpi=100, num_workers=None):
    """
    Drop-in replacement for SubterraneanMapper.generate_video_from_log().

    The 'video_path', 'fps', and 'dpi' parameters are ignored (kept for API compatibility).
    Output will be the same as log_filepath with .mp4 extension at 30fps.
    """
    return render_video_from_log(log_filepath, num_workers=num_workers)


# ============================================================================
# CLI interface
# ============================================================================
if __name__ == "__main__":
    # Required for Windows multiprocessing support
    multiprocessing.freeze_support()

    import argparse

    parser = argparse.ArgumentParser(
        description="Render simulation log to MP4 video using OpenCV",
        epilog="Output will be saved as <log_file>.mp4 at 30fps with 1280x960 resolution"
    )
    parser.add_argument("log_file", help="Path to the .npz log file")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="Number of parallel workers (default: auto, use 1 for sequential)")

    args = parser.parse_args()

    render_video_from_log(args.log_file, num_workers=args.jobs)