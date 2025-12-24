"""
OpenCV-based parallel video renderer for simulation logs.
Optimized for Numpy-based grids.
"""

import cv2
import numpy as np
import multiprocessing
from multiprocessing.pool import ThreadPool
import os
import sys
import time

try:
    from visualization.logger import SimulationLogger
except ImportError:
    print("Error: visualization.logger module not found.")
    sys.exit(1)


# === COLOR DEFINITIONS ===
COLORS_BGR = {
    'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0),
    'yellow': (0, 255, 255), 'magenta': (255, 0, 255), 'cyan': (255, 255, 0),
    'orange': (0, 165, 255), 'purple': (128, 0, 128), 'gray': (128, 128, 128),
    'pink': (203, 192, 255), 'darkgreen': (0, 100, 0), 'brown': (42, 42, 165),
    'navy': (128, 0, 0), 'lime': (0, 255, 0), 'salmon': (114, 128, 250),
    'teal': (128, 128, 0),
}
COLOR_LIST = list(COLORS_BGR.keys())

COLOR_UNEXPLORED = (240, 240, 240) # Match background
COLOR_FREE = (255, 255, 255)
COLOR_OBSTACLE = (0, 0, 0)
COLOR_BACKGROUND = (240, 240, 240)


class CoordinateTransform:
    """Handles world-to-pixel coordinate transformations."""
    def __init__(self, bounds, resolution, image_size, margin=50):
        self.bounds = bounds
        self.resolution = resolution
        self.image_width, self.image_height = image_size
        self.margin = margin

        world_width = bounds['x_max'] - bounds['x_min']
        world_height = bounds['y_max'] - bounds['y_min']
        available_width = self.image_width - 2 * margin
        available_height = self.image_height - 2 * margin

        self.scale = min(available_width / world_width, available_height / world_height)
        self.offset_x = margin + (available_width - world_width * self.scale) / 2
        self.offset_y = margin + (available_height - world_height * self.scale) / 2

    def world_to_pixel(self, x, y):
        px = int(self.offset_x + (x - self.bounds['x_min']) * self.scale)
        py = int(self.image_height - self.offset_y - (y - self.bounds['y_min']) * self.scale)
        return (px, py)

    def grid_to_world_raw(self, gx, gy):
        # Convert raw absolute grid index to world
        x = self.bounds['x_min'] + (gx + 0.5) * self.resolution
        y = self.bounds['y_min'] + (gy + 0.5) * self.resolution
        return x, y

    def world_distance_to_pixels(self, distance):
        return int(distance * self.scale)


# === DRAWING PRIMITIVES (Keep existing) ===
def draw_triangle(img, center, size, angle, color):
    cx, cy = center
    pts = np.array([[size, 0], [-size * 0.6, -size * 0.5], [-size * 0.6, size * 0.5]], dtype=np.float32)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = pts @ rot.T
    pts = pts + np.array([cx, cy])
    pts = pts.astype(np.int32)
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, (0, 0, 0), 1)

def draw_arrow(img, start, direction, length, color):
    sx, sy = start
    dx, dy = direction
    norm = np.sqrt(dx*dx + dy*dy)
    if norm < 1e-6: return
    dx, dy = dx / norm, dy / norm
    ex, ey = int(sx + dx * length), int(sy - dy * length)
    cv2.arrowedLine(img, (int(sx), int(sy)), (ex, ey), color, 2, tipLength=0.3)

def draw_x_marker(img, center, size, color):
    cx, cy = center
    s = size
    cv2.line(img, (cx - s, cy - s), (cx + s, cy + s), (255, 255, 255), 4)
    cv2.line(img, (cx - s, cy + s), (cx + s, cy - s), (255, 255, 255), 4)
    cv2.line(img, (cx - s, cy - s), (cx + s, cy + s), color, 2)
    cv2.line(img, (cx - s, cy + s), (cx + s, cy - s), color, 2)


# === FAST RENDERER ===

def render_occupancy_grid(frame, metadata, transform, panel_size):
    """Render occupancy grid panel using fast image resizing."""
    width, height = panel_size
    img = np.full((height, width, 3), COLOR_BACKGROUND, dtype=np.uint8)

    # 1. FAST GRID RENDERING
    if isinstance(frame['occupancy_grid'], tuple):
        # Optimized path: Numpy array
        grid_arr, (off_x, off_y) = frame['occupancy_grid']
        gh, gw = grid_arr.shape

        # Create colored grid image
        # Initialize with background color
        grid_img = np.full((gh, gw, 3), COLOR_UNEXPLORED, dtype=np.uint8)

        # Vectorized coloring
        grid_img[grid_arr == 1] = COLOR_FREE
        grid_img[grid_arr == 2] = COLOR_OBSTACLE

        # Calculate target pixel position
        # Grid bottom-left in world coords
        wx0, wy0 = transform.grid_to_world_raw(off_x, off_y)
        # Grid top-right in world coords
        wx1, wy1 = transform.grid_to_world_raw(off_x + gw, off_y + gh)

        # Convert to pixels
        # Note: Y-axis is flipped (0 is top in pixels, bottom in world)
        px0, py1_pixel = transform.world_to_pixel(wx0, wy0) # Bottom-left world -> Bottom-left pixel
        px1, py0_pixel = transform.world_to_pixel(wx1, wy1) # Top-right world -> Top-right pixel

        # Target rectangle on canvas
        target_x = px0
        target_y = py0_pixel
        target_w = px1 - px0
        target_h = py1_pixel - py0_pixel

        if target_w > 0 and target_h > 0:
            # Flip grid image vertically because world Y goes up but image index Y goes down
            grid_img_flipped = cv2.flip(grid_img, 0)

            # Resize using Nearest Neighbor (fast & keeps sharp edges)
            resized_grid = cv2.resize(grid_img_flipped, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            # Blit onto canvas (handling boundary clipping)
            y_start = max(0, target_y)
            y_end = min(height, target_y + target_h)
            x_start = max(0, target_x)
            x_end = min(width, target_x + target_w)

            # Crop source if target is partially out of bounds
            src_y_start = max(0, -target_y)
            src_y_end = src_y_start + (y_end - y_start)
            src_x_start = max(0, -target_x)
            src_x_end = src_x_start + (x_end - x_start)

            if (y_end > y_start) and (x_end > x_start):
                img[y_start:y_end, x_start:x_end] = resized_grid[src_y_start:src_y_end, src_x_start:src_x_end]

    else:
        # Slow fallback for legacy dicts
        for cell_key, value in frame['occupancy_grid'].items():
            gx, gy = cell_key
            px, py = transform.world_to_pixel(*transform.grid_to_world_raw(gx, gy))
            s = max(1, int(transform.scale * metadata['grid_resolution'])) // 2
            color = COLOR_FREE if value == 1 else COLOR_OBSTACLE
            cv2.rectangle(img, (px-s, py-s), (px+s, py+s), color, -1)

    # 2. Draw frontiers on main map
    if frame['frontiers']:
        for frontier in frame['frontiers']:
            px, py = transform.world_to_pixel(frontier['pos'][0], frontier['pos'][1])
            cv2.circle(img, (px, py), 6, (0, 255, 255), -1)
            cv2.circle(img, (px, py), 6, (0, 0, 255), 2)

    # 3. Draw robot information
    for i, robot_state in enumerate(frame['robots']):
        color = COLORS_BGR[COLOR_LIST[i % len(COLOR_LIST)]]

        # Trajectory
        if len(robot_state['trajectory']) > 1:
            pts = [transform.world_to_pixel(p[0], p[1]) for p in robot_state['trajectory']]
            cv2.polylines(img, [np.array(pts)], False, color, 1, cv2.LINE_AA)

        # Planned path
        if robot_state['path'] and len(robot_state['path']) > 1:
            path_pts = []
            for p in robot_state['path']:
                wx, wy = transform.grid_to_world_raw(p[0], p[1])
                px, py = transform.world_to_pixel(wx, wy)
                path_pts.append([px, py])
            cv2.polylines(img, [np.array(path_pts)], False, color, 2, cv2.LINE_AA)

        # Robot position
        pos = robot_state['position']
        px = transform.world_to_pixel(pos[0], pos[1])
        draw_triangle(img, px, 12, -robot_state['orientation'], color)

        # Exploration direction arrow
        exp_dir = robot_state['exploration_direction']
        arrow_len = transform.world_distance_to_pixels(1.5)
        draw_arrow(img, px, exp_dir, arrow_len, color)

        # Goal marker
        if robot_state['goal']:
            g_px = transform.world_to_pixel(robot_state['goal'][0], robot_state['goal'][1])
            draw_x_marker(img, g_px, 8, color)

        # Home position
        if i < len(metadata['robot_home_positions']):
            home = metadata['robot_home_positions'][i]
            home_px = transform.world_to_pixel(home[0], home[1])
            cv2.rectangle(img, (home_px[0] - 6, home_px[1] - 6), (home_px[0] + 6, home_px[1] + 6), color, -1)
            cv2.rectangle(img, (home_px[0] - 6, home_px[1] - 6), (home_px[0] + 6, home_px[1] + 6), (255, 255, 255), 2)

    # 4. Draw title and info
    status = "RETURNING HOME" if frame.get('returning_home', False) else "EXPLORING"
    title = f"Occupancy Grid | Step {frame['step']} | {status}"
    cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # Coverage info with background
    coverage_text = f"Coverage: {frame['coverage']:.1f}%"
    cv2.rectangle(img, (8, 35), (180, 65), (200, 220, 255), -1)
    cv2.rectangle(img, (8, 35), (180, 65), (0, 0, 0), 1)
    cv2.putText(img, coverage_text, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return img


def render_frontier_map(frame, metadata, transform, panel_size):
    """Render frontier detection panel."""
    width, height = panel_size
    img = np.full((height, width, 3), COLOR_BACKGROUND, dtype=np.uint8)

    # Simply draw dots for frontiers - fast enough
    if frame['frontiers']:
        for frontier in frame['frontiers']:
            px, py = transform.world_to_pixel(frontier['pos'][0], frontier['pos'][1])
            cv2.circle(img, (px, py), 5, (0, 255, 255), -1)
            cv2.circle(img, (px, py), 5, (0, 0, 255), 1)

    # Draw robots
    for i, robot in enumerate(frame['robots']):
        px, py = transform.world_to_pixel(robot['position'][0], robot['position'][1])
        color = COLORS_BGR[COLOR_LIST[i % len(COLOR_LIST)]]
        draw_triangle(img, (px, py), 10, -robot['orientation'], color)

    cv2.putText(img, f"Frontiers: {len(frame['frontiers'])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    return img


def render_coverage_graph(frames, current_frame_idx, panel_size, max_steps):
    """Coverage progress graph with labels and grid."""
    width, height = panel_size
    img = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)

    # Margins for labels
    margin_left = 60
    margin_right = 20
    margin_top = 40
    margin_bottom = 40

    gw = width - margin_left - margin_right
    gh = height - margin_top - margin_bottom

    # Origin and axes
    origin = (margin_left, height - margin_bottom)
    x_end = (width - margin_right, height - margin_bottom)
    y_end = (margin_left, margin_top)

    # Draw axes
    cv2.line(img, origin, x_end, (0, 0, 0), 2)
    cv2.line(img, origin, y_end, (0, 0, 0), 2)

    # Get data up to current frame
    steps = [f['step'] for f in frames[:current_frame_idx + 1]]
    coverages = [f['coverage'] for f in frames[:current_frame_idx + 1]]

    if not steps:
        return img

    # Calculate scale
    max_s = max(100, max_steps)
    x_scale = gw / max_s
    y_scale = gh / 100.0

    # Draw horizontal grid lines and Y-axis labels
    for pct in [25, 50, 75, 100]:
        y = int(origin[1] - pct * y_scale)
        cv2.line(img, (margin_left, y), (width - margin_right, y), (220, 220, 220), 1)
        cv2.putText(img, f"{pct}%", (5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Draw vertical grid lines and X-axis labels
    step_interval = max(500, (max_s // 4 // 500) * 500)
    for s in range(0, max_s + 1, step_interval):
        x = int(margin_left + s * x_scale)
        cv2.line(img, (x, height - margin_bottom), (x, height - margin_bottom + 5), (0, 0, 0), 1)
        cv2.putText(img, str(s), (x - 15, height - margin_bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    # Plot coverage line with fill
    if len(steps) > 1:
        # Downsample for performance (plot every 5th point)
        step_indices = range(0, len(steps), max(1, len(steps) // 200))
        points = []
        for i in step_indices:
            x = int(margin_left + steps[i] * x_scale)
            y = int(origin[1] - coverages[i] * y_scale)
            points.append([x, y])

        # Add final point if not included
        if step_indices[-1] != len(steps) - 1:
            x = int(margin_left + steps[-1] * x_scale)
            y = int(origin[1] - coverages[-1] * y_scale)
            points.append([x, y])

        points = np.array(points, dtype=np.int32)

        # Fill under the line
        fill_points = np.vstack([
            points,
            [[points[-1, 0], origin[1]], [points[0, 0], origin[1]]]
        ])
        cv2.fillPoly(img, [fill_points], (200, 220, 255))

        # Draw the line
        cv2.polylines(img, [points], False, (255, 0, 0), 2, cv2.LINE_AA)

        # Current value indicator
        current_coverage = coverages[-1]
        y_current = int(origin[1] - current_coverage * y_scale)
        cv2.line(img, (margin_left, y_current), (width - margin_right, y_current),
                 (0, 0, 255), 1, cv2.LINE_AA)

    # Draw axis labels
    cv2.putText(img, "Coverage Progress", (width // 2 - 80, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Simulation Step", (width // 2 - 50, height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Y-axis label (rotated text simulated with vertical text)
    cv2.putText(img, "Coverage (%)", (5, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return img


def render_single_frame(frame_idx, frames, metadata, output_size, max_steps):
    """Composite frame renderer."""
    frame = frames[frame_idx]
    w, h = output_size

    # Layout: Occupancy (Left), Frontier (Top Right), Coverage (Bottom Right)
    main_w = int(w * 0.6)
    side_w = w - main_w
    side_h = h // 2

    transform = CoordinateTransform(metadata['map_bounds'], metadata['grid_resolution'], (main_w, h))
    side_transform = CoordinateTransform(metadata['map_bounds'], metadata['grid_resolution'], (side_w, side_h))

    # Render Panels
    occ = render_occupancy_grid(frame, metadata, transform, (main_w, h))
    front = render_frontier_map(frame, metadata, side_transform, (side_w, side_h))
    cov = render_coverage_graph(frames, frame_idx, (side_w, side_h), max_steps)

    # Composite
    final = np.zeros((h, w, 3), dtype=np.uint8)
    final[0:h, 0:main_w] = occ
    final[0:side_h, main_w:w] = front
    final[side_h:h, main_w:w] = cov

    # Dividers
    cv2.line(final, (main_w, 0), (main_w, h), (100, 100, 100), 2)
    cv2.line(final, (main_w, side_h), (w, side_h), (100, 100, 100), 2)

    # Title bar
    env_config = metadata['env_config']
    title = f"Simulation: {env_config['env_type']} {env_config['maze_size']} | {env_config['num_robots']} robots | Frame {frame_idx}/{len(frames)-1}"
    cv2.rectangle(final, (0, 0), (w, 30), (50, 50, 50), -1)
    cv2.putText(final, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return final


# === PARALLEL WORKERS ===

# Global variables for ThreadPool workers (shared across threads)
_frames = None
_meta = None
_size = None
_max = None

def _worker_render(idx):
    return render_single_frame(idx, _frames, _meta, _size, _max)

def print_progress_bar(current, total, start_time, bar_length=40):
    """Print a progress bar with statistics."""
    elapsed = time.time() - start_time
    progress = current / total
    fps = current / elapsed if elapsed > 0 else 0
    eta = (elapsed / current) * (total - current) if current > 0 else 0

    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)

    print(f'\r  [{bar}] {current}/{total} ({progress*100:.0f}%) | {fps:.1f} fps | ETA: {eta:.0f}s    ', end='', flush=True)

def render_video_from_log(log_filepath, num_workers=None):
    """Render video from log file using ThreadPool for instant startup."""
    if num_workers is None: num_workers = 4

    # 1. Load Data ONCE
    print(f"Loading: {log_filepath}")
    data = SimulationLogger.load(log_filepath)
    frames = data['frames']
    metadata = data['metadata']
    max_steps = frames[-1]['step']

    print(f"  Loaded {len(frames)} frames")
    print(f"  Environment: {metadata['env_config']['env_type']} {metadata['env_config']['maze_size']}")
    print(f"  Robots: {metadata['env_config']['num_robots']}")

    # 2. Set Globals (Threads share these instantly)
    global _frames, _meta, _size, _max
    _frames = frames
    _meta = metadata
    _size = (1280, 720)
    _max = max_steps

    out_path = log_filepath.replace('.npz', '.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, 30, _size)

    print(f"  Rendering to: {out_path}")
    print(f"  Using {num_workers} parallel threads...")

    start_time = time.time()

    # 3. Use ThreadPool instead of Pool
    # No initializer needed because threads share memory
    with ThreadPool(num_workers) as pool:
        for i, img in enumerate(pool.imap(_worker_render, range(len(frames)))):
            writer.write(img)
            if i % 10 == 0 or i == len(frames) - 1:
                print_progress_bar(i + 1, len(frames), start_time)

    writer.release()

    elapsed = time.time() - start_time
    file_size = os.path.getsize(out_path) / (1024 * 1024)

    print(f"\n\n  ✓ Video saved: {out_path}")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Render time: {elapsed:.1f}s ({len(frames)/elapsed:.1f} fps)")
    print("  Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file")
    parser.add_argument("-j", "--jobs", type=int, default=4)
    args = parser.parse_args()
    render_video_from_log(args.log_file, args.jobs)
