"""
Real-time matplotlib visualization for multi-robot coverage mapping.
Handles interactive plotting, zooming, and visualization updates during simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pybullet as p
from mapping.decomposition import decompose_grid_to_rectangles


class RealtimeVisualizer:
    """Interactive real-time visualization using matplotlib."""

    # Color names for robot visualization
    ROBOT_COLOR_NAMES = [
        'red', 'green', 'blue', 'yellow', 'magenta', 'cyan',
        'orange', 'purple', 'gray', 'pink', 'darkgreen', 'brown',
        'navy', 'lime', 'salmon', 'teal'
    ]

    def __init__(self, mapper):
        """
        Initialize visualizer with reference to SimulationManager.

        Args:
            mapper: SimulationManager instance to visualize
        """
        self.mapper = mapper
        self.fig = None
        self.axes = None
        self.current_xlim = None
        self.current_ylim = None

    def setup(self):
        """Setup matplotlib figure and axes with interactive controls."""
        plt.ion()
        self.fig = plt.figure(figsize=(18, 14))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.25, wspace=0.2)

        self.axes = {
            'grid': self.fig.add_subplot(gs[0, 0]),
            'frontier': self.fig.add_subplot(gs[0, 1]),
            'coverage': self.fig.add_subplot(gs[1, :])
        }

        title = 'Multi-Robot Coverage Mapping\n(Scroll to Zoom, "P" to toggle decomposition)'
        self.fig.suptitle(title, fontsize=14, fontweight='bold')

        # Connect event handlers
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        plt.show(block=False)

    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == 'p' or event.key == 'P':
            self.mapper.show_partitions = not self.mapper.show_partitions
            print(f"\n[Viz] Rectangular Decomposition: {'ON' if self.mapper.show_partitions else 'OFF'}")

    def on_scroll(self, event):
        """Handle mouse scroll for zooming in/out."""
        if event.inaxes not in [self.axes['grid'], self.axes['frontier']]:
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

    def update(self, step):
        """
        Update visualization for current simulation step.

        Args:
            step: Current simulation step number
        """
        if self.fig is None:
            return

        # Clear all axes
        for ax in self.axes.values():
            ax.clear()

        # Render each panel
        self._render_occupancy_grid_panel(step)
        self._render_frontier_panel()
        self._render_coverage_panel(step)

        # Update display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def _render_occupancy_grid_panel(self, step):
        """Render the occupancy grid and robot state panel."""
        ax = self.axes['grid']
        mapper = self.mapper

        # Get Numpy grid directly (FAST PATH - no dictionary iteration)
        numpy_grid = mapper.grid_manager.get_numpy_grid()
        grid_h, grid_w = numpy_grid.shape

        # Create grid image (0=unknown, 1=free, 2=obstacle)
        grid_image = np.ones((grid_h, grid_w, 3)) * 0.7  # Unknown = gray

        # Vectorized operations (MUCH faster than dict iteration)
        free_mask = (numpy_grid == 1)
        obstacle_mask = (numpy_grid == 2)

        grid_image[free_mask] = [1, 1, 1]  # Free space = white
        grid_image[obstacle_mask] = [0, 0, 0]  # Obstacle = black

        extent = [mapper.map_bounds['x_min'], mapper.map_bounds['x_max'],
                 mapper.map_bounds['y_min'], mapper.map_bounds['y_max']]
        ax.imshow(grid_image, origin='lower', extent=extent, interpolation='nearest')

        # Draw rectangular decomposition if enabled
        if mapper.show_partitions:
            self._draw_partitions(ax)

        # Draw robot home positions
        self._draw_robot_home_positions(ax)

        # Draw robot trajectories and current states
        self._draw_robot_states(ax)

        # Set view bounds
        self._set_view_bounds(ax)

        # Configure axes
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Title with status
        status = "RETURNING HOME" if mapper.returning_home else "EXPLORING"
        decomp_status = " | [P]artitions: ON" if mapper.show_partitions else ""
        ax.set_title(f'Occupancy Grid | Status: {status}{decomp_status}')

        # Coverage info overlay
        coverage = mapper.calculate_coverage()
        ax.text(0.02, 0.98, f'Coverage: {coverage:.1f}%',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    def _draw_partitions(self, ax):
        """Draw rectangular decomposition overlay."""
        mapper = self.mapper
        # Get Numpy grid and offset
        numpy_grid = mapper.grid_manager.get_numpy_grid()
        offset_x, offset_y = mapper.grid_manager.get_grid_offset()
        rects = decompose_grid_to_rectangles(numpy_grid, offset_x, offset_y, max_rects=5)

        for gx, gy, w, h in rects:
            # Convert to world coordinates
            wx, wy = mapper.grid_to_world(gx, gy)
            rect_w = w * mapper.grid_manager.grid_resolution
            rect_h = h * mapper.grid_manager.grid_resolution

            # grid_to_world returns center, convert to bottom-left
            rect_x = wx - mapper.grid_manager.grid_resolution / 2
            rect_y = wy - mapper.grid_manager.grid_resolution / 2

            # Random pastel color
            color = np.random.rand(3) * 0.5 + 0.5
            rect_patch = patches.Rectangle(
                (rect_x, rect_y), rect_w, rect_h,
                linewidth=1, edgecolor='black', facecolor=(*color, 0.3)
            )
            ax.add_patch(rect_patch)

    def _draw_robot_home_positions(self, ax):
        """Draw robot home positions."""
        mapper = self.mapper

        for i, robot in enumerate(mapper.robots):
            color = self.ROBOT_COLOR_NAMES[i % len(self.ROBOT_COLOR_NAMES)]

            # Draw home position
            ax.scatter(robot.state.home_position[0], robot.state.home_position[1],
                      c=color, s=100, marker='s', edgecolors='white',
                      linewidths=2, zorder=4, label=f'R{i} Home' if i == 0 else '')

    def _draw_robot_states(self, ax):
        """Draw robot trajectories, paths, positions, and goals."""
        mapper = self.mapper

        for i, robot in enumerate(mapper.robots):
            color = self.ROBOT_COLOR_NAMES[i % len(self.ROBOT_COLOR_NAMES)]

            # Draw trajectory
            if robot.state.trajectory:
                traj = np.array(robot.state.trajectory)
                ax.plot(traj[:, 0], traj[:, 1], c=color, linewidth=1.5, alpha=0.6)

            # Draw planned path
            if robot.state.path:
                path_world = [mapper.grid_to_world(p[0], p[1]) for p in robot.state.path]
                path_arr = np.array(path_world)
                ax.plot(path_arr[:, 0], path_arr[:, 1], c=color, linestyle=':', linewidth=2)

            # Draw current position
            pos, _ = p.getBasePositionAndOrientation(robot.state.id)
            ax.scatter(pos[0], pos[1], c=color, s=100, marker='^',
                      edgecolors='black', linewidths=1.5, zorder=5)

            # Draw exploration direction arrow
            arrow_len = 1.5
            ax.arrow(pos[0], pos[1],
                    robot.state.exploration_direction[0] * arrow_len,
                    robot.state.exploration_direction[1] * arrow_len,
                    head_width=0.3, head_length=0.2, fc=color, ec='black',
                    alpha=0.7, zorder=6)

            # Draw goal marker
            if robot.state.goal is not None:
                ax.scatter(robot.state.goal[0], robot.state.goal[1], c=color, s=150,
                          marker='X', edgecolors='white', linewidths=2, zorder=6)

    def _render_frontier_panel(self):
        """Render the frontier detection panel."""
        ax = self.axes['frontier']
        mapper = self.mapper

        # Get Numpy grid and offset (FAST PATH)
        numpy_grid = mapper.grid_manager.get_numpy_grid()
        offset_x, offset_y = mapper.grid_manager.get_grid_offset()

        # Find explored free cells using vectorized operations
        free_coords = np.argwhere(numpy_grid == 1)
        if len(free_coords) > 0:
            # Convert grid coords to world coords (vectorized)
            grid_x = free_coords[:, 1] + offset_x  # column = x
            grid_y = free_coords[:, 0] + offset_y  # row = y
            world_x = mapper.map_bounds['x_min'] + (grid_x + 0.5) * mapper.grid_manager.grid_resolution
            world_y = mapper.map_bounds['y_min'] + (grid_y + 0.5) * mapper.grid_manager.grid_resolution

            ax.scatter(world_x, world_y, c='lightblue', s=3, alpha=0.4, marker='s')

        # Find obstacles using vectorized operations
        obstacle_coords = np.argwhere(numpy_grid == 2)
        if len(obstacle_coords) > 0:
            # Convert grid coords to world coords (vectorized)
            grid_x = obstacle_coords[:, 1] + offset_x
            grid_y = obstacle_coords[:, 0] + offset_y
            world_x = mapper.map_bounds['x_min'] + (grid_x + 0.5) * mapper.grid_manager.grid_resolution
            world_y = mapper.map_bounds['y_min'] + (grid_y + 0.5) * mapper.grid_manager.grid_resolution

            ax.scatter(world_x, world_y, c='black', s=3, marker='s')

        # Draw frontiers
        frontiers_data = mapper.detect_frontiers()
        if frontiers_data:
            frontier_points = [f['pos'] for f in frontiers_data]
            frontier_array = np.array(frontier_points)
            ax.scatter(frontier_array[:, 0], frontier_array[:, 1],
                      c='yellow', s=50, marker='o', edgecolors='red',
                      linewidths=2, label='Frontier Targets', zorder=10)

        # Draw robot positions
        for i, robot in enumerate(mapper.robots):
            color = self.ROBOT_COLOR_NAMES[i % len(self.ROBOT_COLOR_NAMES)]
            pos, _ = p.getBasePositionAndOrientation(robot.state.id)
            ax.scatter(pos[0], pos[1], c=color, s=150, marker='^',
                      edgecolors='black', linewidths=2, zorder=6)

        # Set view bounds
        self._set_view_bounds(ax)

        # Configure axes
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Frontier Detection\n({len(frontiers_data)} targets)')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')

    def _render_coverage_panel(self, step):
        """Render the coverage progress graph."""
        ax = self.axes['coverage']
        mapper = self.mapper

        if mapper.coverage_history:
            steps, coverage_values = zip(*mapper.coverage_history)
            ax.plot(steps, coverage_values, linewidth=2, color='blue')
            ax.fill_between(steps, coverage_values, alpha=0.3, color='blue')

            coverage = mapper.calculate_coverage()
            ax.axhline(y=coverage, color='red', linestyle='--', alpha=0.5)

        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Coverage (%)')
        ax.set_title('Coverage Progress (Direction Bias Enabled)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max(2000, step))

    def _set_view_bounds(self, ax):
        """Set axis limits with default bounds if not zoomed."""
        mapper = self.mapper

        if self.current_xlim is None or self.current_ylim is None:
            bounds_margin = 5
            self.current_xlim = [
                mapper.map_bounds['x_min'] - bounds_margin,
                mapper.map_bounds['x_max'] + bounds_margin
            ]
            self.current_ylim = [
                mapper.map_bounds['y_min'] - bounds_margin,
                mapper.map_bounds['y_max'] + bounds_margin
            ]

        ax.set_xlim(self.current_xlim)
        ax.set_ylim(self.current_ylim)

    def close(self):
        """Close the visualization window."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
