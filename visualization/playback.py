"""
Offline Playback and Video Generation for Simulation Logs

Usage:
    Interactive viewer:  python playback.py <log_file.npz>
    Generate video:      python playback.py <log_file.npz> --video output.mp4
    Both:                python playback.py <log_file.npz> --video output.mp4 --interactive
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
import argparse
import os
import sys

# Import the logger to use its reconstruction logic
try:
    from visualization.logger import SimulationLogger
except ImportError:
    print("Error: visualization.logger module not found.")
    sys.exit(1)


class SimulationPlayback:
    """Plays back logged simulation data with interactive controls."""
    
    def __init__(self, log_filepath):
        # Use SimulationLogger.load to handle delta reconstruction
        self.data = SimulationLogger.load(log_filepath)
        self.metadata = self.data['metadata']
        self.frames = self.data['frames']
        self.num_frames = self.data['num_frames']
        
        print(f"Loaded {self.num_frames} frames")
        print(f"Environment: {self.metadata['env_config']}")
        print(f"Robots: {self.metadata['num_robots']}")
        
        self.current_frame_idx = 0
        self.playing = False
        self.playback_speed = 1.0
        
        # Color names for visualization (matching run_sim.py)
        self.color_names = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 
                           'orange', 'purple', 'gray', 'pink', 'darkgreen', 'brown', 
                           'navy', 'lime', 'salmon', 'teal']
        
    def setup_figure(self):
        """Setup the matplotlib figure with same layout as realtime viz."""
        self.fig = plt.figure(figsize=(18, 14))
        gs = self.fig.add_gridspec(3, 2, height_ratios=[3, 1, 0.3], hspace=0.25, wspace=0.2)
        
        self.axes = {
            'grid': self.fig.add_subplot(gs[0, 0]),
            'frontier': self.fig.add_subplot(gs[0, 1]),
            'coverage': self.fig.add_subplot(gs[1, :]),
        }
        
        # Add slider and buttons
        self.ax_slider = self.fig.add_subplot(gs[2, 0])
        self.ax_buttons = self.fig.add_subplot(gs[2, 1])
        self.ax_buttons.axis('off')
        
        self.slider = Slider(
            self.ax_slider, 'Frame', 0, self.num_frames - 1,
            valinit=0, valstep=1
        )
        self.slider.on_changed(self.on_slider_change)
        
        # Buttons
        btn_width = 0.08
        btn_height = 0.04
        btn_y = 0.02
        
        self.btn_play = Button(
            plt.axes([0.55, btn_y, btn_width, btn_height]), 'Play'
        )
        self.btn_play.on_clicked(self.on_play_click)
        
        self.btn_pause = Button(
            plt.axes([0.64, btn_y, btn_width, btn_height]), 'Pause'
        )
        self.btn_pause.on_clicked(self.on_pause_click)
        
        self.btn_reset = Button(
            plt.axes([0.73, btn_y, btn_width, btn_height]), 'Reset'
        )
        self.btn_reset.on_clicked(self.on_reset_click)
        
        self.btn_faster = Button(
            plt.axes([0.82, btn_y, btn_width, btn_height]), 'Faster'
        )
        self.btn_faster.on_clicked(self.on_faster_click)
        
        self.btn_slower = Button(
            plt.axes([0.91, btn_y, btn_width, btn_height]), 'Slower'
        )
        self.btn_slower.on_clicked(self.on_slower_click)
        
        env_config = self.metadata['env_config']
        title = f"Simulation Playback: {env_config['env_type']} {env_config['maze_size']}"
        self.fig.suptitle(title, fontsize=14, fontweight='bold')
        
    def on_slider_change(self, val):
        """Handle slider change."""
        self.current_frame_idx = int(val)
        self.render_frame(self.current_frame_idx)
        
    def on_play_click(self, event):
        """Start playback."""
        self.playing = True
        
    def on_pause_click(self, event):
        """Pause playback."""
        self.playing = False
        
    def on_reset_click(self, event):
        """Reset to beginning."""
        self.current_frame_idx = 0
        self.slider.set_val(0)
        self.render_frame(0)
        
    def on_faster_click(self, event):
        """Increase playback speed."""
        self.playback_speed = min(10.0, self.playback_speed * 2)
        print(f"Playback speed: {self.playback_speed}x")
        
    def on_slower_click(self, event):
        """Decrease playback speed."""
        self.playback_speed = max(0.1, self.playback_speed / 2)
        print(f"Playback speed: {self.playback_speed}x")
        
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates."""
        x = self.metadata['map_bounds']['x_min'] + (grid_x + 0.5) * self.metadata['grid_resolution']
        y = self.metadata['map_bounds']['y_min'] + (grid_y + 0.5) * self.metadata['grid_resolution']
        return (x, y)
    
    def render_frame(self, frame_idx):
        """Render a single frame."""
        if frame_idx >= len(self.frames):
            return
            
        frame = self.frames[frame_idx]
        
        for ax in self.axes.values():
            ax.clear()
            
        # === Occupancy Grid ===
        ax_grid = self.axes['grid']
        bounds = self.metadata['map_bounds']
        resolution = self.metadata['grid_resolution']
        
        grid_x = int((bounds['x_max'] - bounds['x_min']) / resolution)
        grid_y = int((bounds['y_max'] - bounds['y_min']) / resolution)
        grid_image = np.ones((grid_y, grid_x, 3)) * 0.7
        
        # Parse occupancy grid (tuple keys)
        for cell_key, value in frame['occupancy_grid'].items():
            gx, gy = cell_key

            if 0 <= gx < grid_x and 0 <= gy < grid_y:
                if value == 1:
                    grid_image[gy, gx] = [1, 1, 1]  # White for free
                elif value == 2:
                    grid_image[gy, gx] = [0, 0, 0]  # Black for obstacle
                    
        extent = [bounds['x_min'], bounds['x_max'], bounds['y_min'], bounds['y_max']]
        ax_grid.imshow(grid_image, origin='lower', extent=extent, interpolation='nearest')
        
        # Draw robots
        for i, robot_state in enumerate(frame['robots']):
            color = self.color_names[i % len(self.color_names)]
            pos = robot_state['position']
            
            # Draw trajectory
            if robot_state['trajectory']:
                traj = np.array(robot_state['trajectory'])
                ax_grid.plot(traj[:, 0], traj[:, 1], c=color, linewidth=1.5, alpha=0.6)
            
            # Draw planned path
            if robot_state['path']:
                path_world = [self.grid_to_world(p[0], p[1]) for p in robot_state['path']]
                path_arr = np.array(path_world)
                ax_grid.plot(path_arr[:, 0], path_arr[:, 1], c=color, linestyle=':', linewidth=2)
            
            # Draw robot position
            ax_grid.scatter(pos[0], pos[1], c=color, s=100, marker='^',
                          edgecolors='black', linewidths=1.5, zorder=5)
            
            # Draw exploration direction
            exp_dir = robot_state['exploration_direction']
            arrow_len = 1.5
            ax_grid.arrow(pos[0], pos[1], exp_dir[0] * arrow_len, exp_dir[1] * arrow_len,
                         head_width=0.3, head_length=0.2, fc=color, ec='black',
                         alpha=0.7, zorder=6)
            
            # Draw goal
            if robot_state['goal']:
                goal = robot_state['goal']
                ax_grid.scatter(goal[0], goal[1], c=color, s=150,
                              marker='X', edgecolors='white', linewidths=2, zorder=6)
            
            # Draw home position
            home = self.metadata['robot_home_positions'][i]
            ax_grid.scatter(home[0], home[1], c=color, s=100, marker='s',
                          edgecolors='white', linewidths=2, zorder=4)
        
        status = "RETURNING HOME" if frame['returning_home'] else "EXPLORING"
        ax_grid.set_title(f'Occupancy Grid | Step {frame["step"]} | Status: {status}')
        ax_grid.set_aspect('equal')
        ax_grid.grid(True, alpha=0.3)
        
        # Coverage text
        ax_grid.text(0.02, 0.98, f'Coverage: {frame["coverage"]:.1f}%',
                    transform=ax_grid.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        # === Frontier Map ===
        ax_frontier = self.axes['frontier']
        
        # Draw explored cells
        if frame['explored_cells']:
            explored_points = []
            for cell in frame['explored_cells']:
                cell_tuple = tuple(cell)
                if frame['occupancy_grid'].get(cell_tuple) == 1:
                    x, y = self.grid_to_world(cell[0], cell[1])
                    explored_points.append([x, y])

            if explored_points:
                explored_arr = np.array(explored_points)
                ax_frontier.scatter(explored_arr[:, 0], explored_arr[:, 1],
                                  c='lightblue', s=3, alpha=0.4, marker='s')
        
        # Draw obstacles
        if frame['obstacle_cells']:
            obstacle_points = [self.grid_to_world(c[0], c[1]) for c in frame['obstacle_cells']]
            obstacle_arr = np.array(obstacle_points)
            ax_frontier.scatter(obstacle_arr[:, 0], obstacle_arr[:, 1],
                              c='black', s=3, marker='s')
        
        # Draw frontiers
        if frame['frontiers']:
            frontier_points = [f['pos'] for f in frame['frontiers']]
            frontier_arr = np.array(frontier_points)
            ax_frontier.scatter(frontier_arr[:, 0], frontier_arr[:, 1],
                              c='yellow', s=50, marker='o', edgecolors='red',
                              linewidths=2, zorder=10)
        
        # Draw robots on frontier map
        for i, robot_state in enumerate(frame['robots']):
            color = self.color_names[i % len(self.color_names)]
            pos = robot_state['position']
            ax_frontier.scatter(pos[0], pos[1], c=color, s=150, marker='^',
                              edgecolors='black', linewidths=2, zorder=6)
        
        ax_frontier.set_xlim(bounds['x_min'] - 5, bounds['x_max'] + 5)
        ax_frontier.set_ylim(bounds['y_min'] - 5, bounds['y_max'] + 5)
        ax_frontier.set_aspect('equal')
        ax_frontier.grid(True, alpha=0.3)
        num_frontiers = len(frame['frontiers']) if frame['frontiers'] else 0
        ax_frontier.set_title(f'Frontier Detection ({num_frontiers} targets)')
        ax_frontier.set_xlabel('X (meters)')
        ax_frontier.set_ylabel('Y (meters)')
        
        # === Coverage Graph ===
        ax_coverage = self.axes['coverage']
        
        # Build coverage history up to current frame
        steps = []
        coverages = []
        for i, f in enumerate(self.frames[:frame_idx + 1]):
            steps.append(f['step'])
            coverages.append(f['coverage'])
        
        if steps:
            ax_coverage.plot(steps, coverages, linewidth=2, color='blue')
            ax_coverage.fill_between(steps, coverages, alpha=0.3, color='blue')
            ax_coverage.axhline(y=frame['coverage'], color='red', linestyle='--', alpha=0.5)
        
        ax_coverage.set_xlabel('Simulation Step')
        ax_coverage.set_ylabel('Coverage (%)')
        ax_coverage.set_title(f'Coverage Progress | Speed: {self.playback_speed}x')
        ax_coverage.grid(True, alpha=0.3)
        ax_coverage.set_ylim(0, 100)
        
        # Set x-axis limit based on total steps
        if self.frames:
            max_step = self.frames[-1]['step']
            ax_coverage.set_xlim(0, max(2000, max_step))
        
        self.fig.canvas.draw_idle()
        
    def run_interactive(self):
        """Run interactive playback viewer."""
        self.setup_figure()
        self.render_frame(0)
        
        def update(frame_num):
            if self.playing:
                self.current_frame_idx = min(
                    self.current_frame_idx + int(self.playback_speed),
                    self.num_frames - 1
                )
                self.slider.set_val(self.current_frame_idx)
                self.render_frame(self.current_frame_idx)
                
                if self.current_frame_idx >= self.num_frames - 1:
                    self.playing = False
            return []
        
        ani = animation.FuncAnimation(self.fig, update, interval=50, blit=True)
        plt.show()
        
    def generate_video(self, output_path, fps=30, dpi=100):
        """Generate MP4 video from logged data."""
        print(f"Generating video: {output_path}")
        print(f"  FPS: {fps}, DPI: {dpi}")
        
        self.setup_figure()
        
        # Remove interactive controls for video
        self.ax_slider.set_visible(False)
        self.btn_play.ax.set_visible(False)
        self.btn_pause.ax.set_visible(False)
        self.btn_reset.ax.set_visible(False)
        self.btn_faster.ax.set_visible(False)
        self.btn_slower.ax.set_visible(False)
        
        def animate(frame_idx):
            self.render_frame(frame_idx)
            if frame_idx % 10 == 0:
                print(f"  Rendering frame {frame_idx}/{self.num_frames}")
            return []
        
        ani = animation.FuncAnimation(
            self.fig, animate, frames=self.num_frames,
            interval=1000/fps, blit=True
        )
        
        # Save video
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
        ani.save(output_path, writer=writer, dpi=dpi)
        
        print(f"\nVideo saved to: {output_path}")
        print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(
        description='Playback simulation logs with interactive viewer or video export'
    )
    parser.add_argument('log_file', help='Path to simulation log file (.npz)')
    parser.add_argument('--video', '-v', metavar='OUTPUT', 
                       help='Generate MP4 video to specified path')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video frames per second (default: 30)')
    parser.add_argument('--dpi', type=int, default=100,
                       help='Video resolution DPI (default: 100)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Also show interactive viewer (when using --video)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)
        
    playback = SimulationPlayback(args.log_file)
    
    if args.video:
        playback.generate_video(args.video, fps=args.fps, dpi=args.dpi)
        if args.interactive:
            playback.run_interactive()
    else:
        playback.run_interactive()


if __name__ == "__main__":
    main()