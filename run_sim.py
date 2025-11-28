"""
Multi-robot coverage mapping simulation for subterranean maze environments
Uses the MazeEnvironment class from environment_generator to create procedurally generated mazes
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from environment_generator import MazeEnvironment

# Import the Robot class and MultiRobotMapper from the original file
import sys
sys.path.append(os.path.dirname(__file__))


class Robot:
    def __init__(self, robot_id, position, color):
        self.id = robot_id
        self.position = position
        self.color = color
        self.lidar_data = []
        self.trajectory = []
        self.goal = None
        self.manual_control = False

    def get_lidar_scan(self, num_rays=360, max_range=10):
        """Simulate lidar by casting rays in 360 degrees"""
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        ray_from = []
        ray_to = []

        for i in range(num_rays):
            angle = yaw + (2.0 * np.pi * i / num_rays)
            ray_from.append(pos)
            ray_to.append([
                pos[0] + max_range * np.cos(angle),
                pos[1] + max_range * np.sin(angle),
                pos[2]
            ])

        results = p.rayTestBatch(ray_from, ray_to)

        scan_points = []
        for i, result in enumerate(results):
            hit_fraction = result[2]
            if hit_fraction < 1.0:
                hit_position = result[3]
                scan_points.append((hit_position[0], hit_position[1]))

        self.lidar_data.extend(scan_points)
        self.trajectory.append((pos[0], pos[1]))
        return scan_points

    def move(self, linear_vel, angular_vel):
        """Move robot with linear and angular velocity"""
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        new_x = pos[0] + linear_vel * np.cos(yaw) * 0.1
        new_y = pos[1] + linear_vel * np.sin(yaw) * 0.1
        new_yaw = yaw + angular_vel * 0.1

        new_orn = p.getQuaternionFromEuler([0, 0, new_yaw])
        p.resetBasePositionAndOrientation(self.id, [new_x, new_y, 0.25], new_orn)

    def navigate_to_goal(self):
        """Navigate towards the goal position using simple proportional control"""
        if self.goal is None:
            return 0.0, 0.0

        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        dx = self.goal[0] - pos[0]
        dy = self.goal[1] - pos[1]
        distance = np.sqrt(dx**2 + dy**2)

        if distance < 0.5:
            self.goal = None
            return 0.0, 0.0

        desired_angle = np.arctan2(dy, dx)
        angle_diff = desired_angle - yaw
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        linear_vel = min(1.5, distance * 0.5)
        angular_vel = np.clip(angle_diff * 2.0, -1.0, 1.0)

        return linear_vel, angular_vel


class SubterraneanMapper:
    """Multi-robot mapper for subterranean maze environments"""

    def __init__(self, use_gui=True, maze_size=(10, 10), cell_size=2.0, env_seed=None):
        """
        Initialize the mapper with a maze environment

        Args:
            use_gui: Whether to show PyBullet GUI
            maze_size: Tuple (width, height) of maze in cells
            cell_size: Size of each maze cell in meters
            env_seed: Random seed for reproducible environments
        """
        # Create maze environment (which handles PyBullet connection internally)
        self.env = MazeEnvironment(
            maze_size=maze_size,
            cell_size=cell_size,
            wall_height=2.5,
            wall_thickness=0.2,
            gui=use_gui,
            seed=env_seed
        )

        # Generate and build the maze
        self.env.generate_maze()
        self.env.build_walls()

        # Store physics client reference
        self.physics_client = self.env.physics_client

        # Create robots
        self.robots = []
        self.create_robots()

        # Mapping data
        self.grid_resolution = 0.5
        self.occupancy_grid = {}
        self.explored_cells = set()
        self.obstacle_cells = set()

        # Calculate environment bounds from maze
        maze_world_width = self.env.maze_grid.shape[1] * self.env.cell_size / 2
        maze_world_height = self.env.maze_grid.shape[0] * self.env.cell_size / 2
        self.map_bounds = {
            'x_min': 0,
            'x_max': maze_world_width,
            'y_min': 0,
            'y_max': maze_world_height
        }

        # Coverage tracking
        total_x = int((self.map_bounds['x_max'] - self.map_bounds['x_min']) / self.grid_resolution)
        total_y = int((self.map_bounds['y_max'] - self.map_bounds['y_min']) / self.grid_resolution)
        self.total_cells = total_x * total_y
        self.coverage_history = []

        # Real-time visualization
        self.realtime_fig = None
        self.realtime_axes = None

    def create_robots(self):
        """Create robots at starting positions near the maze entrance"""
        # Get spawn position from maze environment
        spawn_pos = self.env.get_spawn_position()

        # Position robots near the entrance
        start_positions = [
            [spawn_pos[0] - 1.0, spawn_pos[1], 0.25],   # Red robot
            [spawn_pos[0], spawn_pos[1], 0.25],          # Green robot
            [spawn_pos[0] + 1.0, spawn_pos[1], 0.25]     # Blue robot
        ]

        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]  # Red, Green, Blue

        for i, (pos, color) in enumerate(zip(start_positions, colors)):
            # Create sphere robot
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.25)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.25, rgbaColor=color)

            robot_id = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=pos
            )

            # Keep robots stationary with constraint
            constraint = p.createConstraint(
                robot_id, -1, -1, -1, p.JOINT_FIXED,
                [0, 0, 0], [0, 0, 0], pos,
                childFrameOrientation=p.getQuaternionFromEuler([0, 0, 0])
            )
            p.changeConstraint(constraint, maxForce=50)

            robot = Robot(robot_id, pos, color)
            self.robots.append(robot)

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - self.map_bounds['x_min']) / self.grid_resolution)
        grid_y = int((y - self.map_bounds['y_min']) / self.grid_resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        x = self.map_bounds['x_min'] + (grid_x + 0.5) * self.grid_resolution
        y = self.map_bounds['y_min'] + (grid_y + 0.5) * self.grid_resolution
        return (x, y)

    def update_occupancy_grid(self, robot):
        """Update occupancy grid based on robot's latest lidar scan"""
        if not robot.lidar_data:
            return

        pos, _ = p.getBasePositionAndOrientation(robot.id)
        robot_grid = self.world_to_grid(pos[0], pos[1])

        self.occupancy_grid[robot_grid] = 1
        self.explored_cells.add(robot_grid)

        recent_scans = robot.lidar_data[-180:] if len(robot.lidar_data) > 180 else robot.lidar_data

        for hit_x, hit_y in recent_scans:
            obstacle_grid = self.world_to_grid(hit_x, hit_y)
            self.occupancy_grid[obstacle_grid] = 2
            self.obstacle_cells.add(obstacle_grid)

            self.bresenham_line(robot_grid[0], robot_grid[1],
                               obstacle_grid[0], obstacle_grid[1])

    def bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm to mark cells along a line as free"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            cell = (x, y)
            if cell not in self.obstacle_cells:
                self.occupancy_grid[cell] = 1
                self.explored_cells.add(cell)

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def detect_frontiers(self):
        """Detect frontier cells (boundary between explored and unexplored)"""
        frontiers = set()

        for cell in self.explored_cells:
            if self.occupancy_grid.get(cell) != 1:
                continue

            x, y = cell
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    neighbor = (x + dx, y + dy)

                    world_x, world_y = self.grid_to_world(neighbor[0], neighbor[1])
                    if not (self.map_bounds['x_min'] <= world_x <= self.map_bounds['x_max'] and
                           self.map_bounds['y_min'] <= world_y <= self.map_bounds['y_max']):
                        continue

                    if neighbor not in self.occupancy_grid:
                        frontiers.add(cell)
                        break

                if cell in frontiers:
                    break

        return frontiers

    def calculate_coverage(self):
        """Calculate percentage of explored area"""
        explored_count = len(self.explored_cells)
        coverage_percent = (explored_count / self.total_cells) * 100
        return coverage_percent

    def setup_realtime_visualization(self):
        """Setup interactive real-time visualization window"""
        plt.ion()
        self.realtime_fig = plt.figure(figsize=(15, 10))
        gs = self.realtime_fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        self.realtime_axes = {
            'grid': self.realtime_fig.add_subplot(gs[0, 0]),
            'frontier': self.realtime_fig.add_subplot(gs[0, 1]),
            'coverage': self.realtime_fig.add_subplot(gs[1, :])
        }

        title = 'Subterranean Maze Mapping\n(Click on grid map to control RED robot)'

        self.realtime_fig.suptitle(title, fontsize=14, fontweight='bold')

        self.realtime_fig.canvas.mpl_connect('button_press_event', self.on_map_click)

        plt.show(block=False)

    def on_map_click(self, event):
        """Handle mouse clicks on the map to set goals for Robot 0 (Red)"""
        if event.inaxes == self.realtime_axes['grid']:
            x, y = event.xdata, event.ydata

            if (self.map_bounds['x_min'] <= x <= self.map_bounds['x_max'] and
                self.map_bounds['y_min'] <= y <= self.map_bounds['y_max']):

                self.robots[0].goal = (x, y)
                self.robots[0].manual_control = True

                print(f"RED robot goal set to: ({x:.2f}, {y:.2f})")

    def update_realtime_visualization(self, step):
        """Update the real-time visualization"""
        if self.realtime_fig is None:
            return

        for ax in self.realtime_axes.values():
            ax.clear()

        # Update occupancy grid
        ax_grid = self.realtime_axes['grid']
        grid_x = int((self.map_bounds['x_max'] - self.map_bounds['x_min']) / self.grid_resolution)
        grid_y = int((self.map_bounds['y_max'] - self.map_bounds['y_min']) / self.grid_resolution)
        grid_image = np.ones((grid_y, grid_x, 3)) * 0.7

        for cell, value in self.occupancy_grid.items():
            gx, gy = cell
            if 0 <= gx < grid_x and 0 <= gy < grid_y:
                if value == 1:
                    grid_image[gy, gx] = [1, 1, 1]
                elif value == 2:
                    grid_image[gy, gx] = [0, 0, 0]

        extent = [self.map_bounds['x_min'], self.map_bounds['x_max'],
                 self.map_bounds['y_min'], self.map_bounds['y_max']]
        ax_grid.imshow(grid_image, origin='lower', extent=extent, interpolation='nearest')

        # Plot robot trajectories and positions
        for robot, color in zip(self.robots, ['red', 'green', 'blue']):
            if robot.trajectory:
                traj = np.array(robot.trajectory)
                ax_grid.plot(traj[:, 0], traj[:, 1], c=color, linewidth=1.5, alpha=0.6)

            pos, _ = p.getBasePositionAndOrientation(robot.id)

            # Draw lidar range circle
            lidar_range = 15
            circle = plt.Circle((pos[0], pos[1]), lidar_range, color=color,
                               fill=False, linewidth=1.5, alpha=0.15, zorder=3)
            ax_grid.add_patch(circle)

            ax_grid.scatter(pos[0], pos[1], c=color, s=100, marker='^',
                          edgecolors='black', linewidths=1.5, zorder=5)

            if robot.goal is not None:
                ax_grid.scatter(robot.goal[0], robot.goal[1], c=color, s=200,
                              marker='X', edgecolors='white', linewidths=2, zorder=6)
                ax_grid.plot([pos[0], robot.goal[0]], [pos[1], robot.goal[1]],
                           c=color, linestyle='--', linewidth=2, alpha=0.7, zorder=4)

        bounds_margin = 5
        ax_grid.set_xlim(self.map_bounds['x_min'] - bounds_margin, self.map_bounds['x_max'] + bounds_margin)
        ax_grid.set_ylim(self.map_bounds['y_min'] - bounds_margin, self.map_bounds['y_max'] + bounds_margin)
        ax_grid.set_aspect('equal')
        ax_grid.grid(True, alpha=0.3)
        ax_grid.set_title('Occupancy Grid\n(White=Free, Black=Obstacle, Gray=Unexplored)')
        ax_grid.set_xlabel('X (meters)')
        ax_grid.set_ylabel('Y (meters)')

        coverage = self.calculate_coverage()
        ax_grid.text(0.02, 0.98, f'Coverage: {coverage:.1f}%\nCells: {len(self.explored_cells)}/{self.total_cells}\nStep: {step}',
                    transform=ax_grid.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                    fontsize=9, fontweight='bold')

        # Update frontier map
        ax_frontier = self.realtime_axes['frontier']
        frontiers = self.detect_frontiers()

        explored_points = []
        for cell in self.explored_cells:
            if self.occupancy_grid.get(cell) == 1:
                x, y = self.grid_to_world(cell[0], cell[1])
                explored_points.append([x, y])

        if explored_points:
            explored_array = np.array(explored_points)
            ax_frontier.scatter(explored_array[:, 0], explored_array[:, 1],
                              c='lightblue', s=3, alpha=0.4, marker='s')

        obstacle_points = []
        for cell in self.obstacle_cells:
            x, y = self.grid_to_world(cell[0], cell[1])
            obstacle_points.append([x, y])

        if obstacle_points:
            obstacle_array = np.array(obstacle_points)
            ax_frontier.scatter(obstacle_array[:, 0], obstacle_array[:, 1],
                              c='black', s=3, marker='s')

        if frontiers:
            frontier_points = [self.grid_to_world(cell[0], cell[1]) for cell in frontiers]
            frontier_array = np.array(frontier_points)
            ax_frontier.scatter(frontier_array[:, 0], frontier_array[:, 1],
                              c='yellow', s=25, marker='o', edgecolors='orange',
                              linewidths=1.5, label='Frontiers', zorder=5)

        for robot, color in zip(self.robots, ['red', 'green', 'blue']):
            pos, _ = p.getBasePositionAndOrientation(robot.id)

            # Draw lidar range circle
            lidar_range = 15
            circle = plt.Circle((pos[0], pos[1]), lidar_range, color=color,
                               fill=False, linewidth=1.5, alpha=0.15, zorder=3)
            ax_frontier.add_patch(circle)

            ax_frontier.scatter(pos[0], pos[1], c=color, s=150, marker='^',
                              edgecolors='black', linewidths=2, zorder=6)

        ax_frontier.set_xlim(self.map_bounds['x_min'] - bounds_margin, self.map_bounds['x_max'] + bounds_margin)
        ax_frontier.set_ylim(self.map_bounds['y_min'] - bounds_margin, self.map_bounds['y_max'] + bounds_margin)
        ax_frontier.set_aspect('equal')
        ax_frontier.grid(True, alpha=0.3)
        ax_frontier.set_title(f'Frontier Detection\n({len(frontiers)} frontiers)')
        ax_frontier.set_xlabel('X (meters)')
        ax_frontier.set_ylabel('Y (meters)')

        # Update coverage history
        ax_coverage = self.realtime_axes['coverage']
        if self.coverage_history:
            steps, coverage_values = zip(*self.coverage_history)
            ax_coverage.plot(steps, coverage_values, linewidth=2, color='blue')
            ax_coverage.fill_between(steps, coverage_values, alpha=0.3, color='blue')
            ax_coverage.axhline(y=coverage, color='red', linestyle='--', alpha=0.5)

        ax_coverage.set_xlabel('Simulation Step')
        ax_coverage.set_ylabel('Coverage (%)')
        ax_coverage.set_title('Coverage Progress Over Time')
        ax_coverage.grid(True, alpha=0.3)
        ax_coverage.set_ylim(0, 100)
        ax_coverage.set_xlim(0, max(2000, step))

        self.realtime_fig.canvas.draw()
        self.realtime_fig.canvas.flush_events()
        plt.pause(0.001)

    def simple_exploration_move(self, robot, step):
        """Simple movement pattern for exploration"""
        robot_idx = self.robots.index(robot)

        if robot_idx != 0:
            robot.move(0.0, 0.0)
            return

        # If manual control is enabled and goal exists, move straight toward it
        if robot.manual_control and robot.goal is not None:
            pos, orn = p.getBasePositionAndOrientation(robot.id)
            euler = p.getEulerFromQuaternion(orn)
            yaw = euler[2]

            # Calculate distance and angle to goal
            dx = robot.goal[0] - pos[0]
            dy = robot.goal[1] - pos[1]
            distance = np.sqrt(dx**2 + dy**2)

            # Stop if reached goal
            if distance < 0.5:
                robot.goal = None
                robot.manual_control = False
                robot.move(0.0, 0.0)
                return

            # Calculate desired angle to goal
            desired_angle = np.arctan2(dy, dx)
            angle_diff = desired_angle - yaw
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

            # Turn to face goal first, then move forward
            if abs(angle_diff) > 0.1:  # Still need to turn
                robot.move(0.0, 1.0 if angle_diff > 0 else -1.0)  # Just turn
            else:  # Facing the goal, move forward
                robot.move(1.0, 0.0)
            return

        # Otherwise, stationary
        robot.move(0.0, 0.0)

    def run_simulation(self, steps=5000, scan_interval=10, use_gui=True, realtime_viz=True, viz_update_interval=50):
        """Run the simulation"""
        print("Starting subterranean maze mapping simulation...")
        if steps is None:
            print("Running unlimited steps (press Ctrl+C to stop)...")
        else:
            print(f"Running for {steps} steps...")

        if realtime_viz:
            print("Setting up real-time visualization...")
            self.setup_realtime_visualization()

        step = 0
        while True:
            # Check if we've reached the step limit
            if steps is not None and step >= steps:
                break

            for robot in self.robots:
                self.simple_exploration_move(robot, step)

            if step % scan_interval == 0:
                for robot in self.robots:
                    robot.get_lidar_scan(num_rays=180, max_range=15)
                    self.update_occupancy_grid(robot)

                coverage = self.calculate_coverage()
                self.coverage_history.append((step, coverage))

            if realtime_viz and step % viz_update_interval == 0:
                self.update_realtime_visualization(step)

            p.stepSimulation()

            if use_gui:
                time.sleep(1./240.)

            if step % 200 == 0:
                coverage = self.calculate_coverage()
                num_frontiers = len(self.detect_frontiers())
                if steps is None:
                    print(f"Progress: Step {step} | Coverage: {coverage:.1f}% | Frontiers: {num_frontiers}")
                else:
                    print(f"Progress: {step}/{steps} steps | Coverage: {coverage:.1f}% | Frontiers: {num_frontiers}")

            step += 1

        if realtime_viz:
            self.update_realtime_visualization(step)
            print("\nReal-time visualization complete.")

        print("\nSimulation complete!")
        final_coverage = self.calculate_coverage()
        print(f"Final Coverage: {final_coverage:.2f}%")
        print(f"Explored Cells: {len(self.explored_cells)}/{self.total_cells}")

    def cleanup(self):
        """Disconnect from PyBullet"""
        self.env.close()


def main():
    """Main function with maze configuration"""
    print("=" * 60)
    print("Multi-Robot Subterranean Maze Coverage Mapping")
    print("=" * 60)

    # Maze configuration
    maze_size_input = input("\nEnter maze size (e.g., '10' for 10x10, default=10): ").strip()
    maze_size = int(maze_size_input) if maze_size_input.isdigit() else 10

    cell_size_input = input("Enter cell size in meters (default=2.0): ").strip()
    try:
        cell_size = float(cell_size_input)
    except:
        cell_size = 2.0

    seed_input = input("Enter random seed (press Enter for random): ").strip()
    env_seed = int(seed_input) if seed_input.isdigit() else None

    # GUI configuration
    gui_input = input("Show PyBullet 3D window? (y/n, default=n): ").strip().lower()
    use_gui = gui_input == 'y'

    # Simulation steps configuration
    steps_input = input("Number of simulation steps (press Enter for unlimited): ").strip()
    if steps_input.isdigit():
        max_steps = int(steps_input)
    else:
        max_steps = None  # Unlimited

    print(f"\nCreating {maze_size}x{maze_size} maze with {cell_size}m cells...")
    if not use_gui:
        print("Running in headless mode (no 3D window, faster simulation)")
    if max_steps is None:
        print("Running unlimited steps (press Ctrl+C to stop)")

    mapper = SubterraneanMapper(
        use_gui=use_gui,
        maze_size=(maze_size, maze_size),
        cell_size=cell_size,
        env_seed=env_seed
    )

    try:
        mapper.run_simulation(
            steps=max_steps,
            scan_interval=10,
            use_gui=use_gui,
            realtime_viz=True,
            viz_update_interval=50
        )

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        if mapper.realtime_fig is not None:
            plt.close(mapper.realtime_fig)
        mapper.cleanup()
        print("PyBullet disconnected")


if __name__ == "__main__":
    main()
