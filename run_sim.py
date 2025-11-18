import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import os

class Robot:
    def __init__(self, robot_id, position, color):
        self.id = robot_id
        self.position = position
        self.color = color
        self.lidar_data = []
        self.trajectory = []
        
    def get_lidar_scan(self, num_rays=360, max_range=10):
        """Simulate lidar by casting rays in 360 degrees"""
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        
        ray_from = []
        ray_to = []
        
        for i in range(num_rays):
            angle = yaw + (2 * np.pi * i / num_rays)
            ray_end = [
                pos[0] + max_range * np.cos(angle),
                pos[1] + max_range * np.sin(angle),
                pos[2]
            ]
            ray_from.append(pos)
            ray_to.append(ray_end)
        
        # Batch ray test for efficiency
        results = p.rayTestBatch(ray_from, ray_to)
        
        # Store lidar points in world coordinates
        scan_points = []
        for i, result in enumerate(results):
            hit_fraction = result[2]
            if hit_fraction < 1.0:  # Ray hit something
                angle = yaw + (2 * np.pi * i / num_rays)
                distance = hit_fraction * max_range
                hit_x = pos[0] + distance * np.cos(angle)
                hit_y = pos[1] + distance * np.sin(angle)
                scan_points.append((hit_x, hit_y))
        
        self.lidar_data.extend(scan_points)
        self.trajectory.append((pos[0], pos[1]))
        return scan_points
    
    def move(self, linear_vel, angular_vel):
        """Move robot with linear and angular velocity"""
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        
        # Calculate new position
        new_x = pos[0] + linear_vel * np.cos(yaw) * 0.1
        new_y = pos[1] + linear_vel * np.sin(yaw) * 0.1
        new_yaw = yaw + angular_vel * 0.1
        
        # Update position and orientation
        new_orn = p.getQuaternionFromEuler([0, 0, new_yaw])
        p.resetBasePositionAndOrientation(self.id, [new_x, new_y, 0.25], new_orn)

class MultiRobotMapper:
    def __init__(self, use_gui=True):
        # Connect to PyBullet with GUI or DIRECT mode
        if use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # Configure GUI camera if using GUI
        if use_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=25,
                cameraYaw=0,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0]
            )
        
        # Create environment
        self.setup_environment()
        
        # Create robots
        self.robots = []
        self.create_robots()
        
        # Mapping data - occupancy grid
        self.grid_resolution = 0.5  # meters per cell
        self.occupancy_grid = {}  # (grid_x, grid_y): 0=unknown, 1=free, 2=obstacle
        self.explored_cells = set()
        self.obstacle_cells = set()

        # Define map boundaries based on environment
        self.map_bounds = {
            'x_min': -10, 'x_max': 10,
            'y_min': -10, 'y_max': 10
        }

        # Coverage tracking
        total_x = int((self.map_bounds['x_max'] - self.map_bounds['x_min']) / self.grid_resolution)
        total_y = int((self.map_bounds['y_max'] - self.map_bounds['y_min']) / self.grid_resolution)
        self.total_cells = total_x * total_y
        self.coverage_history = []

        # Real-time visualization
        self.realtime_fig = None
        self.realtime_axes = None
        
    def setup_environment(self):
        """Create the environment with walls and obstacles"""
        # Ground plane
        p.loadURDF("plane.urdf")
        
        # Create walls (boundary)
        wall_half_extents = [0.2, 5, 1]
        wall_color = [0.5, 0.5, 0.5, 1]
        
        # Create collision and visual shapes for walls
        wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_half_extents)
        wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=wall_half_extents, 
                                         rgbaColor=wall_color)
        
        # Four walls
        walls_positions = [
            [0, 10, 1],   # North wall
            [0, -10, 1],  # South wall
            [10, 0, 1],   # East wall
            [-10, 0, 1]   # West wall
        ]
        
        walls_orientations = [
            p.getQuaternionFromEuler([0, 0, 0]),
            p.getQuaternionFromEuler([0, 0, 0]),
            p.getQuaternionFromEuler([0, 0, np.pi/2]),
            p.getQuaternionFromEuler([0, 0, np.pi/2])
        ]
        
        for pos, orn in zip(walls_positions, walls_orientations):
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision,
                            baseVisualShapeIndex=wall_visual,
                            basePosition=pos, baseOrientation=orn)
        
        # Add some obstacles
        obstacle_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 1, 1])
        obstacle_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[1, 1, 1],
                                             rgbaColor=[0.8, 0.3, 0.3, 1])
        
        obstacles = [
            [3, 3, 1],
            [-4, 4, 1],
            [5, -3, 1],
            [-3, -5, 1],
            [0, 0, 1]
        ]
        
        for pos in obstacles:
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obstacle_collision,
                            baseVisualShapeIndex=obstacle_visual,
                            basePosition=pos)
    
    def create_robots(self):
        """Create three robots with different colors"""
        robot_positions = [
            [-7, -7, 0.25],
            [7, -7, 0.25],
            [0, 7, 0.25]
        ]
        
        robot_colors = [
            [1, 0, 0, 1],  # Red
            [0, 1, 0, 1],  # Green
            [0, 0, 1, 1]   # Blue
        ]
        
        for i, (pos, color) in enumerate(zip(robot_positions, robot_colors)):
            # Create robot as a cylinder
            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, 
                                                    radius=0.3, height=0.5)
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER,
                                              radius=0.3, length=0.5,
                                              rgbaColor=color)
            
            robot_id = p.createMultiBody(baseMass=1,
                                        baseCollisionShapeIndex=collision_shape,
                                        baseVisualShapeIndex=visual_shape,
                                        basePosition=pos)
            
            # Add constraint to keep robot in 2D (lock z-axis and x,y rotations)
            constraint = p.createConstraint(
                robot_id, -1, -1, -1, 
                p.JOINT_FIXED, 
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

        # Get robot position
        pos, _ = p.getBasePositionAndOrientation(robot.id)
        robot_grid = self.world_to_grid(pos[0], pos[1])

        # Mark robot position as free
        self.occupancy_grid[robot_grid] = 1
        self.explored_cells.add(robot_grid)

        # Process recent lidar hits (last scan)
        recent_scans = robot.lidar_data[-180:] if len(robot.lidar_data) > 180 else robot.lidar_data

        for hit_x, hit_y in recent_scans:
            # Mark obstacle location
            obstacle_grid = self.world_to_grid(hit_x, hit_y)
            self.occupancy_grid[obstacle_grid] = 2
            self.obstacle_cells.add(obstacle_grid)

            # Ray trace from robot to obstacle to mark free cells
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
            # Mark cell as free (but don't overwrite obstacles)
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
            if self.occupancy_grid.get(cell) != 1:  # Only check free cells
                continue

            # Check 8-connected neighbors
            x, y = cell
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    neighbor = (x + dx, y + dy)

                    # Check if neighbor is within bounds
                    world_x, world_y = self.grid_to_world(neighbor[0], neighbor[1])
                    if not (self.map_bounds['x_min'] <= world_x <= self.map_bounds['x_max'] and
                           self.map_bounds['y_min'] <= world_y <= self.map_bounds['y_max']):
                        continue

                    # If neighbor is unknown, current cell is a frontier
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
        plt.ion()  # Turn on interactive mode
        self.realtime_fig = plt.figure(figsize=(15, 10))
        gs = self.realtime_fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        self.realtime_axes = {
            'grid': self.realtime_fig.add_subplot(gs[0, 0]),
            'frontier': self.realtime_fig.add_subplot(gs[0, 1]),
            'coverage': self.realtime_fig.add_subplot(gs[1, :])
        }

        self.realtime_fig.suptitle('Real-Time Multi-Robot Coverage Mapping',
                                   fontsize=14, fontweight='bold')
        plt.show(block=False)

    def update_realtime_visualization(self, step):
        """Update the real-time visualization"""
        if self.realtime_fig is None:
            return

        # Clear all axes
        for ax in self.realtime_axes.values():
            ax.clear()

        # Update occupancy grid
        ax_grid = self.realtime_axes['grid']
        grid_x = int((self.map_bounds['x_max'] - self.map_bounds['x_min']) / self.grid_resolution)
        grid_y = int((self.map_bounds['y_max'] - self.map_bounds['y_min']) / self.grid_resolution)
        grid_image = np.ones((grid_y, grid_x, 3)) * 0.7  # Gray for unexplored

        # Fill in explored and obstacle cells
        for cell, value in self.occupancy_grid.items():
            gx, gy = cell
            if 0 <= gx < grid_x and 0 <= gy < grid_y:
                if value == 1:  # Free space
                    grid_image[gy, gx] = [1, 1, 1]  # White
                elif value == 2:  # Obstacle
                    grid_image[gy, gx] = [0, 0, 0]  # Black

        extent = [self.map_bounds['x_min'], self.map_bounds['x_max'],
                 self.map_bounds['y_min'], self.map_bounds['y_max']]
        ax_grid.imshow(grid_image, origin='lower', extent=extent, interpolation='nearest')

        # Plot robot trajectories and current positions
        for robot, color in zip(self.robots, ['red', 'green', 'blue']):
            if robot.trajectory:
                traj = np.array(robot.trajectory)
                ax_grid.plot(traj[:, 0], traj[:, 1], c=color, linewidth=1.5, alpha=0.6)

            # Current position
            pos, _ = p.getBasePositionAndOrientation(robot.id)
            ax_grid.scatter(pos[0], pos[1], c=color, s=100, marker='^',
                          edgecolors='black', linewidths=1.5, zorder=5)

        ax_grid.set_xlim(-12, 12)
        ax_grid.set_ylim(-12, 12)
        ax_grid.set_aspect('equal')
        ax_grid.grid(True, alpha=0.3)
        ax_grid.set_title('Occupancy Grid\n(White=Free, Black=Obstacle, Gray=Unexplored)')
        ax_grid.set_xlabel('X (meters)')
        ax_grid.set_ylabel('Y (meters)')

        # Add coverage text
        coverage = self.calculate_coverage()
        ax_grid.text(0.02, 0.98, f'Coverage: {coverage:.1f}%\nCells: {len(self.explored_cells)}/{self.total_cells}\nStep: {step}',
                    transform=ax_grid.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                    fontsize=9, fontweight='bold')

        # Update frontier map
        ax_frontier = self.realtime_axes['frontier']
        frontiers = self.detect_frontiers()

        # Plot explored area
        explored_points = []
        for cell in self.explored_cells:
            if self.occupancy_grid.get(cell) == 1:
                x, y = self.grid_to_world(cell[0], cell[1])
                explored_points.append([x, y])

        if explored_points:
            explored_array = np.array(explored_points)
            ax_frontier.scatter(explored_array[:, 0], explored_array[:, 1],
                              c='lightblue', s=3, alpha=0.4, marker='s')

        # Plot obstacles
        obstacle_points = []
        for cell in self.obstacle_cells:
            x, y = self.grid_to_world(cell[0], cell[1])
            obstacle_points.append([x, y])

        if obstacle_points:
            obstacle_array = np.array(obstacle_points)
            ax_frontier.scatter(obstacle_array[:, 0], obstacle_array[:, 1],
                              c='black', s=3, marker='s')

        # Plot frontiers
        if frontiers:
            frontier_points = [self.grid_to_world(cell[0], cell[1]) for cell in frontiers]
            frontier_array = np.array(frontier_points)
            ax_frontier.scatter(frontier_array[:, 0], frontier_array[:, 1],
                              c='yellow', s=25, marker='o', edgecolors='orange',
                              linewidths=1.5, label='Frontiers', zorder=5)

        # Plot robot current positions
        for robot, color in zip(self.robots, ['red', 'green', 'blue']):
            pos, _ = p.getBasePositionAndOrientation(robot.id)
            ax_frontier.scatter(pos[0], pos[1], c=color, s=150, marker='^',
                              edgecolors='black', linewidths=2, zorder=6)

        ax_frontier.set_xlim(-12, 12)
        ax_frontier.set_ylim(-12, 12)
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

        # Force update
        self.realtime_fig.canvas.draw()
        self.realtime_fig.canvas.flush_events()
        plt.pause(0.001)

    def simple_exploration_move(self, robot, step):
        """Simple movement pattern for exploration"""
        robot_idx = self.robots.index(robot)

        # Each robot follows a different pattern
        if robot_idx == 0:  # Red robot - circular pattern
            linear = 1.0
            angular = 0.3
        elif robot_idx == 1:  # Green robot - figure-8 pattern
            linear = 1.0
            angular = 0.5 * np.sin(step * 0.02)
        else:  # Blue robot - spiral pattern
            linear = 1.0
            angular = 0.2 + 0.1 * np.sin(step * 0.01)

        robot.move(linear, angular)
    
    def run_simulation(self, steps=2000, scan_interval=10, use_gui=True, realtime_viz=True, viz_update_interval=50):
        """Run the simulation

        Args:
            steps: Number of simulation steps
            scan_interval: How often to perform lidar scans
            use_gui: Whether to show PyBullet GUI
            realtime_viz: Whether to show real-time map visualization
            viz_update_interval: How often to update the real-time visualization (in steps)
        """
        print("Starting multi-robot mapping simulation...")
        print("Red, Green, and Blue robots are exploring the environment")
        print(f"Running for {steps} steps...")

        # Setup real-time visualization if enabled
        if realtime_viz:
            print("Setting up real-time visualization...")
            self.setup_realtime_visualization()

        for step in range(steps):
            # Move each robot
            for robot in self.robots:
                self.simple_exploration_move(robot, step)

            # Perform lidar scans periodically
            if step % scan_interval == 0:
                for robot in self.robots:
                    robot.get_lidar_scan(num_rays=180, max_range=15)
                    # Update occupancy grid with new scan data
                    self.update_occupancy_grid(robot)

                # Track coverage over time
                coverage = self.calculate_coverage()
                self.coverage_history.append((step, coverage))

            # Update real-time visualization
            if realtime_viz and step % viz_update_interval == 0:
                self.update_realtime_visualization(step)

            # Step simulation
            p.stepSimulation()

            # Add delay for GUI visualization
            if use_gui:
                time.sleep(1./240.)  # 240 Hz simulation rate

            # Print progress with coverage
            if step % 200 == 0:
                coverage = self.calculate_coverage()
                num_frontiers = len(self.detect_frontiers())
                print(f"Progress: {step}/{steps} steps | Coverage: {coverage:.1f}% | Frontiers: {num_frontiers}")

        # Final update of real-time visualization
        if realtime_viz:
            self.update_realtime_visualization(steps)
            print("\nReal-time visualization complete. Close the plot window to continue...")

        print("\nSimulation complete!")
        print(f"Robot 1 (Red) scanned {len(self.robots[0].lidar_data)} points")
        print(f"Robot 2 (Green) scanned {len(self.robots[1].lidar_data)} points")
        print(f"Robot 3 (Blue) scanned {len(self.robots[2].lidar_data)} points")
        final_coverage = self.calculate_coverage()
        print(f"\nFinal Coverage: {final_coverage:.2f}%")
        print(f"Explored Cells: {len(self.explored_cells)}/{self.total_cells}")
    
    def visualize_map(self):
        """Visualize the mapped environment with occupancy grid and frontiers"""
        # Create output directory if it doesn't exist
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)

        # Turn off interactive mode for saving
        plt.ioff()

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        colors = ['red', 'green', 'blue']
        robot_names = ['Robot 1 (Red)', 'Robot 2 (Green)', 'Robot 3 (Blue)']

        # 1. Occupancy Grid Map
        ax_grid = fig.add_subplot(gs[0, 0])
        self._plot_occupancy_grid(ax_grid)

        # 2. Frontier Map
        ax_frontier = fig.add_subplot(gs[0, 1])
        self._plot_frontier_map(ax_frontier)

        # 3. Coverage Over Time
        ax_coverage = fig.add_subplot(gs[0, 2])
        self._plot_coverage_history(ax_coverage)

        # 4-6. Individual robot trajectories with lidar
        for idx, (robot, color, name) in enumerate(zip(self.robots, colors, robot_names)):
            ax = fig.add_subplot(gs[1, idx])

            if robot.lidar_data:
                points = np.array(robot.lidar_data)
                ax.scatter(points[:, 0], points[:, 1], c=color, s=1, alpha=0.2)

            if robot.trajectory:
                traj = np.array(robot.trajectory)
                ax.plot(traj[:, 0], traj[:, 1], c=color, linewidth=2, label='Trajectory')
                # Mark start and end
                ax.scatter(traj[0, 0], traj[0, 1], c='black', s=100, marker='o',
                          label='Start', zorder=5)
                ax.scatter(traj[-1, 0], traj[-1, 1], c='gold', s=100, marker='*',
                          label='End', zorder=5)

            ax.set_xlim(-12, 12)
            ax.set_ylim(-12, 12)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{name} Trajectory')
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.legend(fontsize=8)

        # Add overall title with coverage info
        final_coverage = self.calculate_coverage()
        fig.suptitle(f'Multi-Robot Coverage Mapping - Final Coverage: {final_coverage:.2f}%',
                    fontsize=16, fontweight='bold')

        output_path = os.path.join(output_dir, 'multi_robot_map.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Map visualization saved to {output_path}!")
        plt.close()

    def _plot_occupancy_grid(self, ax):
        """Plot the occupancy grid"""
        # Create grid image
        grid_x = int((self.map_bounds['x_max'] - self.map_bounds['x_min']) / self.grid_resolution)
        grid_y = int((self.map_bounds['y_max'] - self.map_bounds['y_min']) / self.grid_resolution)
        grid_image = np.ones((grid_y, grid_x, 3)) * 0.7  # Gray for unexplored

        # Fill in explored and obstacle cells
        for cell, value in self.occupancy_grid.items():
            gx, gy = cell
            if 0 <= gx < grid_x and 0 <= gy < grid_y:
                if value == 1:  # Free space
                    grid_image[gy, gx] = [1, 1, 1]  # White
                elif value == 2:  # Obstacle
                    grid_image[gy, gx] = [0, 0, 0]  # Black

        # Plot robot trajectories on grid
        for robot, color in zip(self.robots, ['red', 'green', 'blue']):
            if robot.trajectory:
                traj = np.array(robot.trajectory)
                ax.plot(traj[:, 0], traj[:, 1], c=color, linewidth=2, alpha=0.7)

        extent = [self.map_bounds['x_min'], self.map_bounds['x_max'],
                 self.map_bounds['y_min'], self.map_bounds['y_max']]
        ax.imshow(grid_image, origin='lower', extent=extent, interpolation='nearest')

        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Occupancy Grid Map\n(White=Free, Black=Obstacle, Gray=Unexplored)')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')

        # Add legend for explored percentage
        explored_pct = self.calculate_coverage()
        ax.text(0.02, 0.98, f'Explored: {explored_pct:.1f}%\nCells: {len(self.explored_cells)}/{self.total_cells}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9)

    def _plot_frontier_map(self, ax):
        """Plot frontiers (boundary between explored and unexplored)"""
        # Get frontiers
        frontiers = self.detect_frontiers()

        # Plot explored area
        for cell in self.explored_cells:
            if self.occupancy_grid.get(cell) == 1:  # Free space
                x, y = self.grid_to_world(cell[0], cell[1])
                ax.plot(x, y, 's', color='lightblue', markersize=2, alpha=0.5)

        # Plot obstacles
        for cell in self.obstacle_cells:
            x, y = self.grid_to_world(cell[0], cell[1])
            ax.plot(x, y, 's', color='black', markersize=2)

        # Plot frontiers
        if frontiers:
            frontier_points = [self.grid_to_world(cell[0], cell[1]) for cell in frontiers]
            frontier_array = np.array(frontier_points)
            ax.scatter(frontier_array[:, 0], frontier_array[:, 1],
                      c='yellow', s=20, marker='o', edgecolors='orange',
                      linewidths=1, label='Frontiers', zorder=5)

        # Plot robot current positions
        for robot, color in zip(self.robots, ['red', 'green', 'blue']):
            pos, _ = p.getBasePositionAndOrientation(robot.id)
            ax.scatter(pos[0], pos[1], c=color, s=150, marker='^',
                      edgecolors='black', linewidths=2, zorder=6)

        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Frontier Map\n(Yellow=Frontiers, Triangles=Robots)')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')

        # Add stats
        num_frontiers = len(frontiers)
        ax.text(0.02, 0.98, f'Frontiers: {num_frontiers}\nExplored: {len(self.explored_cells)}\nObstacles: {len(self.obstacle_cells)}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9)

        if frontiers:
            ax.legend(loc='upper right', fontsize=8)

    def _plot_coverage_history(self, ax):
        """Plot coverage percentage over time"""
        if self.coverage_history:
            steps, coverage = zip(*self.coverage_history)
            ax.plot(steps, coverage, linewidth=2, color='blue', marker='o',
                   markersize=3, markevery=10)
            ax.fill_between(steps, coverage, alpha=0.3, color='blue')

            ax.set_xlabel('Simulation Step')
            ax.set_ylabel('Coverage (%)')
            ax.set_title('Coverage Progress Over Time')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(100, max(coverage) * 1.1))

            # Add final coverage annotation
            final_coverage = coverage[-1]
            ax.axhline(y=final_coverage, color='red', linestyle='--', alpha=0.5)
            ax.text(0.98, 0.02, f'Final: {final_coverage:.2f}%',
                   transform=ax.transAxes, horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                   fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No coverage data',
                   transform=ax.transAxes, horizontalalignment='center')
            ax.set_title('Coverage Progress Over Time')
    
    def cleanup(self):
        """Disconnect from PyBullet"""
        p.disconnect()

def main():
    # Create mapper with GUI enabled
    mapper = MultiRobotMapper(use_gui=True)

    # Run simulation with real-time visualization
    try:
        mapper.run_simulation(
            steps=2000,
            scan_interval=10,
            use_gui=True,
            realtime_viz=False,  # Enable real-time map visualization
            viz_update_interval=50  # Update map every 50 steps
        )

        # Generate final static map visualization
        print("\nGenerating final map visualization...")

        # Visualize results
        mapper.visualize_map()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        # Close real-time visualization if open
        if mapper.realtime_fig is not None:
            plt.close(mapper.realtime_fig)
        mapper.cleanup()
        print("PyBullet disconnected")

if __name__ == "__main__":
    main()