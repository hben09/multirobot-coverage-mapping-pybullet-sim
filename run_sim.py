"""
Multi-robot coverage mapping simulation for subterranean maze environments
Uses the MazeEnvironment class from environment_generator to create procedurally generated mazes

IMPROVED VERSION: Added direction bias to reduce oscillation at intersections

Architectural influences:
1. MGG Planner: Bifurcated Local/Global state machine.
2. GBPlanner: Frontier-based target generation with DIRECTION BIAS.
3. AMET: Utility-based coordination (Size - Distance).
4. NeBula: A* Pathfinding on risk/occupancy grid.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import math
import heapq
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
        
        # Autonomy State
        self.goal = None
        self.path = []  # List of grid waypoints from A*
        self.manual_control = False
        self.mode = 'IDLE'
        
        # NEW: Direction tracking for exploration bias
        self.exploration_direction = np.array([1.0, 0.0])  # Initial heading (east)
        self.direction_history = []  # Track recent movement directions
        self.direction_smoothing_window = 10  # Number of samples to average
        
        # NEW: Stuck detection
        self.stuck_counter = 0
        self.last_position = np.array(position[:2])
        self.goal_attempts = 0
        self.max_goal_attempts = 3  # Give up after 3 failed attempts to reach a goal

    def update_exploration_direction(self):
        """
        Update the robot's exploration direction based on recent movement.
        Uses a low-pass filter over recent trajectory points (like GBPlanner).
        """
        if len(self.trajectory) < 2:
            return
        
        # Get recent trajectory segment
        recent_points = self.trajectory[-min(20, len(self.trajectory)):]
        
        if len(recent_points) >= 2:
            # Calculate direction from older point to newer point
            start = np.array(recent_points[0])
            end = np.array(recent_points[-1])
            direction = end - start
            
            norm = np.linalg.norm(direction)
            if norm > 0.5:  # Only update if we've moved meaningfully
                direction = direction / norm
                
                # Smooth with existing direction (low-pass filter)
                alpha = 0.3  # Smoothing factor
                self.exploration_direction = (
                    alpha * direction + 
                    (1 - alpha) * self.exploration_direction
                )
                # Renormalize
                self.exploration_direction /= np.linalg.norm(self.exploration_direction)

    def get_heading_vector(self):
        """Get the robot's current facing direction from physics."""
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        return np.array([np.cos(yaw), np.sin(yaw)])

    def check_if_stuck(self, threshold=0.3, stuck_limit=50):
        """
        Check if robot is stuck (not moving despite having a goal).
        Returns True if stuck.
        """
        pos, _ = p.getBasePositionAndOrientation(self.id)
        current_pos = np.array([pos[0], pos[1]])
        
        distance_moved = np.linalg.norm(current_pos - self.last_position)
        
        if distance_moved < threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_position = current_pos
        
        return self.stuck_counter > stuck_limit

    def reset_stuck_state(self):
        """Reset stuck detection when getting a new goal."""
        self.stuck_counter = 0
        pos, _ = p.getBasePositionAndOrientation(self.id)
        self.last_position = np.array([pos[0], pos[1]])

    def get_lidar_scan(self, num_rays=360, max_range=10):
        """Simulate lidar by casting rays in 360 degrees"""
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        # Mount Lidar Higher to avoid self-collision
        lidar_height_offset = 0.3 
        lidar_z = pos[2] + lidar_height_offset

        ray_from = []
        ray_to = []

        for i in range(num_rays):
            angle = yaw + (2.0 * np.pi * i / num_rays)
            ray_from.append([pos[0], pos[1], lidar_z])
            ray_to.append([
                pos[0] + max_range * np.cos(angle),
                pos[1] + max_range * np.sin(angle),
                lidar_z
            ])

        results = p.rayTestBatch(ray_from, ray_to)

        scan_points = []
        for i, result in enumerate(results):
            hit_object_id = result[0]
            hit_fraction = result[2]
            
            # Ignore self-collision
            if hit_fraction < 1.0 and hit_object_id != self.id:
                hit_position = result[3]
                scan_points.append((hit_position[0], hit_position[1]))

        self.lidar_data.extend(scan_points)
        self.trajectory.append((pos[0], pos[1]))
        
        # NEW: Update exploration direction after recording trajectory
        self.update_exploration_direction()
        
        return scan_points

    def move(self, linear_vel, angular_vel):
        """Move robot using physics velocity control."""
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        vx = linear_vel * np.cos(yaw)
        vy = linear_vel * np.sin(yaw)

        p.resetBaseVelocity(self.id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, angular_vel])

    def follow_path(self, mapper):
        """
        Follow the A* path. 
        If path is empty but goal exists, drive to goal.
        """
        if not self.path and self.goal is None:
            return 0.0, 0.0

        target_pos = self.goal
        
        # If we have a path, aim for the next waypoint
        if self.path:
            # Get next waypoint (grid coords)
            next_wp = self.path[0]
            # Convert to world coords
            wx, wy = mapper.grid_to_world(next_wp[0], next_wp[1])
            target_pos = (wx, wy)
            
            # Check if we reached this waypoint
            pos, _ = p.getBasePositionAndOrientation(self.id)
            dist = np.sqrt((wx - pos[0])**2 + (wy - pos[1])**2)
            
            # Waypoint acceptance radius
            if dist < 0.6:
                self.path.pop(0)  # Waypoint reached
                if self.path:  # If more points, recurse to get next target
                    return self.follow_path(mapper)

        # Drive to target_pos (Pure Pursuit / Proportional Control)
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        dx = target_pos[0] - pos[0]
        dy = target_pos[1] - pos[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Stop if reached final goal
        if distance < 0.5 and not self.path:
            self.goal = None 
            return 0.0, 0.0

        desired_angle = np.arctan2(dy, dx)
        angle_diff = desired_angle - yaw
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        # Control Limits (User Requested: 4.0 m/s)
        linear_vel = min(4.0, distance * 2.0)
        angular_vel = np.clip(angle_diff * 4.0, -4.0, 4.0)
        
        # Slow down if turning sharply
        if abs(angle_diff) > 0.5:
            linear_vel *= 0.1

        return linear_vel, angular_vel


class SubterraneanMapper:
    """Multi-robot mapper for subterranean maze environments"""

    def __init__(self, use_gui=True, maze_size=(10, 10), cell_size=2.0, env_seed=None, env_type='maze'):
        self.env = MazeEnvironment(
            maze_size=maze_size,
            cell_size=cell_size,
            wall_height=2.5,
            wall_thickness=0.2,
            gui=use_gui,
            seed=env_seed
        )

        self.env.generate_maze(env_type=env_type)
        self.env.build_walls()
        self.physics_client = self.env.physics_client

        self.robots = []
        self.create_robots()

        self.grid_resolution = 0.5
        self.occupancy_grid = {}  # 1=Free, 2=Obstacle
        self.explored_cells = set()
        self.obstacle_cells = set()

        block_physical_size = self.env.cell_size / 2.0
        half_block = block_physical_size / 2.0
        
        maze_world_width = self.env.maze_grid.shape[1] * block_physical_size
        maze_world_height = self.env.maze_grid.shape[0] * block_physical_size
        
        self.map_bounds = {
            'x_min': -half_block,
            'x_max': maze_world_width - half_block,
            'y_min': -half_block,
            'y_max': maze_world_height - half_block
        }

        self.scale_factor = block_physical_size / self.grid_resolution
        ground_truth_zeros = np.sum(self.env.maze_grid == 0)
        self.total_free_cells = ground_truth_zeros * (self.scale_factor ** 2)
        
        self.coverage_history = []
        self.realtime_fig = None
        self.realtime_axes = None
        self.claimed_goals = {}
        
        # NEW: Direction bias parameters (tunable)
        self.direction_bias_weight = 2.5  # How much to reward forward motion
        self.size_weight = 0.5            # Weight for frontier size
        self.distance_weight = 1.0        # Weight for distance cost

    def create_robots(self):
        spawn_pos = self.env.get_spawn_position()
        # Line them up
        start_positions = [
            [spawn_pos[0] - 1.5, spawn_pos[1], 0.25],
            [spawn_pos[0], spawn_pos[1], 0.25],
            [spawn_pos[0] + 1.5, spawn_pos[1], 0.25]
        ]
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]

        for i, (pos, color) in enumerate(zip(start_positions, colors)):
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.25)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.25, rgbaColor=color)

            robot_id = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=pos
            )

            # Physics: Prevent rolling, allow sliding
            p.changeDynamics(robot_id, -1, localInertiaDiagonal=[0, 0, 1])
            p.changeDynamics(robot_id, -1, lateralFriction=1.0, spinningFriction=0.1, rollingFriction=0.0)

            robot = Robot(robot_id, pos, color)
            self.robots.append(robot)

    def world_to_grid(self, x, y):
        grid_x = int((x - self.map_bounds['x_min']) / self.grid_resolution)
        grid_y = int((y - self.map_bounds['y_min']) / self.grid_resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        x = self.map_bounds['x_min'] + (grid_x + 0.5) * self.grid_resolution
        y = self.map_bounds['y_min'] + (grid_y + 0.5) * self.grid_resolution
        return (x, y)

    def update_occupancy_grid(self, robot):
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
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            cell = (x, y)
            # Don't overwrite obstacles with free space
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
        """Detect and cluster frontier cells."""
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
                    # Check bounds
                    if not (self.map_bounds['x_min'] <= world_x <= self.map_bounds['x_max'] and
                           self.map_bounds['y_min'] <= world_y <= self.map_bounds['y_max']):
                        continue
                    # Neighbor not in grid = Unknown = Frontier
                    if neighbor not in self.occupancy_grid:
                        frontiers.add(cell)
                        break
        
        if not frontiers:
            return []

        # Clustering
        frontier_list = list(frontiers)
        visited = set()
        clusters = []

        for f in frontier_list:
            if f in visited:
                continue
            cluster = []
            queue = [f]
            visited.add(f)
            while queue:
                current = queue.pop(0)
                cluster.append(current)
                cx, cy = current
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nbr = (cx + dx, cy + dy)
                        if nbr in frontiers and nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)
            
            # Filter small noise
            if len(cluster) > 2:
                # Find the member of the cluster closest to the mathematical centroid
                avg_x = sum(c[0] for c in cluster) / len(cluster)
                avg_y = sum(c[1] for c in cluster) / len(cluster)
                
                best_point = cluster[0]
                min_dist = float('inf')
                for point in cluster:
                    d = (point[0] - avg_x)**2 + (point[1] - avg_y)**2
                    if d < min_dist:
                        min_dist = d
                        best_point = point

                wx, wy = self.grid_to_world(best_point[0], best_point[1])
                clusters.append({'pos': (wx, wy), 'grid_pos': best_point, 'size': len(cluster)})

        return clusters

    def plan_path_astar(self, start_grid, goal_grid):
        """
        A* Pathfinding on the occupancy grid.
        FIXED: Prevents diagonal corner cutting and adds obstacle buffer.
        """
        def heuristic(a, b):
            return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

        frontier = []
        heapq.heappush(frontier, (0, start_grid))
        came_from = {}
        cost_so_far = {}
        came_from[start_grid] = None
        cost_so_far[start_grid] = 0
        
        # Pre-compute inflated obstacles (1-cell buffer around walls)
        inflated_obstacles = set(self.obstacle_cells)
        for obs in self.obstacle_cells:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    inflated_obstacles.add((obs[0] + dx, obs[1] + dy))
        
        def is_safe(cell, check_buffer=True):
            """Check if cell is safe to traverse."""
            # Check if it's a known obstacle
            if cell in self.obstacle_cells:
                return False
            # Check if it's in the buffer zone (too close to walls)
            if check_buffer and cell in inflated_obstacles:
                # Allow if it's the start or goal (we might start near a wall)
                if cell != start_grid and cell != goal_grid:
                    return False
            # Must be explored (known free space)
            if cell not in self.occupancy_grid:
                return False
            return True
        
        def can_move_diagonal(current, dx, dy):
            """
            Check if diagonal movement is valid.
            Prevents corner cutting through walls.
            """
            # For diagonal moves, both adjacent cardinal cells must be free
            # e.g., to move NE (+1,+1), both N (0,+1) and E (+1,0) must be clear
            if abs(dx) == 1 and abs(dy) == 1:
                cardinal1 = (current[0] + dx, current[1])  # Horizontal neighbor
                cardinal2 = (current[0], current[1] + dy)  # Vertical neighbor
                
                # Both cardinal directions must be free (not obstacles)
                if cardinal1 in self.obstacle_cells or cardinal2 in self.obstacle_cells:
                    return False
            return True

        found = False
        iterations = 0
        max_iterations = 10000  # Prevent infinite loops
        
        while frontier and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(frontier)[1]

            if current == goal_grid:
                found = True
                break

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    next_cell = (current[0] + dx, current[1] + dy)
                    
                    # Basic safety check
                    if not is_safe(next_cell, check_buffer=True):
                        continue
                    
                    # FIXED: Check diagonal corner cutting
                    if not can_move_diagonal(current, dx, dy):
                        continue

                    new_cost = cost_so_far[current] + np.sqrt(dx*dx + dy*dy)
                    if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                        cost_so_far[next_cell] = new_cost
                        priority = new_cost + heuristic(goal_grid, next_cell)
                        heapq.heappush(frontier, (priority, next_cell))
                        came_from[next_cell] = current
        
        if not found:
            # Try again without buffer if no path found (robot might be in tight space)
            return self._plan_path_astar_no_buffer(start_grid, goal_grid)
            
        # Reconstruct path
        path = []
        curr = goal_grid
        while curr != start_grid:
            path.append(curr)
            curr = came_from[curr]
        path.reverse()
        
        # Downsample path for smoother motion (but keep more points for accuracy)
        if len(path) > 8:
            path = path[::2]  # Keep every 2nd point instead of every 3rd
            if path[-1] != goal_grid:
                path.append(goal_grid)
                
        return path
    
    def _plan_path_astar_no_buffer(self, start_grid, goal_grid):
        """Fallback A* without obstacle inflation for tight spaces."""
        def heuristic(a, b):
            return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

        frontier = []
        heapq.heappush(frontier, (0, start_grid))
        came_from = {}
        cost_so_far = {}
        came_from[start_grid] = None
        cost_so_far[start_grid] = 0
        
        def is_safe(cell):
            if cell in self.obstacle_cells:
                return False
            if cell not in self.occupancy_grid:
                return False
            return True
        
        def can_move_diagonal(current, dx, dy):
            if abs(dx) == 1 and abs(dy) == 1:
                cardinal1 = (current[0] + dx, current[1])
                cardinal2 = (current[0], current[1] + dy)
                if cardinal1 in self.obstacle_cells or cardinal2 in self.obstacle_cells:
                    return False
            return True

        found = False
        iterations = 0
        max_iterations = 10000
        
        while frontier and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(frontier)[1]

            if current == goal_grid:
                found = True
                break

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    next_cell = (current[0] + dx, current[1] + dy)
                    
                    if not is_safe(next_cell):
                        continue
                    
                    if not can_move_diagonal(current, dx, dy):
                        continue

                    new_cost = cost_so_far[current] + np.sqrt(dx*dx + dy*dy)
                    if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                        cost_so_far[next_cell] = new_cost
                        priority = new_cost + heuristic(goal_grid, next_cell)
                        heapq.heappush(frontier, (priority, next_cell))
                        came_from[next_cell] = current
        
        if not found:
            return []
            
        path = []
        curr = goal_grid
        while curr != start_grid:
            path.append(curr)
            curr = came_from[curr]
        path.reverse()
        
        if len(path) > 8:
            path = path[::2]
            if path[-1] != goal_grid:
                path.append(goal_grid)
                
        return path

    def calculate_utility(self, robot, frontier):
        """
        IMPROVED Utility Function with Direction Bias (GBPlanner-inspired)
        
        Utility = (Size × size_weight) - (Distance × dist_weight) + (Alignment × direction_weight)
        
        The alignment term rewards frontiers that are in the robot's current
        exploration direction, reducing oscillation at intersections.
        """
        pos, orn = p.getBasePositionAndOrientation(robot.id)
        robot_pos = np.array([pos[0], pos[1]])
        frontier_pos = np.array(frontier['pos'])
        
        # Distance cost
        dist = np.linalg.norm(frontier_pos - robot_pos)
        
        # Size gain
        size_gain = frontier['size'] * self.size_weight
        
        # Distance cost
        distance_cost = dist * self.distance_weight
        
        # NEW: Direction alignment bonus
        # Use the robot's smoothed exploration direction
        to_frontier = frontier_pos - robot_pos
        to_frontier_norm = np.linalg.norm(to_frontier)
        
        if to_frontier_norm > 0.1:
            to_frontier_unit = to_frontier / to_frontier_norm
            
            # Dot product gives alignment: +1 (same direction) to -1 (opposite)
            alignment = np.dot(robot.exploration_direction, to_frontier_unit)
            
            # Transform from [-1, 1] to [0, 1] for the bonus
            # alignment=1 (forward) -> bonus=1
            # alignment=0 (perpendicular) -> bonus=0.5
            # alignment=-1 (backward) -> bonus=0
            alignment_normalized = (alignment + 1.0) / 2.0
            
            direction_bonus = alignment_normalized * self.direction_bias_weight
        else:
            direction_bonus = 0.0
        
        # Final utility
        utility = size_gain - distance_cost + direction_bonus
        
        return utility, {
            'size_gain': size_gain,
            'distance_cost': distance_cost,
            'direction_bonus': direction_bonus,
            'alignment': alignment if to_frontier_norm > 0.1 else 0.0
        }

    def assign_global_goals(self):
        """
        Market-based Coordination loop with direction bias.
        Assigns frontiers to robots based on improved Utility scores.
        """
        frontiers = self.detect_frontiers()
        if not frontiers:
            return

        for robot in self.robots:
            # Don't interrupt manual control
            if robot.manual_control:
                continue
            
            # Check if current goal is valid/reached
            needs_goal = False
            if robot.goal is None:
                needs_goal = True
            else:
                pos, _ = p.getBasePositionAndOrientation(robot.id)
                dist = np.sqrt((robot.goal[0] - pos[0])**2 + (robot.goal[1] - pos[1])**2)
                if dist < 1.0 and not robot.path:
                    needs_goal = True
            
            if needs_goal:
                robot.mode = 'GLOBAL_RELOCATE'
                pos, _ = p.getBasePositionAndOrientation(robot.id)
                
                best_utility = -9999
                best_target = None
                best_grid_target = None
                best_debug = None
                
                for f in frontiers:
                    target_pos = f['pos']
                    
                    # Coordination: Check if claimed by another robot
                    is_claimed = False
                    for other_r in self.robots:
                        if other_r.id == robot.id:
                            continue
                        if other_r.goal == target_pos:
                            is_claimed = True
                            break
                    if is_claimed:
                        continue
                    
                    # NEW: Use improved utility with direction bias
                    util, debug_info = self.calculate_utility(robot, f)
                    
                    if util > best_utility:
                        best_utility = util
                        best_target = target_pos
                        best_grid_target = f['grid_pos']
                        best_debug = debug_info
                
                if best_target:
                    robot.goal = best_target
                    robot.reset_stuck_state()  # Reset stuck detection for new goal
                    robot.goal_attempts = 0    # Reset attempt counter on successful goal
                    # Plan path using A*
                    start_grid = self.world_to_grid(pos[0], pos[1])
                    path = self.plan_path_astar(start_grid, best_grid_target)
                    if path:
                        robot.path = path
                    else:
                        robot.goal = None
                else:
                    # No best target - spin to search
                    robot.move(0.0, 1.0)

    def exploration_logic(self, robot, step):
        if robot.manual_control:
            l, a = robot.follow_path(self) 
            robot.move(l, a)
            return

        if robot.goal:
            # Check if robot is stuck
            if robot.check_if_stuck(threshold=0.2, stuck_limit=100):
                print(f"Robot {robot.id} is stuck! Abandoning goal and replanning...")
                robot.goal = None
                robot.path = []
                robot.reset_stuck_state()
                robot.goal_attempts += 1
                
                # If stuck too many times, spin to reorient
                if robot.goal_attempts > robot.max_goal_attempts:
                    robot.move(0.0, 2.0)  # Spin fast to break free
                    robot.goal_attempts = 0
                return
            
            l, a = robot.follow_path(self)
            robot.move(l, a)
        else:
            # Local Scanning Mode: Spin slowly to build map if no goal
            robot.move(0.0, 0.5)

    def calculate_coverage(self):
        if self.total_free_cells == 0:
            return 0.0

        block_size = self.env.cell_size / 2.0
        valid_explored_count = 0

        for cell in self.explored_cells:
            wx, wy = self.grid_to_world(cell[0], cell[1])
            mx = int((wx + block_size/2) / block_size)
            my = int((wy + block_size/2) / block_size)
            
            if (0 <= mx < self.env.maze_grid.shape[1] and 
                0 <= my < self.env.maze_grid.shape[0]):
                if self.env.maze_grid[my, mx] == 0:
                    valid_explored_count += 1

        coverage_percent = min(100.0, (valid_explored_count / self.total_free_cells) * 100)
        return coverage_percent

    def setup_realtime_visualization(self):
        plt.ion()
        # Larger figure with more space for top maps
        self.realtime_fig = plt.figure(figsize=(18, 14))
        # Give top row (maps) 3x the height of bottom row (coverage graph)
        gs = self.realtime_fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.25, wspace=0.2)

        self.realtime_axes = {
            'grid': self.realtime_fig.add_subplot(gs[0, 0]),
            'frontier': self.realtime_fig.add_subplot(gs[0, 1]),
            'coverage': self.realtime_fig.add_subplot(gs[1, :])
        }

        title = 'Subterranean Maze Mapping (WITH DIRECTION BIAS)\n(Click on grid map to control RED robot)'
        self.realtime_fig.suptitle(title, fontsize=14, fontweight='bold')
        self.realtime_fig.canvas.mpl_connect('button_press_event', self.on_map_click)
        plt.show(block=False)

    def on_map_click(self, event):
        if event.inaxes == self.realtime_axes['grid']:
            x, y = event.xdata, event.ydata
            if (self.map_bounds['x_min'] <= x <= self.map_bounds['x_max'] and
                self.map_bounds['y_min'] <= y <= self.map_bounds['y_max']):
                
                self.robots[0].goal = (x, y)
                self.robots[0].manual_control = True
                
                pos, _ = p.getBasePositionAndOrientation(self.robots[0].id)
                start = self.world_to_grid(pos[0], pos[1])
                end = self.world_to_grid(x, y)
                
                print(f"Planning path to ({x:.2f}, {y:.2f})...")
                self.robots[0].path = self.plan_path_astar(start, end)
                print(f"Path found: {len(self.robots[0].path)} steps")

    def update_realtime_visualization(self, step):
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

        # Plot robots with direction arrows
        for robot, color in zip(self.robots, ['red', 'green', 'blue']):
            if robot.trajectory:
                traj = np.array(robot.trajectory)
                ax_grid.plot(traj[:, 0], traj[:, 1], c=color, linewidth=1.5, alpha=0.6)
            
            # Plot planned path
            if robot.path:
                path_world = [self.grid_to_world(p[0], p[1]) for p in robot.path]
                path_arr = np.array(path_world)
                ax_grid.plot(path_arr[:, 0], path_arr[:, 1], c=color, linestyle=':', linewidth=2)

            pos, _ = p.getBasePositionAndOrientation(robot.id)
            ax_grid.scatter(pos[0], pos[1], c=color, s=100, marker='^',
                          edgecolors='black', linewidths=1.5, zorder=5)
            
            # NEW: Draw exploration direction arrow
            arrow_len = 1.5
            ax_grid.arrow(pos[0], pos[1], 
                         robot.exploration_direction[0] * arrow_len,
                         robot.exploration_direction[1] * arrow_len,
                         head_width=0.3, head_length=0.2, fc=color, ec='black',
                         alpha=0.7, zorder=6)

            if robot.goal is not None:
                ax_grid.scatter(robot.goal[0], robot.goal[1], c=color, s=150,
                              marker='X', edgecolors='white', linewidths=2, zorder=6)

        bounds_margin = 5
        ax_grid.set_xlim(self.map_bounds['x_min'] - bounds_margin, self.map_bounds['x_max'] + bounds_margin)
        ax_grid.set_ylim(self.map_bounds['y_min'] - bounds_margin, self.map_bounds['y_max'] + bounds_margin)
        ax_grid.set_aspect('equal')
        ax_grid.grid(True, alpha=0.3)
        ax_grid.set_title('Occupancy Grid (Arrows = Exploration Direction)')

        coverage = self.calculate_coverage()
        ax_grid.text(0.02, 0.98, f'Coverage: {coverage:.1f}%',
                    transform=ax_grid.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        # Update frontier map
        ax_frontier = self.realtime_axes['frontier']
        frontiers_data = self.detect_frontiers()
        
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

        if frontiers_data:
            frontier_points = [f['pos'] for f in frontiers_data]
            frontier_array = np.array(frontier_points)
            ax_frontier.scatter(frontier_array[:, 0], frontier_array[:, 1],
                              c='yellow', s=50, marker='o', edgecolors='red',
                              linewidths=2, label='Frontier Targets', zorder=10)

        for robot, color in zip(self.robots, ['red', 'green', 'blue']):
            pos, _ = p.getBasePositionAndOrientation(robot.id)
            ax_frontier.scatter(pos[0], pos[1], c=color, s=150, marker='^',
                              edgecolors='black', linewidths=2, zorder=6)

        ax_frontier.set_xlim(self.map_bounds['x_min'] - bounds_margin, self.map_bounds['x_max'] + bounds_margin)
        ax_frontier.set_ylim(self.map_bounds['y_min'] - bounds_margin, self.map_bounds['y_max'] + bounds_margin)
        ax_frontier.set_aspect('equal')
        ax_frontier.grid(True, alpha=0.3)
        ax_frontier.set_title(f'Frontier Detection\n({len(frontiers_data)} targets)')
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
        ax_coverage.set_title('Coverage Progress (Direction Bias Enabled)')
        ax_coverage.grid(True, alpha=0.3)
        ax_coverage.set_ylim(0, 100)
        ax_coverage.set_xlim(0, max(2000, step))

        self.realtime_fig.canvas.draw()
        self.realtime_fig.canvas.flush_events()
        plt.pause(0.001)

    def run_simulation(self, steps=5000, scan_interval=10, use_gui=True, realtime_viz=True, viz_update_interval=50):
        print("Starting subterranean maze mapping simulation...")
        print("*** DIRECTION BIAS ENABLED ***")
        print(f"  - Direction weight: {self.direction_bias_weight}")
        print(f"  - Size weight: {self.size_weight}")
        print(f"  - Distance weight: {self.distance_weight}")
        
        if steps is None:
            print("Running unlimited steps (press Ctrl+C to stop)...")
        else:
            print(f"Running for {steps} steps...")

        if realtime_viz:
            print("Setting up real-time visualization...")
            self.setup_realtime_visualization()

        step = 0
        while True:
            if steps is not None and step >= steps:
                break

            # COORDINATION STEP (Global Planner)
            if step % 20 == 0:
                self.assign_global_goals()

            # CONTROL STEP (Local Planner)
            for robot in self.robots:
                self.exploration_logic(robot, step)

            # SENSING STEP
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
                print(f"Progress: Step {step} | Coverage: {coverage:.1f}% | Frontiers: {len(self.detect_frontiers())}")

            step += 1

        if realtime_viz:
            self.update_realtime_visualization(step)
            print("\nReal-time visualization complete.")

        print("\nSimulation complete!")
        final_coverage = self.calculate_coverage()
        print(f"Final Coverage: {final_coverage:.2f}%")
        print(f"Explored Free Cells: {int(final_coverage/100 * self.total_free_cells)}/{int(self.total_free_cells)}")

    def cleanup(self):
        self.env.close()


def main():
    print("=" * 60)
    print("Multi-Robot Subterranean Maze Coverage Mapping")
    print("IMPROVED VERSION WITH DIRECTION BIAS")
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

    print("\nEnvironment types:")
    print("  1. Maze (complex maze with walls)")
    print("  2. Blank box (empty room with single wall in middle)")
    print("  3. Cave (organic cellular automata)")
    print("  4. Tunnel (long winding corridor)")
    print("  5. Rooms (dungeon with connected chambers)")
    env_type_input = input("Choose environment type (1-5, default=1): ").strip()

    if env_type_input == '2':
        env_type = 'blank_box'
    elif env_type_input == '3':
        env_type = 'cave'
    elif env_type_input == '4':
        env_type = 'tunnel'
    elif env_type_input == '5':
        env_type = 'rooms'
    else:
        env_type = 'maze'

    gui_input = input("Show PyBullet 3D window? (y/n, default=n): ").strip().lower()
    use_gui = gui_input == 'y'

    steps_input = input("Number of simulation steps (press Enter for unlimited): ").strip()
    if steps_input.isdigit():
        max_steps = int(steps_input)
    else:
        max_steps = None

    print(f"\nCreating {maze_size}x{maze_size} {env_type} with {cell_size}m cells...")
    
    mapper = SubterraneanMapper(
        use_gui=use_gui,
        maze_size=(maze_size, maze_size),
        cell_size=cell_size,
        env_seed=env_seed,
        env_type=env_type
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