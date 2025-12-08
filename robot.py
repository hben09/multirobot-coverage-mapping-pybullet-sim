import pybullet as p
import numpy as np
import heapq
from collections import defaultdict


class Robot:
    # Class constants for bounded data structures
    MAX_TRAJECTORY_LENGTH = 200  # Rolling window for trajectory
    TRAJECTORY_TRIM_SIZE = 150   # Trim to this size when exceeded
    
    def __init__(self, robot_id, position, color):
        self.id = robot_id
        self.position = position
        self.color = color
        self.lidar_data = []  # Now replaced each scan, not extended
        self.trajectory = []
        
        # Autonomy State
        self.goal = None
        self.path = []  # List of grid waypoints from A*
        self.manual_control = False
        self.mode = 'IDLE'  # IDLE, EXPLORING, RETURNING_HOME
        
        # Blacklist for unreachable goals (grid_pos -> expiration_step)
        self.blacklisted_goals = {}
        
        # Direction tracking for exploration bias
        self.exploration_direction = np.array([1.0, 0.0])  # Initial heading (east)
        self.direction_history = []  # Track recent movement directions
        self.direction_smoothing_window = 10  # Number of samples to average
        
        # Stuck detection
        self.stuck_counter = 0
        self.last_position = np.array(position[:2])
        self.goal_attempts = 0
        self.max_goal_attempts = 3  # Give up after 3 failed attempts to reach a goal
        
        # Global graph for navigation - OPTIMIZED with spatial hashing
        self.home_position = np.array(position[:2])  # Remember start position
        self.global_graph_nodes = [tuple(position[:2])]  # List of (x, y) waypoints
        self.global_graph_edges = set()  # SET for O(1) membership checks (was list)
        self.last_graph_node_idx = 0  # Index of last node added to graph
        self.waypoint_spacing = 3.0  # Meters between waypoints
        
        # Spatial hash for O(1) average nearest-neighbor lookup
        self._node_grid = defaultdict(list)  # (grid_x, grid_y) -> [node_indices]
        self._node_grid_cell_size = self.waypoint_spacing  # Grid cell = waypoint spacing
        # Add initial node to spatial hash
        self._add_node_to_spatial_hash(0, position[:2])
    
    def _get_spatial_hash_cell(self, pos):
        """Get the spatial hash grid cell for a position."""
        return (int(pos[0] / self._node_grid_cell_size),
                int(pos[1] / self._node_grid_cell_size))
    
    def _add_node_to_spatial_hash(self, node_idx, pos):
        """Add a node to the spatial hash."""
        cell = self._get_spatial_hash_cell(pos)
        self._node_grid[cell].append(node_idx)
    
    def _find_nearest_node_fast(self, pos):
        """Find nearest node using spatial hash - O(1) average case."""
        cell = self._get_spatial_hash_cell(pos)
        
        # Check current cell and all 8 neighbors
        min_dist_sq = float('inf')
        nearest_idx = -1
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_cell = (cell[0] + dx, cell[1] + dy)
                for node_idx in self._node_grid.get(check_cell, []):
                    node = self.global_graph_nodes[node_idx]
                    dist_sq = (pos[0] - node[0])**2 + (pos[1] - node[1])**2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        nearest_idx = node_idx
        
        return nearest_idx, np.sqrt(min_dist_sq)

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

    def check_if_stuck(self, threshold=0.3, stuck_limit=200):
        """
        Check if robot is stuck (not moving despite having a goal).
        Increased limit to 200 to prevent false positives on slow turns.
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
        
    def cleanup_blacklist(self, current_step):
        """Remove expired entries from the goal blacklist."""
        expired = [pos for pos, expiry in self.blacklisted_goals.items() if current_step >= expiry]
        for pos in expired:
            del self.blacklisted_goals[pos]

    def update_global_graph(self):
        """
        Add current position to global graph if far enough from existing nodes.
        Creates edges to connect new nodes to the graph.
        OPTIMIZED: Uses spatial hashing for O(1) nearest-neighbor lookup.
        """
        pos, _ = p.getBasePositionAndOrientation(self.id)
        current_pos = (pos[0], pos[1])
        
        # Use spatial hash for O(1) average nearest-neighbor lookup
        nearest_node_idx, min_dist = self._find_nearest_node_fast(current_pos)
        
        # Only add new node if far enough from all existing nodes
        if min_dist >= self.waypoint_spacing:
            new_node_idx = len(self.global_graph_nodes)
            self.global_graph_nodes.append(current_pos)
            
            # Add to spatial hash
            self._add_node_to_spatial_hash(new_node_idx, current_pos)
            
            # Connect to the last node we were at (maintains path continuity)
            self.global_graph_edges.add((self.last_graph_node_idx, new_node_idx))
            
            # Also connect to nearest node if different (creates shortcuts)
            if nearest_node_idx != -1 and nearest_node_idx != self.last_graph_node_idx:
                # Normalize edge representation (smaller index first) for consistent hashing
                edge = (min(nearest_node_idx, new_node_idx), max(nearest_node_idx, new_node_idx))
                self.global_graph_edges.add(edge)  # O(1) with set
            
            self.last_graph_node_idx = new_node_idx
        else:
            # Update last node to nearest if we're close to an existing node
            self.last_graph_node_idx = nearest_node_idx

    def plan_path_on_global_graph(self, target_pos):
        """
        Plan a path on the global graph to reach target_pos.
        Uses Dijkstra's algorithm on the graph.
        OPTIMIZED: Uses spatial hashing for nearest-neighbor lookups.
        
        Returns: List of (x, y) waypoints to follow
        """
        if len(self.global_graph_nodes) < 2:
            return [target_pos]
        
        # Find nearest node to current position using spatial hash
        pos, _ = p.getBasePositionAndOrientation(self.id)
        current_pos = (pos[0], pos[1])
        start_node_idx, _ = self._find_nearest_node_fast(current_pos)
        
        # Find nearest node to target using spatial hash
        goal_node_idx, _ = self._find_nearest_node_fast(target_pos)
        
        # Build adjacency list
        adjacency = {i: [] for i in range(len(self.global_graph_nodes))}
        for edge in self.global_graph_edges:
            n1, n2 = edge
            # Calculate edge weight (distance)
            p1 = self.global_graph_nodes[n1]
            p2 = self.global_graph_nodes[n2]
            weight = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            adjacency[n1].append((n2, weight))
            adjacency[n2].append((n1, weight))  # Undirected graph
        
        # Dijkstra's algorithm
        distances = {i: float('inf') for i in range(len(self.global_graph_nodes))}
        distances[start_node_idx] = 0
        came_from = {start_node_idx: None}
        
        frontier = [(0, start_node_idx)]
        visited = set()
        
        while frontier:
            current_dist, current_node = heapq.heappop(frontier)
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            if current_node == goal_node_idx:
                break
            
            for neighbor, weight in adjacency[current_node]:
                if neighbor in visited:
                    continue
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    came_from[neighbor] = current_node
                    heapq.heappush(frontier, (new_dist, neighbor))
        
        # Reconstruct path
        if goal_node_idx not in came_from:
            return [target_pos]  # No path found, just go directly
        
        path = []
        current = goal_node_idx
        while current is not None:
            path.append(self.global_graph_nodes[current])
            current = came_from.get(current)
        path.reverse()
        
        # Add final target position if not at a node
        if path and np.sqrt((path[-1][0] - target_pos[0])**2 + (path[-1][1] - target_pos[1])**2) > 0.5:
            path.append(target_pos)
        
        return path

    def get_lidar_scan(self, num_rays=360, max_range=10):
        """Simulate lidar by casting rays in 360 degrees - OPTIMIZED with numpy"""
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        # Mount Lidar Higher to avoid self-collision
        lidar_height_offset = 0.3 
        lidar_z = pos[2] + lidar_height_offset

        # OPTIMIZED: Use numpy for vectorized angle calculations
        angles = yaw + np.linspace(0, 2.0 * np.pi, num_rays, endpoint=False)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        
        # Build ray arrays
        ray_from = [[pos[0], pos[1], lidar_z]] * num_rays
        ray_to = [
            [pos[0] + max_range * cos_angles[i],
             pos[1] + max_range * sin_angles[i],
             lidar_z]
            for i in range(num_rays)
        ]

        results = p.rayTestBatch(ray_from, ray_to)

        scan_points = []
        for i, result in enumerate(results):
            hit_object_id = result[0]
            hit_fraction = result[2]
            
            # Check if hit something
            if hit_fraction < 1.0 and hit_object_id != self.id:
                hit_position = result[3]
                # Store (x, y, is_hit=True)
                scan_points.append((hit_position[0], hit_position[1], True))
            else:
                # Ray maxed out - hit nothing
                # Store (x, y, is_hit=False) at max range
                scan_points.append((ray_to[i][0], ray_to[i][1], False))

        # OPTIMIZATION: Replace lidar_data entirely instead of extending
        # The occupancy grid only uses the current scan anyway
        self.lidar_data = scan_points
        
        # OPTIMIZATION: Bound trajectory to prevent unbounded growth
        self.trajectory.append((pos[0], pos[1]))
        if len(self.trajectory) > Robot.MAX_TRAJECTORY_LENGTH:
            self.trajectory = self.trajectory[-Robot.TRAJECTORY_TRIM_SIZE:]
        
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

        # Control Limits (Increased for faster exploration)
        linear_vel = min(8.0, distance * 2.0)  # Max 8.0 m/s (was 4.0)
        angular_vel = np.clip(angle_diff * 4.0, -4.0, 4.0)

        # Slow down if turning sharply (reduced penalty for faster turns)
        if abs(angle_diff) > 0.5:
            linear_vel *= 0.3  # 30% speed when turning (was 10%)

        return linear_vel, angular_vel
