"""
Robot class for multi-robot coverage mapping simulation.

Handles individual robot behavior including:
- LIDAR sensing
- Path following and navigation
- Stuck detection
- Goal blacklisting
- Global graph construction for return-to-home planning
"""

import pybullet as p
import numpy as np
import heapq


class Robot:
    """
    Autonomous robot for coverage mapping with LIDAR sensing and path following.

    Features:
    - Frontier-based exploration with direction bias
    - Goal blacklisting to avoid unreachable locations
    - Stuck detection with configurable thresholds
    - Global graph building for efficient return-to-home navigation
    """

    # Movement and control constants
    MAX_LINEAR_VELOCITY = 8.0  # m/s (increased for faster exploration)
    MAX_ANGULAR_VELOCITY = 4.0  # rad/s
    ANGULAR_GAIN = 4.0  # Proportional control gain for steering
    LINEAR_GAIN = 2.0  # Proportional control gain for forward motion
    TURN_SPEED_FACTOR = 0.3  # Speed reduction when turning sharply (30%)
    SHARP_TURN_THRESHOLD = 0.5  # rad - angle threshold for sharp turns

    # Distance thresholds
    WAYPOINT_ACCEPTANCE_RADIUS = 0.6  # meters - distance to consider waypoint reached
    GOAL_ACCEPTANCE_RADIUS = 0.5  # meters - distance to consider goal reached
    WAYPOINT_SPACING = 3.0  # meters - spacing between global graph nodes
    GRAPH_CONNECTION_THRESHOLD = 0.5  # meters - threshold for connecting to existing nodes

    # Stuck detection constants
    STUCK_MOVEMENT_THRESHOLD = 0.3  # meters - minimum movement to not be considered stuck
    STUCK_COUNTER_LIMIT = 200  # steps - increased to prevent false positives on slow turns
    MAX_GOAL_ATTEMPTS = 3  # number of attempts before giving up on a goal

    # Exploration direction constants
    EXPLORATION_SMOOTHING_WINDOW = 10  # number of samples to average
    TRAJECTORY_LOOKBACK = 20  # number of recent points for direction calculation
    DIRECTION_UPDATE_THRESHOLD = 0.5  # meters - minimum movement to update direction
    DIRECTION_SMOOTHING_ALPHA = 0.3  # low-pass filter coefficient

    # LIDAR constants
    LIDAR_HEIGHT_OFFSET = 0.3  # meters - mount LIDAR higher to avoid self-collision
    DEFAULT_LIDAR_RAYS = 360
    DEFAULT_LIDAR_RANGE = 10  # meters

    def __init__(self, robot_id, position, color):
        """
        Initialize a robot.

        Args:
            robot_id: PyBullet body ID
            position: Initial [x, y, z] position
            color: RGBA color array [r, g, b, a]
        """
        self.id = robot_id
        self.position = position
        self.color = color
        self.lidar_data = []
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

        # Stuck detection
        self.stuck_counter = 0
        self.last_position = np.array(position[:2])
        self.goal_attempts = 0

        # Global graph for navigation
        self.home_position = np.array(position[:2])  # Remember start position
        self.global_graph_nodes = [tuple(position[:2])]  # List of (x, y) waypoints
        self.global_graph_edges = []  # List of (node_idx1, node_idx2) connections
        self.last_graph_node_idx = 0  # Index of last node added to graph

    def update_exploration_direction(self):
        """
        Update the robot's exploration direction based on recent movement.
        Uses a low-pass filter over recent trajectory points (like GBPlanner).
        """
        if len(self.trajectory) < 2:
            return

        # Get recent trajectory segment
        recent_points = self.trajectory[-min(self.TRAJECTORY_LOOKBACK, len(self.trajectory)):]

        if len(recent_points) >= 2:
            # Calculate direction from older point to newer point
            start = np.array(recent_points[0])
            end = np.array(recent_points[-1])
            direction = end - start

            norm = np.linalg.norm(direction)
            if norm > self.DIRECTION_UPDATE_THRESHOLD:  # Only update if we've moved meaningfully
                direction = direction / norm

                # Smooth with existing direction (low-pass filter)
                self.exploration_direction = (
                    self.DIRECTION_SMOOTHING_ALPHA * direction +
                    (1 - self.DIRECTION_SMOOTHING_ALPHA) * self.exploration_direction
                )
                # Renormalize
                self.exploration_direction /= np.linalg.norm(self.exploration_direction)

    def get_heading_vector(self):
        """Get the robot's current facing direction from physics."""
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        return np.array([np.cos(yaw), np.sin(yaw)])

    def check_if_stuck(self, threshold=None, stuck_limit=None):
        """
        Check if robot is stuck (not moving despite having a goal).

        Args:
            threshold: Movement threshold in meters (uses class constant if None)
            stuck_limit: Counter limit in steps (uses class constant if None)

        Returns:
            bool: True if robot is stuck
        """
        if threshold is None:
            threshold = self.STUCK_MOVEMENT_THRESHOLD
        if stuck_limit is None:
            stuck_limit = self.STUCK_COUNTER_LIMIT

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
        """
        pos, _ = p.getBasePositionAndOrientation(self.id)
        current_pos = (pos[0], pos[1])

        # Check distance to all existing nodes
        min_dist = float('inf')
        nearest_node_idx = 0

        for i, node in enumerate(self.global_graph_nodes):
            dist = np.sqrt((current_pos[0] - node[0])**2 + (current_pos[1] - node[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node_idx = i

        # Only add new node if far enough from all existing nodes
        if min_dist >= self.WAYPOINT_SPACING:
            new_node_idx = len(self.global_graph_nodes)
            self.global_graph_nodes.append(current_pos)

            # Connect to the last node we were at (maintains path continuity)
            self.global_graph_edges.append((self.last_graph_node_idx, new_node_idx))

            # Also connect to nearest node if different (creates shortcuts)
            if nearest_node_idx != self.last_graph_node_idx:
                edge = (nearest_node_idx, new_node_idx)
                reverse_edge = (new_node_idx, nearest_node_idx)
                if edge not in self.global_graph_edges and reverse_edge not in self.global_graph_edges:
                    self.global_graph_edges.append(edge)

            self.last_graph_node_idx = new_node_idx
        else:
            # Update last node to nearest if we're close to an existing node
            self.last_graph_node_idx = nearest_node_idx

    def plan_path_on_global_graph(self, target_pos):
        """
        Plan a path on the global graph to reach target_pos.
        Uses Dijkstra's algorithm on the graph.

        Args:
            target_pos: Target (x, y) position tuple

        Returns:
            List of (x, y) waypoints to follow
        """
        if len(self.global_graph_nodes) < 2:
            return [target_pos]

        # Find nearest node to current position
        pos, _ = p.getBasePositionAndOrientation(self.id)
        current_pos = (pos[0], pos[1])

        start_node_idx = 0
        min_dist = float('inf')
        for i, node in enumerate(self.global_graph_nodes):
            dist = np.sqrt((current_pos[0] - node[0])**2 + (current_pos[1] - node[1])**2)
            if dist < min_dist:
                min_dist = dist
                start_node_idx = i

        # Find nearest node to target
        goal_node_idx = 0
        min_dist = float('inf')
        for i, node in enumerate(self.global_graph_nodes):
            dist = np.sqrt((target_pos[0] - node[0])**2 + (target_pos[1] - node[1])**2)
            if dist < min_dist:
                min_dist = dist
                goal_node_idx = i

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
        if path and np.sqrt((path[-1][0] - target_pos[0])**2 + (path[-1][1] - target_pos[1])**2) > self.GRAPH_CONNECTION_THRESHOLD:
            path.append(target_pos)

        return path

    def get_lidar_scan(self, num_rays=None, max_range=None):
        """
        Simulate LIDAR by casting rays in 360 degrees.

        Args:
            num_rays: Number of rays to cast (uses default if None)
            max_range: Maximum range in meters (uses default if None)

        Returns:
            List of (x, y, is_hit) tuples for each ray
        """
        if num_rays is None:
            num_rays = self.DEFAULT_LIDAR_RAYS
        if max_range is None:
            max_range = self.DEFAULT_LIDAR_RANGE

        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        # Mount Lidar Higher to avoid self-collision
        lidar_z = pos[2] + self.LIDAR_HEIGHT_OFFSET

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

        self.lidar_data.extend(scan_points)
        self.trajectory.append((pos[0], pos[1]))

        # Update exploration direction after recording trajectory
        self.update_exploration_direction()

        return scan_points

    def move(self, linear_vel, angular_vel):
        """
        Move robot using physics velocity control.

        Args:
            linear_vel: Forward velocity in m/s
            angular_vel: Angular velocity in rad/s
        """
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        vx = linear_vel * np.cos(yaw)
        vy = linear_vel * np.sin(yaw)

        p.resetBaseVelocity(self.id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, angular_vel])

    def follow_path(self, mapper):
        """
        Follow the A* path using pure pursuit control.
        If path is empty but goal exists, drive directly to goal.

        Args:
            mapper: CoverageMapper instance (needed for grid_to_world conversion)

        Returns:
            Tuple of (linear_velocity, angular_velocity) commands
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
            if dist < self.WAYPOINT_ACCEPTANCE_RADIUS:
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
        if distance < self.GOAL_ACCEPTANCE_RADIUS and not self.path:
            self.goal = None
            return 0.0, 0.0

        desired_angle = np.arctan2(dy, dx)
        angle_diff = desired_angle - yaw
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        # Control Limits
        linear_vel = min(self.MAX_LINEAR_VELOCITY, distance * self.LINEAR_GAIN)
        angular_vel = np.clip(angle_diff * self.ANGULAR_GAIN, -self.MAX_ANGULAR_VELOCITY, self.MAX_ANGULAR_VELOCITY)

        # Slow down if turning sharply
        if abs(angle_diff) > self.SHARP_TURN_THRESHOLD:
            linear_vel *= self.TURN_SPEED_FACTOR

        return linear_vel, angular_vel
