import pybullet as p
import numpy as np
import heapq
from collections import defaultdict


class Robot:
    MAX_TRAJECTORY_LENGTH = 200
    TRAJECTORY_TRIM_SIZE = 150

    def __init__(self, robot_id, position, color):
        # 1. Basic identification
        self.id = robot_id
        self.position = position
        self.color = color

        # 2. Sensor and trajectory data
        self.lidar_data = []
        self.trajectory = []

        # 3. Autonomy state
        self.goal = None
        self.path = []
        self.mode = 'IDLE'

        # 4. Goal management
        self.blacklisted_goals = {}
        self.goal_attempts = 0
        self.max_goal_attempts = 3

        # 5. Exploration direction tracking
        self.exploration_direction = np.array([1.0, 0.0])
        self.direction_history = []
        self.direction_smoothing_window = 10

        # 6. Stuck detection
        self.stuck_counter = 0
        self.last_position = np.array(position[:2])

        # 7. Global navigation graph
        self.home_position = np.array(position[:2])
        self.global_graph_nodes = [tuple(position[:2])]
        self.global_graph_edges = set()
        self.last_graph_node_idx = 0
        self.waypoint_spacing = 3.0

        # 8. Spatial hashing for optimization
        self._node_grid = defaultdict(list)
        self._node_grid_cell_size = self.waypoint_spacing
        self._add_node_to_spatial_hash(0, position[:2])

    # === SPATIAL HASHING METHODS ===

    def _get_spatial_hash_cell(self, pos):
        return (int(pos[0] / self._node_grid_cell_size),
                int(pos[1] / self._node_grid_cell_size))

    def _add_node_to_spatial_hash(self, node_idx, pos):
        cell = self._get_spatial_hash_cell(pos)
        self._node_grid[cell].append(node_idx)

    def _find_nearest_node_fast(self, pos):
        # 1. Get cell for position
        cell = self._get_spatial_hash_cell(pos)

        # 2. Check current cell and 8 neighbors
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

    # === EXPLORATION DIRECTION ===

    def update_exploration_direction(self):
        if len(self.trajectory) < 2:
            return

        # 1. Get recent trajectory segment
        recent_points = self.trajectory[-min(20, len(self.trajectory)):]

        if len(recent_points) >= 2:
            # 2. Calculate direction from start to end
            start = np.array(recent_points[0])
            end = np.array(recent_points[-1])
            direction = end - start

            norm = np.linalg.norm(direction)
            if norm > 0.5:
                direction = direction / norm

                # 3. Apply low-pass filter for smoothing
                alpha = 0.3
                self.exploration_direction = (
                    alpha * direction +
                    (1 - alpha) * self.exploration_direction
                )
                self.exploration_direction /= np.linalg.norm(self.exploration_direction)

    def get_heading_vector(self):
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        return np.array([np.cos(yaw), np.sin(yaw)])

    # === STUCK DETECTION ===

    def check_if_stuck(self, threshold=0.3, stuck_limit=200):
        # 1. Get current position
        pos, _ = p.getBasePositionAndOrientation(self.id)
        current_pos = np.array([pos[0], pos[1]])

        # 2. Check distance moved since last check
        distance_moved = np.linalg.norm(current_pos - self.last_position)

        # 3. Update stuck counter
        if distance_moved < threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_position = current_pos

        return self.stuck_counter > stuck_limit

    def reset_stuck_state(self):
        self.stuck_counter = 0
        pos, _ = p.getBasePositionAndOrientation(self.id)
        self.last_position = np.array([pos[0], pos[1]])

    def cleanup_blacklist(self, current_step):
        expired = [pos for pos, expiry in self.blacklisted_goals.items() if current_step >= expiry]
        for pos in expired:
            del self.blacklisted_goals[pos]

    # === GLOBAL GRAPH MANAGEMENT ===

    def update_global_graph(self):
        # 1. Get current position
        pos, _ = p.getBasePositionAndOrientation(self.id)
        current_pos = (pos[0], pos[1])

        # 2. Find nearest existing node
        nearest_node_idx, min_dist = self._find_nearest_node_fast(current_pos)

        # 3. Add new node if far enough from existing nodes
        if min_dist >= self.waypoint_spacing:
            new_node_idx = len(self.global_graph_nodes)
            self.global_graph_nodes.append(current_pos)
            self._add_node_to_spatial_hash(new_node_idx, current_pos)

            # 4. Create edge to last node
            self.global_graph_edges.add((self.last_graph_node_idx, new_node_idx))

            # 5. Create edge to nearest node if different
            if nearest_node_idx != -1 and nearest_node_idx != self.last_graph_node_idx:
                edge = (min(nearest_node_idx, new_node_idx), max(nearest_node_idx, new_node_idx))
                self.global_graph_edges.add(edge)

            self.last_graph_node_idx = new_node_idx
        else:
            self.last_graph_node_idx = nearest_node_idx

    def plan_path_on_global_graph(self, target_pos):
        if len(self.global_graph_nodes) < 2:
            return [target_pos]

        # 1. Find start and goal nodes
        pos, _ = p.getBasePositionAndOrientation(self.id)
        current_pos = (pos[0], pos[1])
        start_node_idx, _ = self._find_nearest_node_fast(current_pos)
        goal_node_idx, _ = self._find_nearest_node_fast(target_pos)

        # 2. Build adjacency list with edge weights
        adjacency = {i: [] for i in range(len(self.global_graph_nodes))}
        for edge in self.global_graph_edges:
            n1, n2 = edge
            p1 = self.global_graph_nodes[n1]
            p2 = self.global_graph_nodes[n2]
            weight = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            adjacency[n1].append((n2, weight))
            adjacency[n2].append((n1, weight))

        # 3. Run Dijkstra's algorithm
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

        # 4. Reconstruct path from goal to start
        if goal_node_idx not in came_from:
            return [target_pos]

        path = []
        current = goal_node_idx
        while current is not None:
            path.append(self.global_graph_nodes[current])
            current = came_from.get(current)
        path.reverse()

        # 5. Add final target if not at a graph node
        if path and np.sqrt((path[-1][0] - target_pos[0])**2 + (path[-1][1] - target_pos[1])**2) > 0.5:
            path.append(target_pos)

        return path

    # === LIDAR SENSING ===

    def get_lidar_scan(self, num_rays=360, max_range=10):
        # 1. Get robot pose
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        # 2. Calculate lidar position
        lidar_height_offset = 0.3
        lidar_z = pos[2] + lidar_height_offset

        # 3. Generate ray directions
        angles = yaw + np.linspace(0, 2.0 * np.pi, num_rays, endpoint=False)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        # 4. Build ray endpoints
        ray_from = [[pos[0], pos[1], lidar_z]] * num_rays
        ray_to = [
            [pos[0] + max_range * cos_angles[i],
             pos[1] + max_range * sin_angles[i],
             lidar_z]
            for i in range(num_rays)
        ]

        # 5. Perform batch raycasting
        results = p.rayTestBatch(ray_from, ray_to)

        # 6. Process ray results
        scan_points = []
        for i, result in enumerate(results):
            hit_object_id = result[0]
            hit_fraction = result[2]

            if hit_fraction < 1.0 and hit_object_id != self.id:
                hit_position = result[3]
                scan_points.append((hit_position[0], hit_position[1], True))
            else:
                scan_points.append((ray_to[i][0], ray_to[i][1], False))

        self.lidar_data = scan_points

        # 7. Update trajectory with bounded length
        self.trajectory.append((pos[0], pos[1]))
        if len(self.trajectory) > Robot.MAX_TRAJECTORY_LENGTH:
            self.trajectory = self.trajectory[-Robot.TRAJECTORY_TRIM_SIZE:]

        # 8. Update exploration direction
        self.update_exploration_direction()

        return scan_points

    # === MOVEMENT CONTROL ===

    def move(self, linear_vel, angular_vel):
        # 1. Get robot orientation
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        # 2. Convert to velocity components
        vx = linear_vel * np.cos(yaw)
        vy = linear_vel * np.sin(yaw)

        # 3. Apply velocity to physics
        p.resetBaseVelocity(self.id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, angular_vel])

    def follow_path(self, mapper):
        if not self.path and self.goal is None:
            return 0.0, 0.0

        target_pos = self.goal

        # 1. Get next waypoint if path exists
        if self.path:
            next_wp = self.path[0]
            wx, wy = mapper.grid_to_world(next_wp[0], next_wp[1])
            target_pos = (wx, wy)

            # 2. Check if waypoint reached
            pos, _ = p.getBasePositionAndOrientation(self.id)
            dist = np.sqrt((wx - pos[0])**2 + (wy - pos[1])**2)

            if dist < 0.6:
                self.path.pop(0)
                if self.path:
                    return self.follow_path(mapper)

        # 3. Calculate direction to target
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        dx = target_pos[0] - pos[0]
        dy = target_pos[1] - pos[1]
        distance = np.sqrt(dx**2 + dy**2)

        # 4. Check if final goal reached
        if distance < 0.5 and not self.path:
            self.goal = None
            return 0.0, 0.0

        # 5. Calculate desired heading
        desired_angle = np.arctan2(dy, dx)
        angle_diff = desired_angle - yaw
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        # 6. Compute velocities with limits
        linear_vel = min(8.0, distance * 2.0)
        angular_vel = np.clip(angle_diff * 4.0, -4.0, 4.0)

        # 7. Slow down for sharp turns
        if abs(angle_diff) > 0.5:
            linear_vel *= 0.3

        return linear_vel, angular_vel
