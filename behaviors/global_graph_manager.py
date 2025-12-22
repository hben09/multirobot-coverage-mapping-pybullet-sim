"""
GlobalGraphManager - Manages global navigation graph and path planning.

Builds and maintains a topological graph of explored space,
and plans paths using Dijkstra's algorithm.
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional
from robot.robot_state import RobotState
from robot.robot_hardware import RobotHardware


class GlobalGraphManager:
    """
    Manages global navigation graph for long-range path planning.

    Maintains a topological graph of waypoints with spatial hashing
    for efficient nearest-neighbor queries.
    """

    def __init__(self, waypoint_spacing: float = 3.0):
        """
        Initialize the global graph manager.

        Args:
            waypoint_spacing: Minimum distance between graph nodes (meters)
        """
        self.waypoint_spacing = waypoint_spacing

    def update_graph(self, state: RobotState, hardware: RobotHardware):
        """
        Update the global graph with the robot's current position.

        Adds new nodes to the graph when the robot moves far enough
        from existing nodes, maintaining connectivity.

        Args:
            state: Robot state containing graph data
            hardware: Hardware interface to get current position
        """
        # 1. Get current position from hardware
        pos_array, _ = hardware.get_pose()
        current_pos = (pos_array[0], pos_array[1])

        # 2. Find nearest existing node
        nearest_node_idx, min_dist = self._find_nearest_node_fast(state, current_pos)

        # 3. Add new node if far enough from existing nodes
        if min_dist >= self.waypoint_spacing:
            new_node_idx = len(state.global_graph_nodes)
            state.global_graph_nodes.append(current_pos)
            self._add_node_to_spatial_hash(state, new_node_idx, current_pos)

            # 4. Create edge to last node
            state.global_graph_edges.add((state.last_graph_node_idx, new_node_idx))

            # 5. Create edge to nearest node if different
            if nearest_node_idx != -1 and nearest_node_idx != state.last_graph_node_idx:
                edge = (min(nearest_node_idx, new_node_idx), max(nearest_node_idx, new_node_idx))
                state.global_graph_edges.add(edge)

            state.last_graph_node_idx = new_node_idx
        else:
            state.last_graph_node_idx = nearest_node_idx

    def plan_path(
        self,
        state: RobotState,
        hardware: RobotHardware,
        target_pos: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        """
        Plan a path on the global graph using Dijkstra's algorithm.

        Args:
            state: Robot state containing graph data
            hardware: Hardware interface to get current position
            target_pos: Target position (x, y) in world coordinates

        Returns:
            List of waypoints (x, y) from current position to target
            Returns [target_pos] if graph is too small or path not found
        """
        if len(state.global_graph_nodes) < 2:
            return [target_pos]

        # 1. Find start and goal nodes from hardware
        pos_array, _ = hardware.get_pose()
        current_pos = (pos_array[0], pos_array[1])
        start_node_idx, _ = self._find_nearest_node_fast(state, current_pos)
        goal_node_idx, _ = self._find_nearest_node_fast(state, target_pos)

        # 2. Build adjacency list with edge weights
        adjacency = {i: [] for i in range(len(state.global_graph_nodes))}
        for edge in state.global_graph_edges:
            n1, n2 = edge
            p1 = state.global_graph_nodes[n1]
            p2 = state.global_graph_nodes[n2]
            weight = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            adjacency[n1].append((n2, weight))
            adjacency[n2].append((n1, weight))

        # 3. Run Dijkstra's algorithm
        distances = {i: float('inf') for i in range(len(state.global_graph_nodes))}
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
            path.append(state.global_graph_nodes[current])
            current = came_from.get(current)
        path.reverse()

        # 5. Add final target if not at a graph node
        if path and np.sqrt((path[-1][0] - target_pos[0])**2 + (path[-1][1] - target_pos[1])**2) > 0.5:
            path.append(target_pos)

        return path

    # === SPATIAL HASHING HELPER METHODS ===

    def _get_spatial_hash_cell(self, state: RobotState, pos: Tuple[float, float]) -> Tuple[int, int]:
        """Get the spatial hash cell for a position."""
        return (
            int(pos[0] / state._node_grid_cell_size),
            int(pos[1] / state._node_grid_cell_size)
        )

    def _add_node_to_spatial_hash(self, state: RobotState, node_idx: int, pos: Tuple[float, float]):
        """Add a node to the spatial hash grid."""
        cell = self._get_spatial_hash_cell(state, pos)
        state._node_grid[cell].append(node_idx)

    def _find_nearest_node_fast(
        self,
        state: RobotState,
        pos: Tuple[float, float]
    ) -> Tuple[int, float]:
        """
        Find the nearest node using spatial hashing.

        Args:
            state: Robot state containing graph and spatial hash
            pos: Position to query

        Returns:
            Tuple of (node_index, distance)
            Returns (-1, inf) if no nodes exist
        """
        # 1. Get cell for position
        cell = self._get_spatial_hash_cell(state, pos)

        # 2. Check current cell and 8 neighbors
        min_dist_sq = float('inf')
        nearest_idx = -1

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_cell = (cell[0] + dx, cell[1] + dy)
                for node_idx in state._node_grid.get(check_cell, []):
                    node = state.global_graph_nodes[node_idx]
                    dist_sq = (pos[0] - node[0])**2 + (pos[1] - node[1])**2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        nearest_idx = node_idx

        return nearest_idx, np.sqrt(min_dist_sq)
