"""
Simulation Logger for Multi-Robot Coverage Mapping

Logs simulation state for offline playback and analysis.
"""

import os
import numpy as np
from datetime import datetime


class SimulationLogger:
    """Logs simulation state for offline playback and analysis."""

    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        self.frames = []
        self.metadata = {}
        self.start_time = None

    def initialize(self, mapper, env_config):
        """Initialize logger with simulation metadata."""
        self.start_time = datetime.now()

        # Store metadata about the simulation
        self.metadata = {
            'timestamp': self.start_time.isoformat(),
            'env_config': env_config,
            'map_bounds': mapper.map_bounds.copy(),
            'grid_resolution': mapper.grid_resolution,
            'num_robots': len(mapper.robots),
            'robot_colors': [robot.color for robot in mapper.robots],
            'robot_home_positions': [robot.home_position.tolist() for robot in mapper.robots],
            'total_free_cells': mapper.total_free_cells,
            'maze_grid': mapper.env.maze_grid.copy(),
        }

        self.frames = []
        print(f"Logger initialized. Will save to: {self.log_dir}")

    def log_frame(self, step, mapper):
        """
        Log a single frame of simulation state.

        Args:
            step: Current simulation step number
            mapper: SubterraneanMapper instance to extract state from
        """
        import pybullet as p

        frame = {
            'step': step,
            'coverage': mapper.calculate_coverage(),
            'returning_home': mapper.returning_home,

            # Occupancy grid (store as dict with tuple keys converted to strings)
            'occupancy_grid': {f"{k[0]},{k[1]}": v for k, v in mapper.occupancy_grid.items()},
            'explored_cells': [list(c) for c in mapper.explored_cells],
            'obstacle_cells': [list(c) for c in mapper.obstacle_cells],

            # Frontier data
            'frontiers': mapper.detect_frontiers(),

            # Robot states
            'robots': []
        }

        for robot in mapper.robots:
            pos, orn = p.getBasePositionAndOrientation(robot.id)
            euler = p.getEulerFromQuaternion(orn)

            robot_state = {
                'position': [pos[0], pos[1], pos[2]],
                'orientation': euler[2],  # yaw
                'goal': list(robot.goal) if robot.goal else None,
                'path': [list(wp) for wp in robot.path] if robot.path else [],
                'mode': robot.mode,
                'exploration_direction': robot.exploration_direction.tolist(),
                'trajectory': [list(t) for t in robot.trajectory[-100:]],  # Last 100 points
                'global_graph_nodes': [list(n) for n in robot.global_graph_nodes],
                'global_graph_edges': list(robot.global_graph_edges),
            }
            frame['robots'].append(robot_state)

        self.frames.append(frame)

    def save(self, filename=None):
        """
        Save logged data to NPZ file.

        Args:
            filename: Optional filename. If None, generates timestamp-based name.

        Returns:
            str: Full path to saved file
        """
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if filename is None:
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"sim_log_{timestamp}.npz"

        filepath = os.path.join(self.log_dir, filename)

        # Convert to numpy-friendly format
        save_data = {
            'metadata': np.array([self.metadata], dtype=object),
            'frames': np.array(self.frames, dtype=object),
            'num_frames': len(self.frames),
        }

        np.savez_compressed(filepath, **save_data)
        print(f"\nSimulation log saved to: {filepath}")
        print(f"  - Total frames: {len(self.frames)}")
        print(f"  - File size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")

        return filepath

    @staticmethod
    def load(filepath):
        """
        Load logged data from NPZ file.

        Args:
            filepath: Path to NPZ log file

        Returns:
            dict: Dictionary containing 'metadata', 'frames', and 'num_frames'
        """
        data = np.load(filepath, allow_pickle=True)
        return {
            'metadata': data['metadata'][0],
            'frames': data['frames'].tolist(),
            'num_frames': int(data['num_frames']),
        }
