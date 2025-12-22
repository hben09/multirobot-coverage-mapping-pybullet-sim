"""
Simulation Logger for Multi-Robot Coverage Mapping

Logs simulation state for offline playback and analysis.
Now uses Delta Encoding for Occupancy Grids to massively reduce file size and save time.
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
        # Cache for delta encoding
        self.last_occupancy_grid = {} 

    def initialize(self, mapper, env_config):
        """Initialize logger with simulation metadata."""
        self.start_time = datetime.now()
        self.last_occupancy_grid = {}

        # Store metadata about the simulation
        self.metadata = {
            'timestamp': self.start_time.isoformat(),
            'env_config': env_config,
            'map_bounds': mapper.map_bounds.copy(),
            'grid_resolution': mapper.grid_manager.grid_resolution,
            'num_robots': len(mapper.robots),
            'robot_colors': [robot.color for robot in mapper.robots],
            'robot_home_positions': [robot.home_position.tolist() for robot in mapper.robots],
            'total_free_cells': mapper.grid_manager.total_free_cells,
            'maze_grid': mapper.env.maze_grid.copy(),
        }

        self.frames = []
        print(f"Logger initialized. Will save to: {self.log_dir}")

    def log_frame(self, step, mapper):
        """
        Log a single frame of simulation state using Delta Encoding.
        
        Instead of saving the whole grid every frame, we only save the cells 
        that have changed since the last frame.
        """
        import pybullet as p

        # 1. Calculate Occupancy Grid Delta
        # Compare current grid with the last saved state
        current_grid = mapper.grid_manager.occupancy_grid
        delta = []

        # We assume cells only change from unknown -> free/obstacle, or rarely free <-> obstacle
        # Since the mapper accumulates keys, we iterate the mapper's grid
        for cell, val in current_grid.items():
            # Check if this cell is new or changed value
            if cell not in self.last_occupancy_grid or self.last_occupancy_grid[cell] != val:
                # Store as integer list [x, y, val] - efficient binary storage
                delta.append([cell[0], cell[1], val])
                # Update our local cache
                self.last_occupancy_grid[cell] = val

        frame = {
            'step': step,
            'coverage': mapper.calculate_coverage(),
            'returning_home': mapper.returning_home,

            # DELTA ENCODING: Store only the changes
            'occupancy_grid_delta': delta,

            # These lists are relatively small, so we store them fully
            'explored_cells': [list(c) for c in mapper.grid_manager.explored_cells],
            'obstacle_cells': [list(c) for c in mapper.grid_manager.obstacle_cells],

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

        print("Compressing and saving log file (this should be fast now)...")
        np.savez_compressed(filepath, **save_data)
        
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"\nSimulation log saved to: {filepath}")
        print(f"  - Total frames: {len(self.frames)}")
        print(f"  - File size: {file_size_mb:.2f} MB")

        return filepath

    @staticmethod
    def load(filepath):
        """
        Load logged data and RECONSTRUCT frames from deltas.
        
        This handles backward compatibility:
        1. If 'occupancy_grid' exists (legacy), use it.
        2. If 'occupancy_grid_delta' exists (new), reconstruct the grid cumulatively.
        
        Returns:
            dict: Data with FULLY RECONSTRUCTED frames.
        """
        print(f"Loading and reconstructing log: {filepath}...")
        data = np.load(filepath, allow_pickle=True)
        
        raw_frames = data['frames'].tolist()
        metadata = data['metadata'][0]
        num_frames = int(data['num_frames'])
        
        reconstructed_frames = []
        accumulated_grid = {} # Keeps track of state across frames
        
        for i, rf in enumerate(raw_frames):
            # Shallow copy to avoid modifying the original numpy array data
            new_frame = rf.copy()
            
            # Handle Delta Encoding
            if 'occupancy_grid_delta' in rf:
                # Apply changes to the accumulator
                for dx, dy, val in rf['occupancy_grid_delta']:
                    accumulated_grid[(dx, dy)] = val
                
                # Store the FULL state in the frame for the viewer
                # Note: We store as a dict with Tuple keys (int, int) -> int
                # This is more efficient than the old string keys "x,y"
                new_frame['occupancy_grid'] = accumulated_grid.copy()
                
                # Remove delta to avoid confusion
                del new_frame['occupancy_grid_delta']
                
            elif 'occupancy_grid' in rf:
                # LEGACY FALLBACK: Convert old string keys to tuple keys if needed
                # The old format was {'10,20': 1}, we prefer {(10,20): 1}
                raw_grid = rf['occupancy_grid']
                if raw_grid and isinstance(next(iter(raw_grid)), str):
                    converted_grid = {}
                    for k, v in raw_grid.items():
                        gx, gy = map(int, k.split(','))
                        converted_grid[(gx, gy)] = v
                    new_frame['occupancy_grid'] = converted_grid
                # If it's already in the right format (or empty), just pass it
            
            reconstructed_frames.append(new_frame)
            
        return {
            'metadata': metadata,
            'frames': reconstructed_frames,
            'num_frames': num_frames,
        }