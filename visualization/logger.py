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
        # Cache for delta encoding (now uses Numpy array)
        self.last_occupancy_grid = None 

    def initialize(self, mapper, env_config):
        """Initialize logger with simulation metadata."""
        self.start_time = datetime.now()
        self.last_occupancy_grid = None

        # Store metadata about the simulation
        self.metadata = {
            'timestamp': self.start_time.isoformat(),
            'env_config': env_config,
            'map_bounds': mapper.map_bounds.copy(),
            'grid_resolution': mapper.grid_manager.grid_resolution,
            'num_robots': len(mapper.robots),
            'robot_colors': [robot.state.color for robot in mapper.robots],
            'robot_home_positions': [robot.state.home_position.tolist() for robot in mapper.robots],
            'total_free_cells': mapper.grid_manager.total_free_cells,
            'maze_grid': mapper.env.maze_grid.copy(),
        }

        self.frames = []
        print(f"Logger initialized. Will save to: {self.log_dir}")

    def log_frame(self, step, mapper):
        """
        Log a single frame of simulation state using Delta Encoding with Numpy arrays.

        Instead of saving the whole grid every frame, we only save the cells
        that have changed since the last frame.
        """
        import pybullet as p

        # 1. Calculate Occupancy Grid Delta using Numpy (FAST PATH)
        current_grid = mapper.grid_manager.get_numpy_grid()
        offset_x, offset_y = mapper.grid_manager.get_grid_offset()

        # Initialize cache on first frame
        if self.last_occupancy_grid is None:
            self.last_occupancy_grid = np.zeros_like(current_grid)

        # Find changed cells using vectorized comparison
        changed_mask = (current_grid != self.last_occupancy_grid)
        changed_coords = np.argwhere(changed_mask)

        # Build delta list with absolute coordinates
        delta = []
        for coord in changed_coords:
            y, x = coord  # row, col
            abs_x = x + offset_x
            abs_y = y + offset_y
            val = int(current_grid[y, x])
            delta.append([abs_x, abs_y, val])

        # Update cache
        self.last_occupancy_grid = current_grid.copy()

        # Extract explored and obstacle cells from Numpy grid
        free_coords = np.argwhere(current_grid == 1)
        explored_cells = [[int(c[1] + offset_x), int(c[0] + offset_y)] for c in free_coords]

        obstacle_coords = np.argwhere(current_grid == 2)
        obstacle_cells = [[int(c[1] + offset_x), int(c[0] + offset_y)] for c in obstacle_coords]

        frame = {
            'step': step,
            'coverage': mapper.calculate_coverage(),
            'returning_home': mapper.returning_home,

            # DELTA ENCODING: Store only the changes
            'occupancy_grid_delta': delta,

            # These lists are now extracted from Numpy grid
            'explored_cells': explored_cells,
            'obstacle_cells': obstacle_cells,

            # Frontier data
            'frontiers': mapper.detect_frontiers(),

            # Robot states
            'robots': []
        }

        for robot in mapper.robots:
            pos, orn = p.getBasePositionAndOrientation(robot.state.id)
            euler = p.getEulerFromQuaternion(orn)

            robot_state = {
                'position': [pos[0], pos[1], pos[2]],
                'orientation': euler[2],  # yaw
                'goal': list(robot.state.goal) if robot.state.goal else None,
                'path': [list(wp) for wp in robot.state.path] if robot.state.path else [],
                'mode': robot.state.mode,
                'exploration_direction': robot.state.exploration_direction.tolist(),
                'trajectory': [list(t) for t in robot.state.trajectory[-100:]],  # Last 100 points
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
                # Already in correct format (tuple keys)
                pass
            
            reconstructed_frames.append(new_frame)
            
        return {
            'metadata': metadata,
            'frames': reconstructed_frames,
            'num_frames': num_frames,
        }