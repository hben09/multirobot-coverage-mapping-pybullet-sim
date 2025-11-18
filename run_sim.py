import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict

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
        
        # Mapping data
        self.occupancy_grid = defaultdict(int)
        
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
    
    def run_simulation(self, steps=2000, scan_interval=10, use_gui=True):
        """Run the simulation"""
        print("Starting multi-robot mapping simulation...")
        print("Red, Green, and Blue robots are exploring the environment")
        print(f"Running for {steps} steps...")

        for step in range(steps):
            # Move each robot
            for robot in self.robots:
                self.simple_exploration_move(robot, step)

            # Perform lidar scans periodically
            if step % scan_interval == 0:
                for robot in self.robots:
                    robot.get_lidar_scan(num_rays=180, max_range=15)

            # Step simulation
            p.stepSimulation()

            # Add delay for GUI visualization
            if use_gui:
                time.sleep(1./240.)  # 240 Hz simulation rate

            # Print progress
            if step % 200 == 0:
                print(f"Progress: {step}/{steps} steps completed")
        
        print("Simulation complete!")
        print(f"Robot 1 (Red) scanned {len(self.robots[0].lidar_data)} points")
        print(f"Robot 2 (Green) scanned {len(self.robots[1].lidar_data)} points")
        print(f"Robot 3 (Blue) scanned {len(self.robots[2].lidar_data)} points")
    
    def visualize_map(self):
        """Visualize the mapped environment"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        
        colors = ['red', 'green', 'blue']
        robot_names = ['Robot 1 (Red)', 'Robot 2 (Green)', 'Robot 3 (Blue)']
        
        # Individual robot maps
        for idx, (ax, robot, color, name) in enumerate(zip(axes.flat[:3], 
                                                           self.robots, 
                                                           colors, 
                                                           robot_names)):
            if robot.lidar_data:
                points = np.array(robot.lidar_data)
                ax.scatter(points[:, 0], points[:, 1], c=color, s=1, alpha=0.3)
                
                # Plot trajectory
                if robot.trajectory:
                    traj = np.array(robot.trajectory)
                    ax.plot(traj[:, 0], traj[:, 1], c=color, linewidth=2, 
                           label='Trajectory')
            
            ax.set_xlim(-12, 12)
            ax.set_ylim(-12, 12)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{name} Map')
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.legend()
        
        # Combined map
        ax_combined = axes.flat[3]
        for robot, color, name in zip(self.robots, colors, robot_names):
            if robot.lidar_data:
                points = np.array(robot.lidar_data)
                ax_combined.scatter(points[:, 0], points[:, 1], c=color, s=1, 
                                  alpha=0.3, label=name)
                
                # Plot trajectories
                if robot.trajectory:
                    traj = np.array(robot.trajectory)
                    ax_combined.plot(traj[:, 0], traj[:, 1], c=color, 
                                   linewidth=2, alpha=0.7)
        
        ax_combined.set_xlim(-12, 12)
        ax_combined.set_ylim(-12, 12)
        ax_combined.set_aspect('equal')
        ax_combined.grid(True, alpha=0.3)
        ax_combined.set_title('Combined Multi-Robot Map')
        ax_combined.set_xlabel('X (meters)')
        ax_combined.set_ylabel('Y (meters)')
        ax_combined.legend()
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/multi_robot_map.png', dpi=150, 
                   bbox_inches='tight')
        print("Map visualization saved!")
        plt.close()
    
    def cleanup(self):
        """Disconnect from PyBullet"""
        p.disconnect()

def main():
    # Create mapper with GUI enabled
    mapper = MultiRobotMapper(use_gui=True)

    # Run simulation
    try:
        mapper.run_simulation(steps=2000, scan_interval=10, use_gui=True)
        
        # Generate map visualization
        print("\nGenerating map visualization...")
        
        # Visualize results
        mapper.visualize_map()
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        mapper.cleanup()
        print("PyBullet disconnected")

if __name__ == "__main__":
    main()