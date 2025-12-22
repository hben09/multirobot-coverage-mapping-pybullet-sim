import pybullet as p
import time
from collections import defaultdict
from sim_logger import SimulationLogger
from pathfinding import NumbaAStarHelper
from occupancy_grid_manager import OccupancyGridManager
from coordination_controller import CoordinationController
from simulation_initializer import SimulationInitializer
from sim_config import get_simulation_config, print_config
import sim_config as cfg
from realtime_visualizer import RealtimeVisualizer

class SimulationManager:
    """Manages multi-robot coverage mapping simulation for procedurally generated environments"""

    def __init__(self, use_gui=True, maze_size=(10, 10), cell_size=2.0, env_seed=None, env_type='maze', num_robots=3, show_partitions=False, grid_resolution=0.5):
        self.num_robots = num_robots
        self.show_partitions = show_partitions  # Flag for rectangular decomposition

        # Initialize environment and robots using the initializer
        initializer = SimulationInitializer(
            use_gui=use_gui,
            maze_size=maze_size,
            cell_size=cell_size,
            env_seed=env_seed,
            env_type=env_type,
            num_robots=num_robots,
            show_partitions=show_partitions
        )

        # Setup environment
        self.env, self.physics_client, self.map_bounds = initializer.initialize_environment()

        # Create robots
        self.robots = initializer.create_robots(self.env)

        # Initialize occupancy grid manager
        self.grid_manager = OccupancyGridManager(self.map_bounds, grid_resolution, self.env)

        # Initialize coordination controller
        self.coordinator = CoordinationController(
            direction_bias_weight=cfg.DIRECTION_BIAS_WEIGHT,
            size_weight=cfg.SIZE_WEIGHT,
            distance_weight=cfg.DISTANCE_WEIGHT,
            crowding_penalty_weight=cfg.CROWDING_PENALTY_WEIGHT,
            crowding_radius=cfg.CROWDING_RADIUS
        )

        self.coverage_history = []
        self.claimed_goals = {}

        # Real-time visualization
        self.visualizer = RealtimeVisualizer(self)

        # Return-to-home settings (from config)
        self.return_home_coverage = cfg.RETURN_HOME_COVERAGE
        self.returning_home = False
        self.robots_home = set()

        # Safety margin (from config)
        self.safety_margin = cfg.SAFETY_MARGIN

        # Numba A* helper
        self._numba_astar = NumbaAStarHelper()
        
        # Logging
        self.logger = None
        self.env_config = initializer.get_env_config()

    def world_to_grid(self, x, y):
        return self.grid_manager.world_to_grid(x, y)

    def grid_to_world(self, grid_x, grid_y):
        return self.grid_manager.grid_to_world(grid_x, grid_y)

    def update_occupancy_grid(self, robot):
        """Update occupancy grid from robot LIDAR data - NUMBA OPTIMIZED."""
        if not robot.state.lidar_data:
            return

        pos, _ = p.getBasePositionAndOrientation(robot.state.id)
        self.grid_manager.update_occupancy_grid((pos[0], pos[1]), robot.state.lidar_data)

    def detect_frontiers(self, use_cache=False):
        """Detect and cluster frontier cells - OPTIMIZED with candidate tracking."""
        return self.grid_manager.detect_frontiers(use_cache)

    def plan_path_astar(self, start_grid, goal_grid):
        """
        A* Pathfinding on the occupancy grid.
        Uses Numba JIT compilation for 6-18x speedup.
        """
        self._numba_astar.update_grid(
            self.grid_manager.occupancy_grid,
            self.grid_manager.obstacle_cells,
            self.safety_margin
        )
        path = self._numba_astar.plan_path(start_grid, goal_grid, use_inflation=True)
        return path
    
    def calculate_utility(self, robot, frontier):
        """Utility Function with Direction Bias and Volumetric Gain."""
        return self.coordinator.calculate_utility(robot, frontier)

    def assign_global_goals(self, step):
        """Market-based Coordination loop (Improved: Global Best-First)."""
        frontiers = self.detect_frontiers()
        self.coordinator.assign_global_goals(
            self.robots,
            frontiers,
            step,
            self.world_to_grid,
            self.plan_path_astar
        )

    def exploration_logic(self, robot, step):
        """Execute exploration logic for a robot."""
        action = self.coordinator.execute_exploration_logic(
            robot,
            step,
            self.plan_return_path_for_robot
        )

        if action == 'FOLLOW_PATH':
            l, a = robot.path_follower.compute_velocities(
                robot.state,
                robot.hardware,
                self.grid_to_world
            )
            robot.hardware.set_velocity(l, a)

    def plan_return_path_for_robot(self, robot):
        """Plan a path home for a single robot using A*."""
        self.coordinator.plan_return_path(robot, self.world_to_grid, self.plan_path_astar)

    def trigger_return_home(self):
        """Trigger all robots to return to their home positions using A*."""
        self.coordinator.trigger_return_home(self.robots, self.world_to_grid, self.plan_path_astar)

    def calculate_coverage(self, use_cache=True):
        """Calculate coverage percentage - INCREMENTAL VERSION."""
        return self.grid_manager.calculate_coverage(use_cache)

    def invalidate_coverage_cache(self):
        self.grid_manager.invalidate_coverage_cache()


    def run_simulation(self, steps=5000, scan_interval=10, use_gui=True, realtime_viz=True,
                       viz_update_interval=50, viz_mode='realtime', log_path='./logs', performance_report_interval=3.0):
        """
        Run the simulation with configurable visualization and logging.
        """
        print("Starting multi-robot coverage mapping simulation...")
        
        if steps is None:
            print("Running unlimited steps (press Ctrl+C to stop)...")
        else:
            print(f"Running for {steps} steps...")

        do_realtime = viz_mode in ['realtime', 'both']
        do_logging = viz_mode in ['logging', 'both']
        
        fast_mode = viz_mode == 'logging' and not use_gui
        if fast_mode:
            print("*** FAST MODE ENABLED - Maximum simulation speed ***")
        
        print(f"Visualization mode: {viz_mode}")
        if do_realtime:
            print("  - Real-time visualization: ENABLED")
        if do_logging:
            print(f"  - Logging: ENABLED (saving to {log_path})")

        if do_realtime:
            print("Setting up real-time visualization...")
            self.visualizer.setup()
            
        if do_logging:
            self.logger = SimulationLogger(log_dir=log_path)
            self.logger.initialize(self, self.env_config)

        step = 0
        
        robots = self.robots
        num_robots = len(robots)
        returning_home = self.returning_home
        robots_home = self.robots_home
        return_home_coverage = self.return_home_coverage
        
        start_time = time.perf_counter()
        last_report_time = start_time
        last_report_step = 0
        report_interval_seconds = performance_report_interval
        
        perf_stats = defaultdict(float)
        
        while True:
            if steps is not None and step >= steps:
                break

            if returning_home and len(robots_home) == num_robots:
                print("\n*** ALL ROBOTS RETURNED HOME ***")
                break

            t0 = time.perf_counter()

            active_robots = robots
            any_idle = any(r.state.goal is None for r in active_robots)
            
            should_plan = (step % 50 == 0) or (any_idle and step % 5 == 0)
            
            if should_plan:
                coverage = self.calculate_coverage(use_cache=False)
                
                if not returning_home and coverage >= return_home_coverage:
                    print(f"\n*** COVERAGE {coverage:.1f}% >= {return_home_coverage}% - RETURNING HOME ***")
                    returning_home = True
                    self.returning_home = True
                    self.trigger_return_home()
                elif not returning_home:
                    self.assign_global_goals(step)
            
            perf_stats['global_planning'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            for robot in robots:
                self.exploration_logic(robot, step)
                
                if returning_home and robot.state.mode == 'RETURNING_HOME':
                    pos, _ = p.getBasePositionAndOrientation(robot.state.id)
                    dx = pos[0] - robot.state.home_position[0]
                    dy = pos[1] - robot.state.home_position[1]
                    if dx*dx + dy*dy < 1.0:
                        robot.state.mode = 'HOME'
                        robot.state.goal = None
                        robot.state.path = []
                        robots_home.add(robot.state.id)
                        print(f"Robot {robot.state.id} arrived home!")
            perf_stats['local_planning'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            if step % scan_interval == 0:
                for robot in robots:
                    # Perform LIDAR scan and update state
                    scan_points = robot.hardware.get_lidar_scan(num_rays=cfg.LIDAR_NUM_RAYS, max_range=cfg.LIDAR_MAX_RANGE)
                    robot.state.lidar_data = scan_points
                    pos_array, _ = robot.hardware.get_pose()
                    robot.state.trajectory.append((pos_array[0], pos_array[1]))
                    robot.state.trim_trajectory_if_needed()
                    robot.direction_tracker.update(robot.state)

                    self.update_occupancy_grid(robot)

                self.grid_manager.invalidate_coverage_cache()
                self.grid_manager.invalidate_frontier_cache()
                
                coverage = self.calculate_coverage()
                self.coverage_history.append((step, coverage))
                
            perf_stats['sensing'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            if step % viz_update_interval == 0:
                if do_realtime:
                    self.visualizer.update(step)
                if do_logging:
                    self.logger.log_frame(step, self)
            perf_stats['visualization'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            p.stepSimulation()
            perf_stats['physics'] += time.perf_counter() - t0

            if use_gui:
                time.sleep(1./240.)

            current_time = time.perf_counter()
            elapsed_since_report = current_time - last_report_time
            
            if elapsed_since_report >= report_interval_seconds:
                steps_done = step - last_report_step
                sps = steps_done / elapsed_since_report if elapsed_since_report > 0 else 0
                coverage = self.calculate_coverage()
                status = "RETURNING HOME" if returning_home else "EXPLORING"
                
                print(f"Step {step} | Coverage: {coverage:.1f}% | {sps:.0f} steps/sec | Status: {status}")
                
                total_time = sum(perf_stats.values())
                if total_time > 0:
                    print("  [Performance Breakdown]:")
                    print(f"   - Sensing (LIDAR):   {100*perf_stats['sensing']/total_time:.1f}%")
                    print(f"   - Global Planning:   {100*perf_stats['global_planning']/total_time:.1f}%")
                    print(f"   - Local Planning:    {100*perf_stats['local_planning']/total_time:.1f}%")
                    print(f"   - Visualization:     {100*perf_stats['visualization']/total_time:.1f}%")
                    print(f"   - Physics Engine:    {100*perf_stats['physics']/total_time:.1f}%")
                
                perf_stats.clear()
                
                last_report_time = current_time
                last_report_step = step

            step += 1

        total_time = time.perf_counter() - start_time
        print(f"\n*** SIMULATION COMPLETE ***")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Total steps: {step}")
        print(f"  Average speed: {step/total_time:.0f} steps/second")

        if do_realtime:
            self.visualizer.update(step)
            print("\nReal-time visualization complete.")
            
        if do_logging:
            self.logger.log_frame(step, self)
            log_filepath = self.logger.save()

            print(f"\nTo replay this simulation interactively, run:")
            print(f"  python playback.py {log_filepath}")

            return log_filepath

        print("\nSimulation complete!")
        final_coverage = self.calculate_coverage()
        print(f"Final Coverage: {final_coverage:.2f}%")
        print(f"Explored Free Cells: {int(final_coverage/100 * self.grid_manager.total_free_cells)}/{int(self.grid_manager.total_free_cells)}")

        return None

    def cleanup(self):
        self.env.close()


def main():
    # Get user configuration (environment, visualization settings)
    user_config = get_simulation_config()

    # Print configuration summary
    print_config(user_config)

    # Create mapper with configuration
    mapper = SimulationManager(
        use_gui=user_config['use_gui'],
        maze_size=(user_config['maze_size'], user_config['maze_size']),
        cell_size=user_config['cell_size'],
        env_seed=user_config['env_seed'],
        env_type=user_config['env_type'],
        num_robots=user_config['num_robots'],
        show_partitions=user_config['show_partitions'],
        grid_resolution=user_config['grid_resolution']
    )

    log_filepath = None

    try:
        log_filepath = mapper.run_simulation(
            steps=user_config['max_steps'],
            scan_interval=user_config['scan_interval'],
            use_gui=user_config['use_gui'],
            viz_mode=user_config['viz_mode'],
            viz_update_interval=user_config['viz_update_interval'],
            log_path='./logs',
            performance_report_interval=user_config['performance_report_interval']
        )

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        if mapper.logger is not None and len(mapper.logger.frames) > 0:
            print("Saving partial log...")
            log_filepath = mapper.logger.save()
            print(f"\nTo replay this simulation interactively, run:")
            print(f"  python playback.py {log_filepath}")
    finally:
        mapper.visualizer.close()
        mapper.cleanup()
        print("PyBullet disconnected")

    # Handle video rendering
    if log_filepath is not None and user_config['render_video']:
        try:
            from video_renderer import render_video_from_log
            print("\nRendering video with OpenCV (fast parallel renderer)...")
            render_video_from_log(log_filepath)
        except ImportError:
            print("Warning: video_renderer.py not found, skipping video rendering")
        except Exception as e:
            print(f"Error rendering video: {e}")


if __name__ == "__main__":
    main()