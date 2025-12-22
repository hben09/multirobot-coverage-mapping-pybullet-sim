import pybullet as p
import time
from collections import defaultdict
import os

from sim_logger import SimulationLogger
from pathfinding import NumbaAStarHelper
from occupancy_grid_manager import OccupancyGridManager
from coordination_controller import CoordinationController
from simulation_initializer import SimulationInitializer
from realtime_visualizer import RealtimeVisualizer
from utils.config_loader import load_config

class SimulationManager:
    """Manages multi-robot coverage mapping simulation."""

    def __init__(self, config):
        """
        Initialize the simulation manager with a configuration dictionary.
        
        Args:
            config (dict): Complete configuration loaded from YAML.
        """
        self.config = config
        self.env_cfg = config['environment']
        self.robot_cfg = config['robots']
        self.sys_cfg = config['system']
        self.plan_cfg = config['planning']
        
        self.show_partitions = self.sys_cfg['show_partitions']

        # Initialize environment and robots using the initializer
        initializer = SimulationInitializer(self.env_cfg)

        # Setup environment
        self.env, self.physics_client, self.map_bounds = initializer.initialize_environment(
            use_gui=self.sys_cfg['use_gui']
        )

        # Create robots
        self.robots = initializer.create_robots(
            self.env,
            self.robot_cfg,
            self.config['physics'],
            lidar_config=self.config['sensors']['lidar']
        )

        # Initialize occupancy grid manager
        self.grid_manager = OccupancyGridManager(
            self.map_bounds, 
            self.plan_cfg['grid_resolution'], 
            self.env
        )

        # Initialize coordination controller
        coord_cfg = self.plan_cfg['coordination']
        util_weights = self.plan_cfg['utility_weights']
        
        self.coordinator = CoordinationController(
            direction_bias_weight=util_weights['direction_bias'],
            size_weight=util_weights['size'],
            distance_weight=util_weights['distance'],
            crowding_penalty_weight=coord_cfg['crowding_penalty'],
            crowding_radius=coord_cfg['crowding_radius']
        )

        self.coverage_history = []
        self.claimed_goals = {}

        # Real-time visualization
        self.visualizer = RealtimeVisualizer(self)

        # Return-to-home settings
        self.return_home_coverage = self.plan_cfg['return_home_coverage']
        self.returning_home = False
        self.robots_home = set()

        # Safety margin
        self.safety_margin = self.env_cfg['safety_margin']

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
        """Update occupancy grid from robot LIDAR data."""
        if not robot.state.lidar_data:
            return

        pos, _ = p.getBasePositionAndOrientation(robot.state.id)
        self.grid_manager.update_occupancy_grid((pos[0], pos[1]), robot.state.lidar_data)

    def detect_frontiers(self, use_cache=False):
        """Detect and cluster frontier cells."""
        return self.grid_manager.detect_frontiers(use_cache)

    def plan_path_astar(self, start_grid, goal_grid):
        """A* Pathfinding on the occupancy grid."""
        self._numba_astar.update_grid(
            self.grid_manager.occupancy_grid,
            self.grid_manager.obstacle_cells,
            self.safety_margin
        )
        path = self._numba_astar.plan_path(start_grid, goal_grid, use_inflation=True)
        return path
    
    def calculate_utility(self, robot, frontier):
        """Utility Function."""
        return self.coordinator.calculate_utility(robot, frontier)

    def assign_global_goals(self, step):
        """Market-based Coordination loop."""
        frontiers = self.detect_frontiers()
        self.coordinator.assign_global_goals(
            self.robots,
            frontiers,
            step,
            self.world_to_grid,
            self.plan_path_astar
        )

    def exploration_logic(self, robot, should_sense=True):
        """
        Execute exploration logic for a robot using agent-based control.

        Args:
            robot: RobotContainer with agent
            should_sense: Whether the robot should perform sensing this step
        """
        # Use the robot's autonomous agent to execute sense-think-act cycle
        robot.agent.update(
            should_sense=should_sense,
            plan_return_path_fn=self.plan_return_path_for_robot,
            grid_to_world_fn=self.grid_to_world
        )

    def plan_return_path_for_robot(self, robot):
        """Plan a path home for a single robot using A*."""
        self.coordinator.plan_return_path(robot, self.world_to_grid, self.plan_path_astar)

    def trigger_return_home(self):
        """Trigger all robots to return to their home positions using A*."""
        self.coordinator.trigger_return_home(self.robots, self.world_to_grid, self.plan_path_astar)

    def calculate_coverage(self, use_cache=True):
        """Calculate coverage percentage."""
        return self.grid_manager.calculate_coverage(use_cache)

    def invalidate_coverage_cache(self):
        self.grid_manager.invalidate_coverage_cache()


    def run_simulation(self, log_path='./logs'):
        """
        Run the simulation with configurable visualization and logging.
        """
        print("Starting multi-robot coverage mapping simulation...")
        
        # Pull parameters from config
        max_steps = self.sys_cfg['max_steps']
        intervals = self.sys_cfg['intervals']
        viz_mode = self.sys_cfg['viz_mode']
        
        if max_steps is None:
            print("Running unlimited steps (press Ctrl+C to stop)...")
        else:
            print(f"Running for {max_steps} steps...")

        do_realtime = viz_mode in ['realtime', 'both']
        do_logging = viz_mode in ['logging', 'both']
        use_gui = self.sys_cfg['use_gui']
        
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

        perf_stats = defaultdict(float)

        while True:
            if max_steps is not None and step >= max_steps:
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

            # Agent update cycle with optional sensing
            t0 = time.perf_counter()
            should_sense = (step % intervals['scan'] == 0)

            for robot in robots:
                # Agent handles sense-think-act autonomously
                self.exploration_logic(robot, should_sense=should_sense)

                # Check if robot reached home
                if returning_home and robot.state.mode == 'RETURNING_HOME':
                    if robot.agent.check_if_home():
                        robots_home.add(robot.state.id)
                        print(f"Robot {robot.state.id} arrived home!")

            perf_stats['local_planning'] += time.perf_counter() - t0

            # Update occupancy grid from robot sensor data
            t0 = time.perf_counter()
            if should_sense:
                for robot in robots:
                    self.update_occupancy_grid(robot)

                self.grid_manager.invalidate_coverage_cache()
                self.grid_manager.invalidate_frontier_cache()

                coverage = self.calculate_coverage()
                self.coverage_history.append((step, coverage))

            perf_stats['sensing'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            if step % intervals['viz_update'] == 0:
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
            
            if elapsed_since_report >= intervals['performance_report']:
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
    # 1. Load Configuration
    try:
        cfg = load_config("config/default.yaml")
        print(f"Loaded configuration from config/default.yaml")
    except Exception as e:
        print(f"Failed to load config: {e}")
        return

    # 2. Print summary
    env = cfg['environment']
    rob = cfg['robots']
    print(f"\nCreating {env['maze_size']}x{env['maze_size']} {env['type']} "
          f"with {rob['count']} robots...")

    # 3. Initialize Manager with full config
    mapper = SimulationManager(cfg)

    log_filepath = None

    try:
        log_filepath = mapper.run_simulation(log_path='./logs')

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        if mapper.logger is not None and len(mapper.logger.frames) > 0:
            print("Saving partial log...")
            log_filepath = mapper.logger.save()
            print(f"\nTo replay this simulation interactively, run:")
            print(f"  python playback.py {log_filepath}")
    finally:
        if hasattr(mapper, 'visualizer'):
            mapper.visualizer.close()
        mapper.cleanup()
        print("PyBullet disconnected")

    # Handle video rendering
    if log_filepath is not None and cfg['system']['render_video']:
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