import pybullet as p
import time
import os
from collections import defaultdict

from visualization.logger import SimulationLogger
from navigation.pathfinding import NumbaAStarHelper
from mapping.grid_manager import OccupancyGridManager
from coordination.controller import CoordinationController
from simulation.initializer import SimulationInitializer
from visualization.realtime import RealtimeVisualizer
from utils.config_schema import SimulationConfig


class SimulationManager:
    """Manages multi-robot coverage mapping simulation."""

    def __init__(self, config: SimulationConfig):
        """
        Initialize the simulation manager with a typed configuration object.

        Args:
            config: Type-safe configuration object with all simulation parameters.
        """
        self.config = config
        self.env_cfg = config.environment
        self.robot_cfg = config.robots
        self.sys_cfg = config.system
        self.plan_cfg = config.planning

        self.show_partitions = self.sys_cfg.show_partitions

        # Initialize environment and robots using the initializer
        initializer = SimulationInitializer(self.env_cfg)

        # Setup environment
        self.env, self.physics_client, self.map_bounds = initializer.initialize_environment(
            use_gui=self.sys_cfg.use_gui
        )

        # Create robots
        self.robots = initializer.create_robots(
            self.env,
            self.robot_cfg,
            self.config.physics,
            lidar_config=self.config.sensors.lidar
        )

        # Safety margin (needed before grid initialization)
        self.safety_margin = self.env_cfg.safety_margin

        # Numba A* helper (needed before grid initialization)
        self._numba_astar = NumbaAStarHelper()

        # Initialize occupancy grid manager
        self.grid_manager = OccupancyGridManager(
            self.map_bounds,
            self.plan_cfg.grid_resolution,
            self.env
        )

        # Initialize coordination controller
        coord_cfg = self.plan_cfg.coordination
        util_weights = self.plan_cfg.utility_weights

        self.coordinator = CoordinationController(
            direction_bias_weight=util_weights.direction_bias,
            size_weight=util_weights.size,
            distance_weight=util_weights.distance,
            crowding_penalty_weight=coord_cfg.crowding_penalty,
            crowding_radius=coord_cfg.crowding_radius
        )

        # Inject planning dependencies into robot agents
        # (Simulates robots subscribing to "map server" in ROS)
        for robot in self.robots:
            robot.agent.set_planning_dependencies(
                grid_manager=self.grid_manager,
                planner=self._numba_astar,
                safety_margin=self.safety_margin,
                utility_calculator=self.coordinator.utility_calculator
            )

        self.coverage_history = []

        # Real-time visualization
        self.visualizer = RealtimeVisualizer(self)

        # Return-to-home state (triggered when no frontiers remain)
        self.returning_home = False
        self.robots_home = set()

        # Logging
        self.logger = None
        self.env_config = initializer.get_env_config()

        # Terminal display state
        self._display_initialized = False

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

    def assign_global_goals(self, step):
        """
        Auction-based Coordination loop.

        Uses the "Auctioneer" pattern where robots calculate their own bids
        based on their internal state.
        """
        frontiers = self.detect_frontiers()
        self.coordinator.assign_global_goals_auction(
            self.robots,
            frontiers,
            step
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
            grid_to_world_fn=self.grid_to_world
        )

    def trigger_return_home(self):
        """Trigger all robots to return to their home positions."""
        self.coordinator.trigger_return_home(self.robots)

    def calculate_coverage(self, use_cache=True):
        """Calculate coverage percentage."""
        return self.grid_manager.calculate_coverage(use_cache)

    def invalidate_coverage_cache(self):
        self.grid_manager.invalidate_coverage_cache()

    def _clear_terminal(self):
        """Clear terminal screen using ANSI escape codes."""
        # Move cursor to home position and clear screen
        print('\033[H\033[2J', end='')

    def _print_header(self):
        """Print simulation header banner."""
        print("=" * 70)
        print(" " * 15 + "MULTI-ROBOT COVERAGE SIMULATION")
        print("=" * 70)

    def _print_status_dashboard(self, step, coverage, sps, status, perf_stats, robots_home_count=0):
        """Print a clean, in-place updating status dashboard."""
        if self._display_initialized:
            self._clear_terminal()
        else:
            self._display_initialized = True

        self._print_header()

        # Main status
        print(f"\nSIMULATION STATUS")
        print(f"   Step:         {step:,}")
        print(f"   Coverage:     {coverage:.1f}%")
        print(f"   Speed:        {sps:.0f} steps/sec")
        print(f"   Mode:         {status}")

        if robots_home_count > 0:
            print(f"   Robots Home:  {robots_home_count}/{len(self.robots)}")

        # Robot assignment summary
        robots_with_goals = sum(1 for r in self.robots if r.state.goal is not None)
        robots_idle = len(self.robots) - robots_with_goals
        robots_stuck = sum(1 for r in self.robots if r.state.goal_attempts > 0)
        print(f"   Active:       {robots_with_goals}/{len(self.robots)} robots")
        if robots_idle > 0:
            print(f"   Idle:         {robots_idle} robots")
        if robots_stuck > 0:
            print(f"   Recovering:   {robots_stuck} robots (stuck/replanning)")

        # Performance breakdown
        total_time = sum(perf_stats.values())
        if total_time > 0:
            print(f"\nPERFORMANCE BREAKDOWN")
            components = [
                ("Sensing (LIDAR)", perf_stats['sensing']),
                ("Global Planning", perf_stats['global_planning']),
                ("Local Planning", perf_stats['local_planning']),
                ("Visualization", perf_stats['visualization']),
                ("Physics Engine", perf_stats['physics'])
            ]

            for name, time_spent in components:
                pct = 100 * time_spent / total_time
                bar_length = 30
                filled = int(bar_length * pct / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"   {name:20s} [{bar}] {pct:5.1f}%")

        print("\n" + "=" * 70)
        print()  # Extra line for breathing room


    def run_simulation(self, log_path='./logs'):
        """
        Run the simulation with configurable visualization and logging.
        """
        print("Starting multi-robot coverage mapping simulation...")

        # Pull parameters from config
        max_steps = self.sys_cfg.max_steps
        intervals = self.sys_cfg.intervals
        viz_mode = self.sys_cfg.viz_mode

        if max_steps is None:
            print("Running unlimited steps (press Ctrl+C to stop)...")
        else:
            print(f"Running for {max_steps} steps...")

        do_realtime = viz_mode in ['realtime', 'both']
        do_logging = viz_mode in ['logging', 'both']
        use_gui = self.sys_cfg.use_gui

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

        start_time = time.perf_counter()
        last_report_time = start_time
        last_report_step = 0

        perf_stats = defaultdict(float)

        # Track completion reason
        completion_reason = "max_steps"

        while True:
            if max_steps is not None and step >= max_steps:
                completion_reason = "max_steps"
                break

            if returning_home and len(robots_home) == num_robots:
                completion_reason = "all_home"
                break

            t0 = time.perf_counter()

            active_robots = robots
            any_idle = any(r.state.goal is None for r in active_robots)

            should_plan = (step % 50 == 0) or (any_idle and step % 5 == 0)

            if should_plan:
                coverage = self.calculate_coverage(use_cache=False)

                # Check if there are any accessible frontiers
                if not returning_home:
                    frontiers = self.detect_frontiers(use_cache=False)

                    # Check if any frontiers are accessible (not blacklisted by all robots)
                    has_accessible_frontiers = False
                    if frontiers:
                        for frontier in frontiers:
                            # Check if at least one robot can access this frontier
                            for robot in robots:
                                if frontier['grid_pos'] not in robot.state.blacklisted_goals:
                                    has_accessible_frontiers = True
                                    break
                            if has_accessible_frontiers:
                                break

                    # Trigger return home when no accessible frontiers remain
                    if coverage > 1.0 and not has_accessible_frontiers:
                        returning_home = True
                        self.returning_home = True
                        self.trigger_return_home()
                    elif has_accessible_frontiers:
                        # Use auction pattern for task allocation
                        self.assign_global_goals(step)

            perf_stats['global_planning'] += time.perf_counter() - t0

            # Agent update cycle with optional sensing
            t0 = time.perf_counter()
            should_sense = (step % intervals.scan == 0)

            for robot in robots:
                # Agent handles sense-think-act autonomously
                self.exploration_logic(robot, should_sense=should_sense)

                # Check if robot reached home
                if returning_home and robot.state.mode == 'RETURNING_HOME':
                    if robot.agent.check_if_home():
                        robots_home.add(robot.state.id)

            perf_stats['local_planning'] += time.perf_counter() - t0

            # Update occupancy grid from robot sensor data
            t0 = time.perf_counter()
            if should_sense:
                for robot in robots:
                    self.update_occupancy_grid(robot)

                self.grid_manager.invalidate_coverage_cache()

                coverage = self.calculate_coverage()
                self.coverage_history.append((step, coverage))

            perf_stats['sensing'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            if step % intervals.viz_update == 0:
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

            if elapsed_since_report >= intervals.performance_report:
                steps_done = step - last_report_step
                sps = steps_done / elapsed_since_report if elapsed_since_report > 0 else 0
                coverage = self.calculate_coverage()
                status = "RETURNING HOME" if returning_home else "EXPLORING"

                # Use clean dashboard display
                self._print_status_dashboard(
                    step, coverage, sps, status, perf_stats,
                    robots_home_count=len(robots_home)
                )

                perf_stats.clear()

                last_report_time = current_time
                last_report_step = step

            step += 1

        # Clear screen and print final summary
        self._clear_terminal()

        total_time = time.perf_counter() - start_time
        final_coverage = self.calculate_coverage()

        print("=" * 70)
        print(" " * 20 + "SIMULATION COMPLETE")
        print("=" * 70)

        # Completion reason
        if completion_reason == "all_home":
            print("\nAll robots returned home successfully")
        elif completion_reason == "max_steps":
            print(f"\nReached maximum steps limit ({max_steps:,})")

        print(f"\nFINAL STATISTICS")
        print(f"   Coverage:     {final_coverage:.1f}%")
        print(f"   Free cells:   {int(final_coverage/100 * self.grid_manager.total_free_cells):,}/{int(self.grid_manager.total_free_cells):,}")
        print(f"   Total steps:  {step:,}")
        print(f"   Total time:   {total_time:.2f} seconds")
        print(f"   Avg speed:    {step/total_time:.0f} steps/second")
        print()

        if do_realtime:
            self.visualizer.update(step)

        if do_logging:
            self.logger.log_frame(step, self)
            log_filepath = self.logger.save()

            print(f"REPLAY")
            print(f"   To replay this simulation, run:")
            print(f"   python playback.py {log_filepath}")
            print("\n" + "=" * 70)

            return log_filepath

        print("=" * 70)
        return None

    def cleanup(self):
        self.env.close()
