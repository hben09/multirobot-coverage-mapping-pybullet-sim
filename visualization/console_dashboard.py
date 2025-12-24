"""
Console dashboard for displaying simulation status in the terminal.

This module provides a clean, in-place updating status display for the
multi-robot coverage simulation. It separates UI concerns from simulation logic.
"""


class ConsoleDashboard:
    """
    Manages terminal-based status display for the simulation.

    Provides a clean, in-place updating dashboard that shows simulation progress,
    robot status, and performance metrics without cluttering simulation logic.
    """

    def __init__(self, robots):
        """
        Initialize the console dashboard.

        Args:
            robots: List of robot containers to monitor
        """
        self.robots = robots
        self._display_initialized = False

    def clear(self):
        """Clear terminal screen using ANSI escape codes."""
        # Move cursor to home position and clear screen
        print('\033[H\033[2J', end='')

    def print_header(self):
        """Print simulation header banner."""
        print("=" * 70)
        print(" " * 15 + "MULTI-ROBOT COVERAGE SIMULATION")
        print("=" * 70)

    def print_status(self, step, coverage, sps, status, perf_stats, robots_home_count=0):
        """
        Print a clean, in-place updating status dashboard.

        Args:
            step: Current simulation step
            coverage: Coverage percentage
            sps: Steps per second (simulation speed)
            status: Current simulation status (e.g., "EXPLORING", "RETURNING HOME")
            perf_stats: Dictionary of performance statistics
            robots_home_count: Number of robots that have returned home
        """
        if self._display_initialized:
            self.clear()
        else:
            self._display_initialized = True

        self.print_header()

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

    def print_completion_summary(self, completion_reason, max_steps, final_coverage,
                                 total_free_cells, step, total_time):
        """
        Print final simulation completion summary.

        Args:
            completion_reason: Why the simulation ended ("all_home" or "max_steps")
            max_steps: Maximum steps configured
            final_coverage: Final coverage percentage achieved
            total_free_cells: Total number of free cells in the map
            step: Final step count
            total_time: Total simulation time in seconds
        """
        self.clear()

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
        print(f"   Free cells:   {int(final_coverage/100 * total_free_cells):,}/{int(total_free_cells):,}")
        print(f"   Total steps:  {step:,}")
        print(f"   Total time:   {total_time:.2f} seconds")
        print(f"   Avg speed:    {step/total_time:.0f} steps/second")
        print()

    def print_replay_instructions(self, log_filepath):
        """
        Print instructions for replaying the simulation.

        Args:
            log_filepath: Path to the saved log file
        """
        print(f"REPLAY")
        print(f"   To replay this simulation, run:")
        print(f"   python playback.py {log_filepath}")
        print("\n" + "=" * 70)

    def print_final_separator(self):
        """Print final separator line."""
        print("=" * 70)
