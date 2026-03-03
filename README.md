# PyBullet Multi-Robot Coverage Mapping

![](/simulation.png)

This project implements a decentralized multi-robot coverage mapping simulation using the PyBullet physics engine. It progresses from basic autonomous navigation to a complex, market-based coordination system where robots bid on frontier tasks to efficiently explore and map unknown procedural environments.

## Project Structure

The project is organized into modular components, separating hardware drivers, autonomy, and simulation logic:

| Directory | Description |
|-----------|-------------|
| **`robot/`** | Contains the core robot logic, including the `RobotAgent` (autonomy node), `RobotState`, and the `PyBulletDriver` hardware abstraction layer. |
| **`behaviors/`** | Implements specific robot behaviors such as `PathFollower` (Pure Pursuit), `StuckDetector`, and `ExplorationDirectionTracker`. |
| **`coordination/`** | Handles multi-robot task allocation using an auction-based system (`TaskAllocator`) and utility calculations (`FrontierUtilityCalculator`). |
| **`navigation/`** | Geometry utilities and A* pathfinding for robot motion planning. |
| **`mapping/`** | Manages the global occupancy grid and frontier detection. Includes `numba_accelerator.py` for optimized ray-casting and grid updates. |
| **`simulation/`** | Handles procedural level generation (`MapGenerator`), the physics engine interface, and the overall simulation manager. |
| **`visualization/`** | Tools for real-time monitoring (`RealtimeVisualizer`), data logging (`SimulationLogger`), and offline playback (`playback.py`). |
| **`utils/`** | Configuration loading and schema validation. |

## Key Features

* **Procedural Generation**: Capable of generating 7 distinct environment types (`maze`, `cave`, `rooms`, `sewer`, `tunnel`, `corridor_rooms`, `blank_box`) to test robustness.
* **Market-Based Coordination**: Robots bid on frontiers based on a utility function that balances distance, frontier size, and their current heading.
* **Stuck Recovery**: An integrated stuck detector monitors robot motion and triggers recovery maneuvers if a robot gets trapped by obstacles.
* **Optimized Performance**: Uses `Numba` JIT compilation for high-performance LIDAR ray-casting and A* pathfinding.

## Prerequisites

* **Python**: Version 3.11 or higher.
* [**uv**](https://docs.astral.sh/uv/): Fast Python package manager (recommended).

## Installation and Run Instructions

Install dependencies and run using `uv`:

```bash
uv sync
```

### 1. Run Simulation

To start the simulation with the default configuration:

```bash
uv run python run.py
```

### 2. Replay Logged Data

To replay a saved simulation log (using interactive controls):

```bash
uv run python visualization/playback.py logs/sim_log_TIMESTAMP.npz
```

### 3. Render Video

To convert a log file into a high-quality MP4 video using the parallel renderer:

```bash
uv run python visualization/renderer.py logs/sim_log_TIMESTAMP.npz
```

## Configuration

### Headless vs. Visual Mode

By default, the simulation may run in a hybrid mode or headless mode depending on your needs. You do not need to recompile code; simply edit the configuration file.

1. Open [config/default.yaml](config/default.yaml).
2. Locate the `system` section.
3. Adjust `use_gui` and `viz_mode`:

**For Visual Mode (PyBullet GUI + Realtime Plotting):**

```yaml
system:
  use_gui: true
  viz_mode: "realtime"  # or "both"
```

**For Headless / Fast Benchmarking:**

```yaml
system:
  use_gui: false
  viz_mode: "logging"
```

### Visualization Controls

When running in visual mode (`viz_mode: "realtime"`), the simulation opens a Matplotlib dashboard with the following controls:

**Mouse Controls:**

* **Scroll**: Zoom in/out of the map.
* **PyBullet Window**: Click and drag to rotate the 3D camera.

**Keyboard Controls:**

* **P** - Toggle Rectangular Decomposition visualization
  * When enabled, colored rectangles show how the free space is partitioned.

### Simulation Parameters

Key simulation parameters can be adjusted via [config/default.yaml](config/default.yaml):

* **Environment:**
  * `maze_size`: Dimensions of the procedural map (e.g., 10, 15, 20).
  * `type`: Map style (maze, cave, rooms, sewer).

* **Robots:**
  * `count`: Number of robots to spawn.
  * `lidar`: Configuration for sensor range and ray count.

* **Planning:**
  * `utility_weights`: Tune the `direction_bias`, `size`, and `distance` weights to change how robots select frontiers.

### Output Format

When running the simulation, the console displays a live dashboard of performance metrics:

```
SIMULATION STATUS
   Step:         542
   Coverage:     98.2%
   Speed:        120 steps/sec
   Mode:         EXPLORING
   Active:       3/3 robots

PERFORMANCE BREAKDOWN
   Sensing (LIDAR)      [████████░░░░░░░░░░░░░░]  35.2%
   Global Planning      [██░░░░░░░░░░░░░░░░░░░░]  12.5%
   ...
```

The simulation saves logs in `.npz` format using delta encoding to minimize file size. These logs contain:

* Reconstructable Occupancy Grids
* Robot Trajectories & States
* Frontier Locations
* Coverage Statistics