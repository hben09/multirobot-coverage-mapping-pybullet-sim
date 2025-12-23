from simulation.simulation_manager import SimulationManager
from utils.config_loader import load_config


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
            from visualization.renderer import render_video_from_log
            print("\nRendering video with OpenCV (fast parallel renderer)...")
            render_video_from_log(log_filepath)
        except ImportError:
            print("Warning: visualization.renderer not found, skipping video rendering")
        except Exception as e:
            print(f"Error rendering video: {e}")


if __name__ == "__main__":
    main()
