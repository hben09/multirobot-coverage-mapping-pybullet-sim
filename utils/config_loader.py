import yaml
import os
from utils.config_schema import (
    SimulationConfig,
    EnvironmentConfig,
    RobotConfig,
    SensorConfig,
    LidarConfig,
    PhysicsConfig,
    PlanningConfig,
    UtilityWeightsConfig,
    CoordinationConfig,
    SystemConfig,
    IntervalConfig
)


def load_config(config_path="config/default.yaml") -> SimulationConfig:
    """
    Load configuration from a YAML file and convert to typed dataclass objects.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        SimulationConfig: Type-safe configuration object.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # Parse nested configuration structures
    try:
        config = SimulationConfig(
            environment=EnvironmentConfig(**raw_config['environment']),
            robots=RobotConfig(**raw_config['robots']),
            sensors=SensorConfig(
                lidar=LidarConfig(**raw_config['sensors']['lidar'])
            ),
            physics=PhysicsConfig(**raw_config['physics']),
            planning=PlanningConfig(
                grid_resolution=raw_config['planning']['grid_resolution'],
                utility_weights=UtilityWeightsConfig(**raw_config['planning']['utility_weights']),
                coordination=CoordinationConfig(**raw_config['planning']['coordination'])
            ),
            system=SystemConfig(
                use_gui=raw_config['system']['use_gui'],
                viz_mode=raw_config['system']['viz_mode'],
                render_video=raw_config['system']['render_video'],
                show_partitions=raw_config['system']['show_partitions'],
                max_steps=raw_config['system']['max_steps'],
                intervals=IntervalConfig(**raw_config['system']['intervals'])
            )
        )
        return config
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid configuration format: {e}")