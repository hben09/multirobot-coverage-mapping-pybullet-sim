import yaml
import os

def load_config(config_path="config/default.yaml"):
    """
    Load configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

# Optional: Helper to make dictionary access dot-notation friendly (cfg.robots.count)
class ConfigNamespace:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                value = ConfigNamespace(value)
            setattr(self, key, value)