import os
import yaml
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Load yaml file
def load_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

class Config:
    def __init__(self):
        # Load env variables
        self.env = os.getenv("ENV", "local") # Get ENV or default to local
        self.data_root = os.getenv("DATA_ROOT", "data") # Get environment dataset path

        # Load yaml files
        self.configs = {
            "paths": load_yaml("configs/paths.yaml"), # Where things are located
            "model": load_yaml("configs/model.yaml") # How models are configured
        }
    
    def get(self, config_name, key_path):
        if config_name not in self.configs:
            raise ValueError(f"Unknown config: {config_name}")

        config = self.configs[config_name]
            
        try:
            keys = key_path.split(".")
            value = config

            # Get through all keys
            for k in keys:
                value = value[k]

            # Return dataset path (env path + dataset path)
            if config_name == "paths" and keys[0] == "dataset":
                return os.path.join(self.data_root, value)

            return value
        except KeyError:
            raise KeyError(f"Invalid config path: {key_path}")