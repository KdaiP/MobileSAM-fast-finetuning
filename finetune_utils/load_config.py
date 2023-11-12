import json
import argparse

from typing import Any, Dict, Union
from pathlib import Path

DEFAULT_CONFIG_PATH = './configs/mobileSAM.json'

class Args:
    """
    A simple utility class to convert dictionary to an object.
    Nested dictionaries are converted to nested Args objects.
    
    Example:
        config = {
            "model": "SAM",
            "dataset": {
                "train": "./train",
                "val": "./val"
            }
        }
        args = Args(config)
        print(args.model)          # SAM
        print(args.dataset.train)  # ./train
    """
    
    def __init__(self, dictionary: Dict[str, Any]):
        """
        Initialize the Args object from a dictionary.
        
        Args:
            dictionary (dict): The input dictionary to be converted.
        """
        for key, value in dictionary.items():
            if not isinstance(key, str):
                raise ValueError(f"Expected string as dictionary key, got {type(key).__name__}")

            if isinstance(value, dict):
                setattr(self, key, Args(value))
            else:
                setattr(self, key, value)

    def __repr__(self, indent: int = 0) -> str:
        spaces = ' ' * indent
        result = []
        for key, value in self.__dict__.items():
            if isinstance(value, Args):
                value_str = '\n' + value.__repr__(indent + 2)
            else:
                value_str = str(value)
            result.append(f"{spaces}{key}: {value_str}")
        return '\n'.join(result)

def load_args_from_json(json_file: Path) -> Args:
    """
    Load parameters from a given JSON file and return as an Args object.

    Args:
        json_file (Path): Path to the JSON configuration file.

    Returns:
        Args: Parameters loaded from JSON as an Args object.
    """
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file {json_file} not found!")
    
    with open(json_file, 'r') as f:
        params = json.load(f)
    
    return Args(params)

def get_config() -> Args:
    """
    Parse command-line arguments and return configuration as an Args object.

    Returns:
        Args: Parsed configuration as an Args object.
    """
    parser = argparse.ArgumentParser(description='PyTorch MobileSAM Training')
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH, type=Path, help='path to the config file')
    args = parser.parse_args()
    
    config = load_args_from_json(args.config)
    return config