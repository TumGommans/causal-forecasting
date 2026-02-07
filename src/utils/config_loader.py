"""Configuration loader utility for YAML files."""

from pathlib import Path
from typing import Any, Dict, Union
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing configuration parameters.
        
    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If the YAML file is malformed.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Dictionary containing configuration parameters.
    config_path : str or Path
        Path where to save the YAML configuration file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)