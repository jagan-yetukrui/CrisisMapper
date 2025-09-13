"""
Configuration management utilities for CrisisMapper.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Use default config path
        current_dir = Path(__file__).parent.parent.parent
        config_path = current_dir / "config" / "settings.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables if present
        config = _override_with_env_vars(config)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def _override_with_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override configuration values with environment variables.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Updated configuration dictionary
    """
    # Model configuration overrides
    if os.getenv('CRISIS_MAPPER_MODEL_NAME'):
        config['model']['name'] = os.getenv('CRISIS_MAPPER_MODEL_NAME')
    
    if os.getenv('CRISIS_MAPPER_MODEL_PATH'):
        config['model']['weights_path'] = os.getenv('CRISIS_MAPPER_MODEL_PATH')
    
    if os.getenv('CRISIS_MAPPER_DEVICE'):
        config['model']['device'] = os.getenv('CRISIS_MAPPER_DEVICE')
    
    if os.getenv('CRISIS_MAPPER_CONFIDENCE'):
        config['model']['confidence_threshold'] = float(os.getenv('CRISIS_MAPPER_CONFIDENCE'))
    
    # Data source credentials
    if os.getenv('NASA_EARTHDATA_USERNAME'):
        config['data_sources']['nasa_earthdata']['username'] = os.getenv('NASA_EARTHDATA_USERNAME')
    
    if os.getenv('NASA_EARTHDATA_PASSWORD'):
        config['data_sources']['nasa_earthdata']['password'] = os.getenv('NASA_EARTHDATA_PASSWORD')
    
    if os.getenv('SENTINEL2_USERNAME'):
        config['data_sources']['sentinel2']['username'] = os.getenv('SENTINEL2_USERNAME')
    
    if os.getenv('SENTINEL2_PASSWORD'):
        config['data_sources']['sentinel2']['password'] = os.getenv('SENTINEL2_PASSWORD')
    
    # API configuration
    if os.getenv('CRISIS_MAPPER_API_HOST'):
        config['api']['host'] = os.getenv('CRISIS_MAPPER_API_HOST')
    
    if os.getenv('CRISIS_MAPPER_API_PORT'):
        config['api']['port'] = int(os.getenv('CRISIS_MAPPER_API_PORT'))
    
    if os.getenv('CRISIS_MAPPER_DEBUG'):
        config['api']['debug'] = os.getenv('CRISIS_MAPPER_DEBUG').lower() == 'true'
    
    # Output directory
    if os.getenv('CRISIS_MAPPER_OUTPUT_DIR'):
        config['output']['base_dir'] = os.getenv('CRISIS_MAPPER_OUTPUT_DIR')
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_sections = ['model', 'disaster_classes', 'data_processing', 'geospatial']
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: {section}")
            return False
    
    # Validate model configuration
    model_config = config['model']
    required_model_keys = ['name', 'weights_path', 'confidence_threshold', 'iou_threshold']
    
    for key in required_model_keys:
        if key not in model_config:
            logger.error(f"Missing required model configuration: {key}")
            return False
    
    # Validate disaster classes
    disaster_classes = config['disaster_classes']
    if not isinstance(disaster_classes, list) or len(disaster_classes) == 0:
        logger.error("disaster_classes must be a non-empty list")
        return False
    
    for i, cls in enumerate(disaster_classes):
        required_class_keys = ['name', 'id', 'color']
        for key in required_class_keys:
            if key not in cls:
                logger.error(f"Missing required class configuration at index {i}: {key}")
                return False
    
    logger.info("Configuration validation passed")
    return True
