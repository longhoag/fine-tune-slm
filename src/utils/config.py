"""
Configuration management utilities.

Load and validate configuration files for AWS, training, and application settings.
Automatically resolves SSM Parameter Store references.
"""

from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger
import yaml


class Config:
    """Configuration manager with SSM Parameter Store resolution."""
    
    def __init__(self, config_dir: str = "config", ssm_client=None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing config files
            ssm_client: AWS SSM client (optional, for parameter resolution)
        """
        self.config_dir = Path(config_dir)
        self.ssm_client = ssm_client
        self._aws_config: Dict[str, Any] = {}
        self._training_config: Dict[str, Any] = {}
        self._resolved_cache: Dict[str, Any] = {}  # Cache for resolved SSM parameters
        
    def load_aws_config(self) -> Dict[str, Any]:
        """Load AWS configuration."""
        config_path = self.config_dir / "aws_config.yml"
        logger.info(f"Loading AWS config from: {config_path}")
        
        # TODO: Load YAML file
        # with open(config_path) as f:
        #     self._aws_config = yaml.safe_load(f)
        
        # TODO: Validate required fields
        
        return self._aws_config
        
    def load_training_config(self) -> Dict[str, Any]:
        """Load training configuration."""
        config_path = self.config_dir / "training_config.yml"
        logger.info(f"Loading training config from: {config_path}")
        
        # TODO: Load YAML file
        # with open(config_path) as f:
        #     self._training_config = yaml.safe_load(f)
        
        # TODO: Validate required fields
        
        return self._training_config
        
    def _resolve_ssm_value(self, config_item: Any) -> Any:
        """
        Resolve SSM parameter if config item contains ssm_param key.
        
        Args:
            config_item: Config value (dict with ssm_param or direct value)
            
        Returns:
            Resolved value from SSM or default value
        """
        # If not a dict, return as-is
        if not isinstance(config_item, dict):
            return config_item
            
        # Check if this is an SSM parameter reference
        ssm_param = config_item.get('ssm_param')
        if not ssm_param:
            return config_item
            
        # Check cache first
        if ssm_param in self._resolved_cache:
            logger.debug(f"Using cached value for SSM parameter: {ssm_param}")
            return self._resolved_cache[ssm_param]
            
        # Resolve from SSM if client available
        if self.ssm_client:
            try:
                # TODO: Implement SSM parameter retrieval
                # response = self.ssm_client.get_parameter(Name=ssm_param, WithDecryption=True)
                # value = response['Parameter']['Value']
                # self._resolved_cache[ssm_param] = value
                # logger.info(f"Resolved SSM parameter: {ssm_param}")
                # return value
                pass
            except Exception as e:
                logger.warning(f"Failed to resolve SSM parameter {ssm_param}: {e}")
        
        # Fall back to default value if provided
        default_value = config_item.get('default')
        if default_value is not None:
            logger.info(f"Using default value for {ssm_param}: {default_value}")
            return default_value
            
        # If no default, log warning and return the config item
        logger.warning(f"No value found for SSM parameter {ssm_param} and no default provided")
        return config_item
        
    def _resolve_nested_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively resolve all SSM parameters in nested config.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with resolved SSM parameters
        """
        resolved = {}
        for key, value in config.items():
            if isinstance(value, dict):
                # Check if this is an SSM param reference or nested config
                if 'ssm_param' in value:
                    resolved[key] = self._resolve_ssm_value(value)
                else:
                    # Recursively resolve nested dict
                    resolved[key] = self._resolve_nested_config(value)
            elif isinstance(value, list):
                # Resolve list items
                resolved[key] = [self._resolve_ssm_value(item) for item in value]
            else:
                resolved[key] = value
        return resolved
        
    def get(self, key: str, default: Any = None, resolve_ssm: bool = True) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key (e.g., 'aws.ec2.instance_id')
            default: Default value if key not found
            resolve_ssm: Whether to resolve SSM parameters
            
        Returns:
            Configuration value
        """
        # TODO: Implement nested key lookup
        # parts = key.split('.')
        # value = self._aws_config if parts[0] == 'aws' else self._training_config
        # for part in parts[1:]:
        #     value = value.get(part, {})
        # 
        # if resolve_ssm and isinstance(value, dict) and 'ssm_param' in value:
        #     return self._resolve_ssm_value(value)
        # 
        # return value if value != {} else default
        
        return default
        
    def get_all_resolved(self, config_type: str = 'all') -> Dict[str, Any]:
        """
        Get all configuration with SSM parameters resolved.
        
        Args:
            config_type: 'aws', 'training', or 'all'
            
        Returns:
            Fully resolved configuration
        """
        if config_type == 'aws':
            return self._resolve_nested_config(self._aws_config)
        elif config_type == 'training':
            return self._resolve_nested_config(self._training_config)
        else:
            return {
                'aws': self._resolve_nested_config(self._aws_config),
                'training': self._resolve_nested_config(self._training_config)
            }
        
    def validate(self) -> bool:
        """
        Validate all configurations.
        
        Returns:
            True if valid, raises exception otherwise
        """
        # TODO: Check required fields are present
        # TODO: Validate field types and formats
        # TODO: Ensure critical SSM parameters can be resolved
        
        return True


def load_config(config_dir: str = "config", ssm_client=None) -> Config:
    """
    Load all configurations.
    
    Args:
        config_dir: Configuration directory path
        ssm_client: Optional AWS SSM client for parameter resolution
        
    Returns:
        Config instance with loaded configurations
    """
    config = Config(config_dir, ssm_client)
    config.load_aws_config()
    config.load_training_config()
    config.validate()
    return config
