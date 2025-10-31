"""
Configuration management utilities.

Load and validate configuration files for AWS, training, and application settings.
Automatically resolves SSM Parameter Store references.
"""

from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger
import yaml
import boto3
from botocore.exceptions import ClientError


class ConfigLoader:
    """Configuration manager with SSM Parameter Store resolution."""
    
    def __init__(self, config_path: str, use_ssm: bool = True, region: str = "us-east-1"):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file
            use_ssm: Whether to resolve SSM parameters (False for local testing)
            region: AWS region for SSM client
        """
        self.config_path = Path(config_path)
        self.use_ssm = use_ssm
        self.region = region
        self._raw_config: Dict[str, Any] = {}
        self._resolved_cache: Dict[str, Any] = {}
        
        # Initialize SSM client if enabled
        self.ssm_client = None
        if use_ssm:
            try:
                self.ssm_client = boto3.client('ssm', region_name=region)
                logger.info(f"Initialized SSM client for region: {region}")
            except Exception as e:
                logger.warning(f"Failed to initialize SSM client: {e}. Will use default values.")
                self.use_ssm = False
        
        # Load configuration
        self._load_yaml()
        
    def _load_yaml(self) -> None:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        logger.info(f"Loading configuration from: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                self._raw_config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded configuration with {len(self._raw_config)} top-level keys")
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            raise
            
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
            # Not an SSM reference, could be a nested config dict
            # Recursively resolve if it has nested dicts
            return self._resolve_nested_config(config_item)
            
        # Check cache first
        if ssm_param in self._resolved_cache:
            logger.debug(f"Using cached value for SSM parameter: {ssm_param}")
            return self._resolved_cache[ssm_param]
            
        # Resolve from SSM if enabled and client available
        if self.use_ssm and self.ssm_client:
            try:
                response = self.ssm_client.get_parameter(
                    Name=ssm_param,
                    WithDecryption=True
                )
                value = response['Parameter']['Value']
                self._resolved_cache[ssm_param] = value
                logger.info(f"Resolved SSM parameter: {ssm_param} = {value}")
                return value
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ParameterNotFound':
                    logger.warning(f"SSM parameter not found: {ssm_param}")
                else:
                    logger.warning(f"Failed to resolve SSM parameter {ssm_param}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error resolving SSM parameter {ssm_param}: {e}")
        
        # Fall back to default value if provided
        default_value = config_item.get('default')
        if default_value is not None:
            logger.info(f"Using default value for {ssm_param}: {default_value}")
            return default_value
            
        # If no default, log warning and return None
        logger.warning(f"No value found for SSM parameter {ssm_param} and no default provided")
        return None
        
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
                # Check if this is an SSM param reference
                if 'ssm_param' in value:
                    resolved[key] = self._resolve_ssm_value(value)
                else:
                    # Recursively resolve nested dict
                    resolved[key] = self._resolve_nested_config(value)
            elif isinstance(value, list):
                # Resolve list items
                resolved[key] = [
                    self._resolve_ssm_value(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                resolved[key] = value
        return resolved
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key (e.g., 'ec2.instance_id' or 'model.name')
            default: Default value if key not found
            
        Returns:
            Configuration value (SSM parameters are automatically resolved)
        
        Examples:
            >>> config = ConfigLoader('config/aws_config.yml')
            >>> instance_id = config.get('ec2.instance_id')
            >>> bucket = config.get('s3.bucket')
        """
        parts = key.split('.')
        value = self._raw_config
        
        # Navigate through nested dict
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
                if value is None:
                    logger.debug(f"Key not found: {key}, using default: {default}")
                    return default
            else:
                logger.debug(f"Cannot navigate further in key: {key}, using default: {default}")
                return default
        
        # Resolve SSM parameter if applicable
        if isinstance(value, dict) and 'ssm_param' in value:
            resolved = self._resolve_ssm_value(value)
            return resolved if resolved is not None else default
        
        return value if value is not None else default
        
    def get_all_resolved(self) -> Dict[str, Any]:
        """
        Get entire configuration with all SSM parameters resolved.
        
        Returns:
            Fully resolved configuration dictionary
        """
        return self._resolve_nested_config(self._raw_config)
        
    def get_raw(self) -> Dict[str, Any]:
        """
        Get raw configuration without SSM resolution.
        
        Returns:
            Raw configuration dictionary
        """
        return self._raw_config.copy()
        
    def validate_required_keys(self, required_keys: list) -> bool:
        """
        Validate that required keys exist in configuration.
        
        Args:
            required_keys: List of required keys in dot notation
            
        Returns:
            True if all required keys exist
            
        Raises:
            ValueError: If any required key is missing
        """
        missing_keys = []
        for key in required_keys:
            value = self.get(key)
            if value is None:
                missing_keys.append(key)
        
        if missing_keys:
            error_msg = f"Missing required configuration keys: {', '.join(missing_keys)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("All required configuration keys present")
        return True
        
    def reload(self) -> None:
        """Reload configuration from file and clear cache."""
        logger.info("Reloading configuration...")
        self._resolved_cache.clear()
        self._load_yaml()
        logger.info("Configuration reloaded successfully")


class MultiConfigLoader:
    """Load and manage multiple configuration files."""
    
    def __init__(self, config_dir: str = "config", use_ssm: bool = True, region: str = "us-east-1"):
        """
        Initialize multi-config loader.
        
        Args:
            config_dir: Directory containing config files
            use_ssm: Whether to resolve SSM parameters
            region: AWS region for SSM client
        """
        self.config_dir = Path(config_dir)
        self.use_ssm = use_ssm
        self.region = region
        
        # Load all configs
        self.aws_config = self._load_if_exists('aws_config.yml')
        self.training_config = self._load_if_exists('training_config.yml')
        
        logger.info("All configurations loaded successfully")
        
    def _load_if_exists(self, filename: str) -> Optional[ConfigLoader]:
        """Load config file if it exists."""
        config_path = self.config_dir / filename
        if config_path.exists():
            logger.info(f"Loading {filename}")
            return ConfigLoader(str(config_path), use_ssm=self.use_ssm, region=self.region)
        else:
            logger.warning(f"Configuration file not found: {config_path}")
            return None
            
    def get_aws(self, key: str, default: Any = None) -> Any:
        """Get value from AWS configuration."""
        if self.aws_config:
            return self.aws_config.get(key, default)
        return default
        
    def get_training(self, key: str, default: Any = None) -> Any:
        """Get value from training configuration."""
        if self.training_config:
            return self.training_config.get(key, default)
        return default
        
    def get_all_resolved(self) -> Dict[str, Any]:
        """Get all configurations fully resolved."""
        result = {}
        if self.aws_config:
            result['aws'] = self.aws_config.get_all_resolved()
        if self.training_config:
            result['training'] = self.training_config.get_all_resolved()
        return result


# Convenience function for simple use cases
def load_config(
    config_path: str,
    use_ssm: bool = True,
    region: str = "us-east-1"
) -> ConfigLoader:
    """
    Load a single configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        use_ssm: Whether to resolve SSM parameters (set False for local testing)
        region: AWS region for SSM client
        
    Returns:
        ConfigLoader instance
        
    Example:
        >>> config = load_config('config/aws_config.yml')
        >>> instance_id = config.get('ec2.instance_id')
    """
    return ConfigLoader(config_path, use_ssm=use_ssm, region=region)


# Convenience function for loading all configs
def load_all_configs(
    config_dir: str = "config",
    use_ssm: bool = True,
    region: str = "us-east-1"
) -> MultiConfigLoader:
    """
    Load all configuration files from a directory.
    
    Args:
        config_dir: Directory containing config files
        use_ssm: Whether to resolve SSM parameters
        region: AWS region for SSM client
        
    Returns:
        MultiConfigLoader instance
        
    Example:
        >>> configs = load_all_configs('config')
        >>> instance_id = configs.get_aws('ec2.instance_id')
        >>> model_name = configs.get_training('model.name')
    """
    return MultiConfigLoader(config_dir, use_ssm=use_ssm, region=region)
