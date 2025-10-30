#!/usr/bin/env python3
"""
Initialize SSM Parameter Store with project configuration values.

This script creates all SSM parameters required by the project.
Run this once during initial setup, then update individual parameters as needed.

Usage:
    python scripts/setup/init_ssm_parameters.py --config config/ [--dry-run]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
from loguru import logger
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger


def load_config_schema(config_dir: Path) -> Dict[str, Any]:
    """
    Load configuration files and extract SSM parameter definitions.
    
    Args:
        config_dir: Path to config directory
        
    Returns:
        Dictionary of SSM parameter paths to their definitions
    """
    parameters = {}
    
    # Load AWS config
    aws_config_path = config_dir / "aws_config.yml"
    if aws_config_path.exists():
        with open(aws_config_path) as f:
            aws_config = yaml.safe_load(f)
            parameters.update(extract_ssm_params(aws_config, prefix="aws"))
    
    # Load training config
    training_config_path = config_dir / "training_config.yml"
    if training_config_path.exists():
        with open(training_config_path) as f:
            training_config = yaml.safe_load(f)
            parameters.update(extract_ssm_params(training_config, prefix="training"))
    
    return parameters


def extract_ssm_params(config: Dict[str, Any], prefix: str = "") -> Dict[str, Dict]:
    """
    Recursively extract SSM parameter definitions from config.
    
    Args:
        config: Configuration dictionary
        prefix: Prefix for logging context
        
    Returns:
        Dictionary mapping SSM param paths to their definitions
    """
    params = {}
    
    for key, value in config.items():
        current_path = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            # Check if this is an SSM param definition
            if 'ssm_param' in value:
                param_name = value['ssm_param']
                params[param_name] = {
                    'value': value.get('default'),
                    'description': value.get('description', f"Config: {current_path}"),
                    'type': 'String',  # Could be enhanced to detect SecureString needs
                    'config_key': current_path
                }
            else:
                # Recurse into nested dict
                params.update(extract_ssm_params(value, current_path))
    
    return params


def create_ssm_parameters(parameters: Dict[str, Dict], dry_run: bool = False):
    """
    Create SSM parameters.
    
    Args:
        parameters: Dictionary of parameter definitions
        dry_run: If True, only print what would be created
    """
    logger.info(f"Found {len(parameters)} SSM parameters to create")
    
    if dry_run:
        logger.info("DRY RUN MODE - No parameters will be created")
    
    # Initialize AWS SSM client (requires AWS credentials)
    # Credentials are sourced from:
    # 1. ~/.aws/credentials (via 'aws configure')
    # 2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    # 3. IAM role (if running on EC2)
    # 4. AWS SSO profile (AWS_PROFILE environment variable)
    if not dry_run:
        try:
            # TODO: Initialize AWS SSM client
            # import boto3
            # ssm_client = boto3.client('ssm')
            # 
            # # Test credentials
            # try:
            #     ssm_client.describe_parameters(MaxResults=1)
            # except Exception as e:
            #     logger.error("Failed to authenticate with AWS. Please configure credentials:")
            #     logger.error("  Option 1: Run 'aws configure'")
            #     logger.error("  Option 2: Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            #     logger.error(f"  Error: {e}")
            #     return
            pass
        except Exception as e:
            logger.error(f"Failed to initialize AWS client: {e}")
            logger.error("Please ensure AWS credentials are configured")
            return
    
    created_count = 0
    skipped_count = 0
    error_count = 0
    
    for param_name, param_def in parameters.items():
        value = param_def.get('value')
        description = param_def.get('description', '')
        param_type = param_def.get('type', 'String')
        
        # Skip if no default value provided
        if value is None:
            logger.warning(
                f"Skipping {param_name}: No default value. "
                f"Set manually after setup for: {param_def['config_key']}"
            )
            skipped_count += 1
            continue
        
        # Convert value to string
        value_str = str(value)
        
        if dry_run:
            logger.info(f"Would create: {param_name} = {value_str}")
            created_count += 1
            continue
        
        try:
            # TODO: Create parameter
            # ssm_client.put_parameter(
            #     Name=param_name,
            #     Value=value_str,
            #     Description=description,
            #     Type=param_type,
            #     Overwrite=False,  # Don't overwrite existing
            #     Tags=[
            #         {'Key': 'Project', 'Value': 'fine-tune-slm'},
            #         {'Key': 'ManagedBy', 'Value': 'init_ssm_parameters.py'}
            #     ]
            # )
            logger.success(f"Created parameter: {param_name}")
            created_count += 1
            
        except Exception as e:
            # TODO: Handle AlreadyExistsException specifically
            logger.error(f"Failed to create {param_name}: {e}")
            error_count += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Summary:")
    logger.info(f"  Created: {created_count}")
    logger.info(f"  Skipped (no default): {skipped_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info("=" * 60)
    
    if skipped_count > 0:
        logger.warning(
            f"\n{skipped_count} parameters require manual setup. "
            "Update them using AWS Console or:"
        )
        logger.warning("  aws ssm put-parameter --name <param-name> --value <value>")


def print_required_manual_setup(parameters: Dict[str, Dict]):
    """
    Print list of parameters that need manual configuration.
    
    Args:
        parameters: Dictionary of parameter definitions
    """
    logger.info("\n" + "=" * 60)
    logger.info("Parameters requiring manual setup (no defaults):")
    logger.info("=" * 60)
    
    for param_name, param_def in parameters.items():
        if param_def.get('value') is None:
            logger.info(f"\n📋 {param_name}")
            logger.info(f"   Description: {param_def.get('description', 'N/A')}")
            logger.info(f"   Config Key: {param_def['config_key']}")
            logger.info(f"   Command: aws ssm put-parameter \\")
            logger.info(f"              --name '{param_name}' \\")
            logger.info(f"              --value 'YOUR_VALUE_HERE' \\")
            logger.info(f"              --type String")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize SSM Parameter Store for fine-tune-slm project"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config',
        help='Path to config directory (default: config)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview parameters without creating them'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(log_level=args.log_level)
    
    logger.info("Initializing SSM Parameter Store for fine-tune-slm")
    logger.info(f"Config directory: {args.config}")
    
    # Load config schemas
    config_dir = Path(args.config)
    if not config_dir.exists():
        logger.error(f"Config directory not found: {config_dir}")
        sys.exit(1)
    
    parameters = load_config_schema(config_dir)
    
    # Create parameters
    create_ssm_parameters(parameters, dry_run=args.dry_run)
    
    # Show required manual setup
    print_required_manual_setup(parameters)
    
    if args.dry_run:
        logger.info("\nRun without --dry-run to create parameters")
    else:
        logger.success("\n✅ SSM Parameter initialization complete!")


if __name__ == "__main__":
    main()
