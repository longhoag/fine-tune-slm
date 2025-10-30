"""
Logging configuration using loguru.

Sets up structured logging for both local scripts and remote execution.
CloudWatch integration for SSM command outputs.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str = None,
    rotation: str = "10 MB",
    retention: str = "1 week"
):
    """
    Configure loguru logger.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        rotation: Log rotation size
        retention: Log retention period
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with custom format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="gz"
        )
        
        logger.info(f"Logging to file: {log_file}")


def log_aws_operation(operation: str, resource: str, details: dict = None):
    """
    Log AWS operation with structured data.
    
    Args:
        operation: Operation name (e.g., 'start_instance', 'send_command')
        resource: Resource identifier
        details: Additional details to log
    """
    logger.info(
        f"AWS Operation: {operation}",
        extra={
            "operation": operation,
            "resource": resource,
            "details": details or {}
        }
    )


def log_training_metrics(epoch: int, step: int, metrics: dict):
    """
    Log training metrics.
    
    Args:
        epoch: Current epoch
        step: Current step
        metrics: Training metrics (loss, accuracy, etc.)
    """
    logger.info(
        f"Training - Epoch {epoch}, Step {step}: {metrics}",
        extra={
            "epoch": epoch,
            "step": step,
            **metrics
        }
    )
