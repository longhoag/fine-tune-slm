#!/usr/bin/env python3
"""
Run fine-tuning on EC2 instance via SSM.

This script:
1. Verifies EC2 instance is running and ready
2. Sends training command via SSM
3. Monitors training progress via CloudWatch logs (optional)
4. Waits for training completion or runs in background

Usage:
    # Dry run - test environment without training
    poetry run python scripts/finetune/run_training.py --dry-run
    
    # Test mode - run 1 step to verify everything works
    poetry run python scripts/finetune/run_training.py --test
    
    # Full training in background
    poetry run python scripts/finetune/run_training.py --background

    # Full training with monitoring (blocks until complete)
    poetry run python scripts/finetune/run_training.py
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.utils.config import load_all_configs
from src.utils.aws_helpers import AWSClient, EC2Manager, SSMManager


def verify_instance_ready(ec2_manager: EC2Manager, ssm_manager: SSMManager, instance_id: str) -> bool:
    """
    Verify EC2 instance is running and SSM is online.
    
    Args:
        ec2_manager: EC2Manager instance
        ssm_manager: SSMManager instance
        instance_id: EC2 instance ID
        
    Returns:
        True if ready, False otherwise
    """
    logger.info("Verifying instance state...")
    
    # Check instance is running
    status_info = ec2_manager.get_instance_status(instance_id)
    if status_info['state'] != "running":
        logger.error(f"Instance is not running. Current state: {status_info['state']}")
        logger.info("Start the instance first: poetry run python scripts/setup/start_ec2.py")
        return False
    
    logger.info("Instance is running ✓")
    
    # Check SSM connectivity
    logger.info("Verifying SSM connectivity...")
    if not ssm_manager.is_instance_online(instance_id):
        logger.error("Instance is not reachable via SSM")
        return False
    
    logger.success("Instance is running and SSM is online! ✓")
    return True


def build_training_command(
    ecr_registry: str,
    repository: str,
    dry_run: bool = False,
    test_mode: bool = False,
    max_steps: int = None,
    output_dir: str = "/mnt/training/checkpoints"
) -> str:
    """
    Build Docker command to run training.
    
    Args:
        ecr_registry: ECR registry URL
        repository: ECR repository name
        dry_run: If True, only validate environment
        test_mode: If True, run minimal training (1 step)
        max_steps: Override max training steps
        output_dir: Checkpoint output directory
        
    Returns:
        Complete bash command string
    """
    # Base Docker run command with GPU access
    # Override entrypoint since Dockerfile has ENTRYPOINT ["python3"]
    docker_cmd_parts = [
        "docker run --rm --gpus all",
        "--entrypoint python3",
        "-v /mnt/training:/mnt/training",
        f"{ecr_registry}/{repository}:latest",
    ]
    
    # Python training command (without 'python' since entrypoint is python3)
    python_cmd_parts = [
        "-m src.train",
        "--use-ssm",  # Use SSM Parameter Store for config
        f"--output-dir {output_dir}",
    ]
    
    if dry_run:
        python_cmd_parts.append("--dry-run")
    
    # Test mode: limited steps
    if test_mode and not dry_run:
        # Run just a few steps to verify everything works
        python_cmd_parts.append("--max-steps 5")
    
    # Override max steps if specified
    if max_steps and not test_mode:
        python_cmd_parts.append(f"--max-steps {max_steps}")
    
    # Combine Docker and Python commands
    full_command = " ".join(docker_cmd_parts) + " " + " ".join(python_cmd_parts)
    
    return full_command


def run_training(
    ssm_manager: SSMManager,
    instance_id: str,
    ecr_registry: str,
    repository: str,
    cloudwatch_log_group: str = None,
    dry_run: bool = False,
    test_mode: bool = False,
    background: bool = False,
    max_steps: int = None,
    timeout: int = 14400  # 4 hours default
) -> dict:
    """
    Execute training on EC2 instance.
    
    Args:
        ssm_manager: SSMManager instance
        instance_id: EC2 instance ID
        ecr_registry: ECR registry URL
        repository: ECR repository name
        cloudwatch_log_group: CloudWatch log group for SSM output
        dry_run: Only validate environment
        test_mode: Run minimal steps to test
        background: Don't wait for completion
        max_steps: Override max training steps
        timeout: Maximum time to wait (seconds)
        
    Returns:
        Command output dict
    """
    if dry_run:
        mode = "DRY RUN (Environment Validation Only)"
    elif test_mode:
        mode = "TEST MODE (5 steps to verify setup)"
    else:
        mode = "FULL TRAINING"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Fine-Tuning: {mode}")
    logger.info(f"{'='*60}\n")
    
    # Build training command
    training_cmd = build_training_command(
        ecr_registry=ecr_registry,
        repository=repository,
        dry_run=dry_run,
        test_mode=test_mode,
        max_steps=max_steps
    )
    
    logger.info("Training command:")
    logger.info(training_cmd)
    logger.info("")
    
    # Set appropriate timeout
    if dry_run or test_mode:
        cmd_timeout = 600  # 10 minutes for validation/test
    else:
        cmd_timeout = timeout
    
    # Send SSM command
    logger.info("Sending training command via SSM...")
    command_id = ssm_manager.send_command(
        instance_id=instance_id,
        commands=[training_cmd],
        comment=f"Fine-tuning: {mode}",
        timeout_seconds=cmd_timeout,
        cloudwatch_log_group=cloudwatch_log_group
    )
    
    logger.success(f"Command sent! Command ID: {command_id}")
    
    if cloudwatch_log_group:
        logger.info(f"📊 CloudWatch logs: {cloudwatch_log_group}")
        logger.info(f"   View: aws logs tail {cloudwatch_log_group} --follow")

    
    if background:
        logger.info("\n🎯 Running in background mode")
        logger.info(f"Command ID: {command_id}")
        logger.info("\nTo check status:")
        logger.info(f"  aws ssm get-command-invocation --command-id {command_id} --instance-id {instance_id}")
        logger.info("\nTo view CloudWatch logs:")
        logger.info("  aws logs tail /aws/ssm/fine-tune-llama --follow")
        return {"status": "InProgress", "command_id": command_id}
    
    # Wait for completion
    logger.info(f"\n⏳ Waiting for training to complete (timeout: {cmd_timeout}s)...")
    logger.info("This may take a while. You can Ctrl+C and check CloudWatch logs instead.")
    
    try:
        output = ssm_manager.wait_for_command(
            command_id=command_id,
            instance_id=instance_id,
            timeout=cmd_timeout,
            poll_interval=30  # Check every 30 seconds
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training completed with status: {output['status']}")
        logger.info(f"{'='*60}\n")
        
        if output['status'] == 'Success':
            logger.success("✅ Training completed successfully!")
            logger.info("\nOutput:")
            logger.info(output['stdout'])
            
            if output['stderr']:
                logger.warning("\nWarnings/Errors:")
                logger.warning(output['stderr'])
        else:
            logger.error(f"❌ Training failed with status: {output['status']}")
            logger.error(f"\nStdout:\n{output['stdout']}")
            logger.error(f"\nStderr:\n{output['stderr']}")
        
        return output
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        logger.info(f"Training is still running in background. Command ID: {command_id}")
        logger.info("\nTo check status:")
        logger.info(f"  aws ssm get-command-invocation --command-id {command_id} --instance-id {instance_id}")
        return {"status": "Interrupted", "command_id": command_id}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run fine-tuning on EC2 instance via SSM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test environment without training
  poetry run python scripts/finetune/run_training.py --dry-run
  
  # Run 5 training steps to verify everything works
  poetry run python scripts/finetune/run_training.py --test
  
  # Full training in background
  poetry run python scripts/finetune/run_training.py --background

  # Full training with live monitoring (blocks until complete)
  poetry run python scripts/finetune/run_training.py
        """
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Configuration directory (default: config/)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate environment without training"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run 5 training steps to test (overrides --dry-run)"
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run in background without waiting for completion"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override max training steps (for custom testing)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=14400,
        help="Maximum time to wait in seconds (default: 14400 = 4 hours)"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Running Fine-Tuning on EC2 Instance")
    logger.info("="*60)
    logger.info(f"Loading configurations from '{args.config_dir}'...")
    
    try:
        # Load configs
        configs = load_all_configs(args.config_dir)
        
        instance_id = configs.get_aws('aws.ec2.instance_id')
        region = configs.get_aws('aws.region')
        ecr_registry = configs.get_aws('aws.ecr.registry')
        repository = configs.get_aws('aws.ecr.repository')
        cloudwatch_log_group = configs.get_aws('aws.cloudwatch.log_group')
        
        logger.info(f"Instance ID: {instance_id}")
        logger.info(f"Region: {region}")
        logger.info(f"ECR Image: {ecr_registry}/{repository}:latest")
        logger.info(f"CloudWatch Logs: {cloudwatch_log_group}")
        
        # Initialize AWS clients
        aws_client = AWSClient(region=region)
        ec2_manager = EC2Manager(aws_client)
        ssm_manager = SSMManager(aws_client)
        
        # Verify instance is ready
        if not verify_instance_ready(ec2_manager, ssm_manager, instance_id):
            logger.error("Instance is not ready. Exiting.")
            logger.info("\nDid you run these first?")
            logger.info("  1. poetry run python scripts/setup/start_ec2.py")
            logger.info("  2. poetry run python scripts/setup/deploy_via_ssm.py")
            sys.exit(1)
        
        # Run training
        result = run_training(
            ssm_manager=ssm_manager,
            instance_id=instance_id,
            ecr_registry=ecr_registry,
            repository=repository,
            cloudwatch_log_group=cloudwatch_log_group,
            dry_run=args.dry_run,
            test_mode=args.test,
            background=args.background,
            max_steps=args.max_steps,
            timeout=args.timeout
        )
        
        # Exit code based on result
        if result['status'] in ['Success', 'InProgress', 'Interrupted']:
            logger.success("\n✅ Script completed successfully!")
            if result['status'] == 'Success':
                logger.info("\nNext steps:")
                logger.info("  1. Check model artifacts: ls /mnt/training/checkpoints")
                logger.info("  2. Push to HuggingFace: python scripts/finetune/push_to_hf.py")
                logger.info("  3. Stop instance: python scripts/setup/stop_ec2.py")
            sys.exit(0)
        else:
            logger.error("\n❌ Training failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
