#!/usr/bin/env python3
"""
Clean up test checkpoints from EC2 EBS volume.

This script removes:
- checkpoint-* directories (from test/interrupted runs)
- final_model directory (from old test runs)  
- backup_* directories (old backups)

Usage:
    # Clean all checkpoints
    poetry run python scripts/utils/cleanup_checkpoints.py
    
    # Dry run (see what would be deleted)
    poetry run python scripts/utils/cleanup_checkpoints.py --dry-run
    
    # Keep final_model, only delete checkpoints
    poetry run python scripts/utils/cleanup_checkpoints.py --keep-final-model
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.utils.config import load_all_configs
from src.utils.aws_helpers import AWSClient, EC2Manager, SSMManager


def cleanup_checkpoints(
    ssm_manager: SSMManager,
    instance_id: str,
    checkpoint_dir: str = "/mnt/training/checkpoints",
    keep_final_model: bool = False,
    dry_run: bool = False
) -> bool:
    """
    Clean up checkpoint directories on EC2 instance.
    
    Args:
        ssm_manager: SSMManager instance
        instance_id: EC2 instance ID
        checkpoint_dir: Base checkpoint directory
        keep_final_model: If True, don't delete final_model
        dry_run: If True, only list files without deleting
        
    Returns:
        True if successful
    """
    logger.info("="*60)
    logger.info("Checkpoint Cleanup")
    logger.info("="*60)
    
    # Build cleanup command
    commands = []
    
    if dry_run:
        commands.append(f"echo '=== Current contents of {checkpoint_dir} ==='")
        commands.append(f"ls -lh {checkpoint_dir}/ || echo 'Directory empty or not found'")
        commands.append(f"echo ''")
        commands.append(f"echo '=== Files that would be deleted ==='")
        commands.append(f"ls -d {checkpoint_dir}/checkpoint-* 2>/dev/null || echo 'No checkpoint-* found'")
        commands.append(f"ls -d {checkpoint_dir}/backup_* 2>/dev/null || echo 'No backup_* found'")
        if not keep_final_model:
            commands.append(f"ls -d {checkpoint_dir}/final_model 2>/dev/null || echo 'No final_model found'")
    else:
        commands.append(f"echo 'Cleaning up checkpoints in {checkpoint_dir}...'")
        commands.append(f"rm -rf {checkpoint_dir}/checkpoint-* && echo '✓ Deleted checkpoint-*' || echo '- No checkpoint-* found'")
        commands.append(f"rm -rf {checkpoint_dir}/backup_* && echo '✓ Deleted backup_*' || echo '- No backup_* found'")
        if not keep_final_model:
            commands.append(f"rm -rf {checkpoint_dir}/final_model && echo '✓ Deleted final_model' || echo '- No final_model found'")
        commands.append(f"echo ''")
        commands.append(f"echo '=== Remaining contents ==='")
        commands.append(f"ls -lh {checkpoint_dir}/ || echo 'Directory is now empty'")
    
    cmd = " && ".join(commands)
    
    logger.info(f"Target directory: {checkpoint_dir}")
    if dry_run:
        logger.info("Mode: DRY RUN (no files will be deleted)")
    if keep_final_model:
        logger.info("Keeping: final_model")
    
    # Send command via SSM
    logger.info("\n📤 Sending cleanup command via SSM...")
    
    try:
        command_id = ssm_manager.send_command(
            instance_id=instance_id,
            commands=[f"export AWS_DEFAULT_REGION=us-east-1 && {cmd}"],
            comment="Cleanup checkpoints",
            timeout_seconds=120
        )
        
        logger.success(f"Command sent! Command ID: {command_id}")
        logger.info("⏳ Waiting for command to complete...")
        
        # Wait for result
        output = ssm_manager.wait_for_command(
            command_id=command_id,
            instance_id=instance_id,
            timeout=120,
            poll_interval=5
        )
        
        if output['status'] == 'Success':
            logger.success("\n✅ Cleanup completed successfully!")
            logger.info("\nOutput:")
            logger.info(output['stdout'])
            return True
        else:
            logger.error(f"\n❌ Cleanup failed with status: {output['status']}")
            logger.error(f"\nStdout:\n{output['stdout']}")
            logger.error(f"\nStderr:\n{output['stderr']}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to execute cleanup: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean up checkpoints from EC2 EBS volume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean all checkpoints and final_model
  poetry run python scripts/utils/cleanup_checkpoints.py
  
  # Dry run to see what would be deleted
  poetry run python scripts/utils/cleanup_checkpoints.py --dry-run
  
  # Keep final_model, only delete checkpoint-* and backup_*
  poetry run python scripts/utils/cleanup_checkpoints.py --keep-final-model
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
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--keep-final-model",
        action="store_true",
        help="Don't delete final_model directory"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configs
        logger.info(f"Loading configurations from '{args.config_dir}'...")
        configs = load_all_configs(args.config_dir, use_ssm=True)
        
        instance_id = configs.get_aws('aws.ec2.instance_id')
        region = configs.get_aws('aws.region')
        
        logger.info(f"Instance ID: {instance_id}")
        logger.info(f"Region: {region}")
        
        # Initialize AWS clients
        aws_client = AWSClient(region=region)
        ec2_manager = EC2Manager(aws_client)
        ssm_manager = SSMManager(aws_client)
        
        # Check instance status
        logger.info("\n🔍 Checking instance status...")
        status_info = ec2_manager.get_instance_status(instance_id)
        
        if status_info['state'] != "running":
            logger.error(f"❌ Instance is not running (state: {status_info['state']})")
            logger.info("\nStart the instance first:")
            logger.info("  poetry run python scripts/setup/start_ec2.py")
            sys.exit(1)
        
        logger.success("✅ Instance is running")
        
        # Check SSM connectivity
        if not ssm_manager.is_instance_online(instance_id):
            logger.error("❌ Instance is not reachable via SSM")
            logger.info("Wait a moment for SSM agent to connect, then try again")
            sys.exit(1)
        
        logger.success("✅ SSM is online")
        
        # Run cleanup
        success = cleanup_checkpoints(
            ssm_manager=ssm_manager,
            instance_id=instance_id,
            keep_final_model=args.keep_final_model,
            dry_run=args.dry_run
        )
        
        if success:
            if not args.dry_run:
                logger.info("\n💡 Tip: You can now stop the instance to save costs:")
                logger.info("  poetry run python scripts/setup/stop_ec2.py")
            sys.exit(0)
        else:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
