#!/usr/bin/env python3
"""
Stop EC2 instance after training or for cost savings.

This script:
1. Optionally verifies training artifacts are saved to S3
2. Shows cost savings summary
3. Stops the EC2 instance
4. Waits for instance to reach 'stopped' state

Usage:
    poetry run python scripts/setup/stop_ec2.py [--config-dir CONFIG_DIR] [--verify-s3] [--force]

Example:
    poetry run python scripts/setup/stop_ec2.py
    poetry run python scripts/setup/stop_ec2.py --verify-s3  # Check S3 before stopping
    poetry run python scripts/setup/stop_ec2.py --force       # Skip confirmation prompt
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.aws_helpers import AWSClient, EC2Manager, S3Manager
from src.utils.config import load_all_configs
from src.utils.logger import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stop EC2 instance to save costs"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing config files (default: config)",
    )
    parser.add_argument(
        "--verify-s3",
        action="store_true",
        help="Verify model artifacts exist in S3 before stopping",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    return parser.parse_args()


def verify_s3_artifacts(s3_manager: S3Manager, bucket: str, prefix: str) -> bool:
    """
    Verify that training artifacts exist in S3.
    
    Args:
        s3_manager: S3Manager instance
        bucket: S3 bucket name
        prefix: S3 prefix/path
        
    Returns:
        True if artifacts found, False otherwise
    """
    logger.info(f"Checking S3 for artifacts: s3://{bucket}/{prefix}")
    
    try:
        # List objects with the prefix
        objects = s3_manager.list_objects(bucket, prefix)
        
        if not objects:
            logger.warning(f"No artifacts found in s3://{bucket}/{prefix}")
            return False
        
        logger.info(f"Found {len(objects)} objects in S3:")
        for obj in objects[:10]:  # Show first 10
            size_mb = obj.get('Size', 0) / (1024 * 1024)
            logger.info(f"  - {obj['Key']} ({size_mb:.2f} MB)")
        
        if len(objects) > 10:
            logger.info(f"  ... and {len(objects) - 10} more objects")
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking S3: {e}")
        return False


def wait_for_instance_stopped(ec2_manager: EC2Manager, instance_id: str, timeout: int = 500):
    """
    Wait for EC2 instance to reach 'stopped' state.
    
    Args:
        ec2_manager: EC2Manager instance
        instance_id: EC2 instance ID
        timeout: Maximum seconds to wait (default: 300)
        
    Returns:
        True if instance is stopped, False if timeout
    """
    logger.info(f"Waiting for instance {instance_id} to reach 'stopped' state...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status_info = ec2_manager.get_instance_status(instance_id)
        state = status_info['state']
        logger.info(f"Current state: {state}")
        
        if state == "stopped":
            logger.success(f"Instance {instance_id} is now stopped!")
            return True
        elif state in ["terminated", "terminating"]:
            logger.error(f"Instance {instance_id} is {state}!")
            return False
        
        time.sleep(10)
    
    logger.error(f"Timeout waiting for instance to stop (waited {timeout}s)")
    return False


def calculate_cost_savings(instance_type: str = "g6.2xlarge"):
    """Calculate and display cost savings."""
    hourly_rate = 0.7512  # g6.2xlarge on-demand rate
    daily_savings = hourly_rate * 24
    monthly_savings = daily_savings * 30
    
    logger.info("\n" + "=" * 60)
    logger.info("💰 Cost Savings Summary")
    logger.info("=" * 60)
    logger.info(f"Instance Type: {instance_type}")
    logger.info(f"Hourly Rate: ${hourly_rate:.4f}/hour")
    logger.info(f"Daily Savings: ${daily_savings:.2f}/day when stopped")
    logger.info(f"Monthly Savings: ${monthly_savings:.2f}/month when stopped")
    logger.info("\nYou only pay for:")
    logger.info("  - EBS storage: ~$16/month (2 × 100GB gp3 volumes)")
    logger.info("  - S3 storage: ~$1-5/month (model artifacts)")
    logger.info("  - ECR storage: ~$5/month (Docker images)")
    logger.info("\nTotal monthly cost when stopped: ~$22-26/month")
    logger.info("=" * 60)


def main():
    """Main execution function."""
    args = parse_args()
    
    try:
        logger.info("=" * 60)
        logger.info("Stopping EC2 Instance")
        logger.info("=" * 60)
        
        # Load configurations
        logger.info(f"Loading configurations from '{args.config_dir}'...")
        configs = load_all_configs(args.config_dir, use_ssm=True)
        
        # Get configuration values
        instance_id = configs.get_aws("aws.ec2.instance_id")
        region = configs.get_aws("aws.region")
        s3_bucket = configs.get_aws("aws.s3.bucket")
        s3_prefix = configs.get_aws("aws.s3.prefix")
        
        logger.info(f"Instance ID: {instance_id}")
        logger.info(f"Region: {region}")
        
        # Initialize AWS client and managers
        aws_client = AWSClient(region=region)
        ec2_manager = EC2Manager(client=aws_client)
        
        # Check current instance state
        logger.info("\nChecking current instance state...")
        status_info = ec2_manager.get_instance_status(instance_id)
        current_state = status_info['state']
        logger.info(f"Current state: {current_state}")
        
        if current_state == "stopped":
            logger.info("Instance is already stopped!")
            calculate_cost_savings()
            logger.success("\n✅ No action needed - instance is stopped")
            return 0
        
        elif current_state != "running":
            logger.warning(f"Instance is in '{current_state}' state")
            logger.warning("Cannot stop instance in this state")
            return 1
        
        # Verify S3 artifacts if requested
        if args.verify_s3:
            logger.info("\n" + "=" * 60)
            logger.info("Verifying S3 Artifacts")
            logger.info("=" * 60)
            
            s3_manager = S3Manager(client=aws_client)
            has_artifacts = verify_s3_artifacts(s3_manager, s3_bucket, s3_prefix)
            
            if not has_artifacts:
                logger.warning("\n⚠️  No training artifacts found in S3!")
                logger.warning("This might indicate training hasn't completed or failed.")
                
                if not args.force:
                    response = input("\nDo you still want to stop the instance? (yes/no): ")
                    if response.lower() not in ['yes', 'y']:
                        logger.info("Stop cancelled by user")
                        return 0
        
        # Show cost savings
        calculate_cost_savings()
        
        # Confirmation prompt (unless --force)
        if not args.force:
            logger.info("\n" + "=" * 60)
            logger.warning("⚠️  You are about to stop the EC2 instance")
            logger.info("=" * 60)
            logger.info("This will:")
            logger.info("  ✅ Stop hourly charges ($0.7512/hour)")
            logger.info("  ✅ Preserve all data on EBS volumes")
            logger.info("  ✅ Keep Docker images in ECR")
            logger.info("  ✅ Keep model artifacts in S3")
            logger.info("\nYou can restart anytime with:")
            logger.info("  python scripts/setup/start_ec2.py")
            
            response = input("\nProceed with stopping instance? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                logger.info("Stop cancelled by user")
                return 0
        
        # Stop the instance
        logger.info("\n" + "=" * 60)
        logger.info("Stopping EC2 Instance")
        logger.info("=" * 60)
        
        success = ec2_manager.stop_instance(instance_id)
        
        if not success:
            logger.error("Failed to send stop command")
            return 1
        
        logger.success("Stop command sent successfully!")
        
        # Wait for instance to stop
        if not wait_for_instance_stopped(ec2_manager, instance_id):
            logger.error("\n❌ Instance failed to reach 'stopped' state")
            logger.warning("Check AWS Console for instance status")
            return 1
        
        # Success!
        logger.success("\n" + "=" * 60)
        logger.success("✅ EC2 Instance Stopped Successfully!")
        logger.success("=" * 60)
        logger.info("\nYour instance is now stopped and not incurring compute charges.")
        logger.info("\nTo restart:")
        logger.info("  python scripts/setup/start_ec2.py")
        logger.info("\nTo deploy environment after restart:")
        logger.info("  python scripts/setup/deploy_via_ssm.py")
        logger.info("\nStorage costs continue:")
        logger.info("  - EBS volumes: ~$16/month")
        logger.info("  - S3 artifacts: ~$1-5/month")
        logger.info("  - ECR images: ~$5/month")
        logger.info("  Total: ~$22-26/month")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
