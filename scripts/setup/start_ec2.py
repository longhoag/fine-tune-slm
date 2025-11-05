#!/usr/bin/env python3
"""
Start EC2 instance and verify SSM connectivity.

This script:
1. Retrieves instance ID from SSM Parameter Store
2. Starts the EC2 instance if it's stopped
3. Waits for instance to reach 'running' state
4. Waits for SSM agent to become 'Online'
5. Verifies instance is ready for deployment

Usage:
    poetry run python scripts/setup/start_ec2.py [--config-dir CONFIG_DIR]

Example:
    poetry run python scripts/setup/start_ec2.py
    poetry run python scripts/setup/start_ec2.py --config-dir config
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.aws_helpers import AWSClient, EC2Manager, SSMManager
from src.utils.config import load_all_configs
from src.utils.logger import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start EC2 instance for fine-tuning"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing config files (default: config)",
    )
    return parser.parse_args()


def wait_for_instance_running(ec2_manager: EC2Manager, instance_id: str, timeout: int = 300):
    """
    Wait for EC2 instance to reach 'running' state.
    
    Args:
        ec2_manager: EC2Manager instance
        instance_id: EC2 instance ID
        timeout: Maximum seconds to wait (default: 300)
        
    Returns:
        True if instance is running, False if timeout
    """
    logger.info(f"Waiting for instance {instance_id} to reach 'running' state...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status_info = ec2_manager.get_instance_status(instance_id)
        state = status_info['state']
        logger.info(f"Current state: {state}")
        
        if state == "running":
            logger.success(f"Instance {instance_id} is now running!")
            return True
        elif state in ["terminated", "terminating"]:
            logger.error(f"Instance {instance_id} is {state}. Cannot start.")
            return False
        
        time.sleep(10)
    
    logger.error(f"Timeout waiting for instance to start (waited {timeout}s)")
    return False


def wait_for_ssm_online(aws_client: AWSClient, instance_id: str, timeout: int = 300):
    """
    Wait for SSM agent to become 'Online'.
    
    Args:
        aws_client: AWSClient instance  
        instance_id: EC2 instance ID
        timeout: Maximum seconds to wait (default: 300)
        
    Returns:
        True if SSM is online, False if timeout
    """
    logger.info(f"Waiting for SSM agent to become 'Online' on {instance_id}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = aws_client.ssm.describe_instance_information(
                Filters=[{'Key': 'InstanceIds', 'Values': [instance_id]}]
            )
            
            if response['InstanceInformationList']:
                status = response['InstanceInformationList'][0]['PingStatus']
                logger.info(f"SSM PingStatus: {status}")
                
                if status == "Online":
                    logger.success(f"SSM agent is now Online on {instance_id}!")
                    return True
                elif status == "ConnectionLost":
                    logger.warning("SSM agent connection lost, waiting for reconnection...")
            else:
                logger.debug(f"Instance not yet registered with SSM")
        except Exception as e:
            logger.debug(f"SSM status check failed (will retry): {e}")
        
        time.sleep(10)
    
    logger.error(f"Timeout waiting for SSM agent (waited {timeout}s)")
    return False


def main():
    """Main execution function."""
    args = parse_args()
    
    try:
        logger.info("=" * 60)
        logger.info("Starting EC2 Instance for Fine-Tuning")
        logger.info("=" * 60)
        
        # Load configurations
        logger.info(f"Loading configurations from '{args.config_dir}'...")
        configs = load_all_configs(args.config_dir, use_ssm=True)
        
        # Get instance ID from SSM Parameter Store
        instance_id = configs.get_aws("aws.ec2.instance_id")
        region = configs.get_aws("aws.region")
        
        logger.info(f"Instance ID: {instance_id}")
        logger.info(f"Region: {region}")
        
        # Initialize AWS client and managers
        aws_client = AWSClient(region=region)
        ec2_manager = EC2Manager(client=aws_client)
        # Note: SSM connectivity checked via aws_client.ssm directly
        
        # Check current instance state
        logger.info("\nChecking current instance state...")
        status_info = ec2_manager.get_instance_status(instance_id)
        current_state = status_info['state']
        logger.info(f"Current state: {current_state}")
        
        if current_state == "running":
            logger.info("Instance is already running!")
            
            # Still verify SSM connectivity
            if wait_for_ssm_online(aws_client, instance_id, timeout=60):
                logger.success("\n✅ Instance is running and SSM is Online!")
                logger.success("Ready for deployment!")
                return 0
            else:
                logger.warning("Instance is running but SSM agent not online")
                logger.info("Waiting for SSM to come online...")
                if wait_for_ssm_online(aws_client, instance_id):
                    logger.success("\n✅ Instance is ready for deployment!")
                    return 0
                else:
                    logger.error("\n❌ SSM agent did not come online")
                    return 1
        
        elif current_state == "stopped":
            logger.info("\nStarting EC2 instance...")
            
            # Start instance using boto3 directly (faster - no status check wait)
            try:
                aws_client.ec2.start_instances(InstanceIds=[instance_id])
                logger.success("Start command sent successfully!")
            except Exception as e:
                logger.error(f"Failed to start instance: {e}")
                return 1
            
            # Wait for instance to start (just 'running' state, not status checks)
            if not wait_for_instance_running(ec2_manager, instance_id):
                logger.error("\n❌ Instance failed to reach 'running' state")
                return 1
            
            # Wait for SSM agent to come online
            if not wait_for_ssm_online(aws_client, instance_id):
                logger.error("\n❌ SSM agent did not come online")
                logger.warning("Instance is running but not accessible via SSM")
                logger.warning("You may need to check SSM agent on the instance")
                return 1
            
            # Get instance details
            instance_info = ec2_manager.get_instance_status(instance_id)
            logger.info("\nInstance Details:")
            logger.info(f"  Instance Type: {instance_info.get('instance_type', 'N/A')}")
            logger.info(f"  Availability Zone: {instance_info.get('availability_zone', 'N/A')}")
            logger.info(f"  Private IP: {instance_info.get('private_ip', 'N/A')}")
            logger.info(f"  Public IP: {instance_info.get('public_ip', 'N/A')}")
            
            logger.success("\n" + "=" * 60)
            logger.success("✅ EC2 Instance Started Successfully!")
            logger.success("=" * 60)
            logger.info("\nNext steps:")
            logger.info("  1. Run: python scripts/setup/deploy_via_ssm.py")
            logger.info("  2. This will mount EBS volume and pull Docker image")
            logger.info("\nCost Alert:")
            logger.info("  💰 Instance is now running at $0.7512/hour")
            logger.info("  💰 Remember to stop when done: python scripts/setup/stop_ec2.py")
            
            return 0
            
        elif current_state in ["pending", "stopping"]:
            logger.warning(f"\nInstance is in '{current_state}' state")
            logger.info("Waiting for instance to stabilize...")
            
            # Wait longer for transition states
            max_wait = 120  # 2 minutes max
            waited = 0
            while waited < max_wait:
                time.sleep(10)
                waited += 10
                
                new_status = ec2_manager.get_instance_status(instance_id)
                new_state = new_status['state']
                logger.info(f"Current state: {new_state} (waited {waited}s)")
                
                if new_state == "running":
                    logger.success("Instance is now running!")
                    if wait_for_ssm_online(aws_client, instance_id):
                        logger.success("\n✅ Instance is ready for deployment!")
                        return 0
                    else:
                        logger.error("\n❌ SSM agent did not come online")
                        return 1
                        
                elif new_state == "stopped":
                    logger.info("Instance is now stopped, starting it...")
                    # Restart from the beginning with stopped state
                    try:
                        aws_client.ec2.start_instances(InstanceIds=[instance_id])
                        logger.success("Start command sent successfully!")
                    except Exception as e:
                        logger.error(f"Failed to start instance: {e}")
                        return 1
                    
                    if not wait_for_instance_running(ec2_manager, instance_id):
                        logger.error("\n❌ Instance failed to reach 'running' state")
                        return 1
                    
                    if not wait_for_ssm_online(aws_client, instance_id):
                        logger.error("\n❌ SSM agent did not come online")
                        return 1
                    
                    logger.success("\n✅ Instance is ready for deployment!")
                    return 0
                    
                elif new_state not in ["pending", "stopping"]:
                    logger.error(f"Instance in unexpected state: {new_state}")
                    return 1
            
            logger.error(f"Timeout waiting for instance to stabilize from '{current_state}'")
            return 1
        
        else:
            logger.error(f"\nUnexpected instance state: {current_state}")
            logger.error("Cannot start instance")
            return 1
    
    except KeyboardInterrupt:
        logger.warning("\n\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
