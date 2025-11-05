#!/usr/bin/env python3
"""
Deploy training environment to EC2 via AWS SSM.

This script:
1. Attaches EBS checkpoint volume (if not already attached)
2. Formats and mounts EBS volume to /mnt/training
3. Pulls Docker image from ECR
4. Verifies GPU access
5. Sets up environment for training

Usage:
    python scripts/setup/deploy_via_ssm.py [--config-dir CONFIG_DIR] [--skip-volume] [--skip-docker]

Example:
    python scripts/setup/deploy_via_ssm.py
    python scripts/setup/deploy_via_ssm.py --skip-volume  # Skip EBS setup
    python scripts/setup/deploy_via_ssm.py --skip-docker  # Skip Docker pull
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
        description="Deploy training environment to EC2 via SSM"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing config files (default: config)",
    )
    parser.add_argument(
        "--skip-volume",
        action="store_true",
        help="Skip EBS volume attachment and mounting",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker image pull",
    )
    return parser.parse_args()


def attach_ebs_volume(ec2_manager: EC2Manager, instance_id: str, volume_id: str, device: str = "/dev/sdf"):
    """
    Attach EBS volume to EC2 instance if not already attached.
    
    Args:
        ec2_manager: EC2Manager instance
        instance_id: EC2 instance ID
        volume_id: EBS volume ID
        device: Device name (default: /dev/sdf)
        
    Returns:
        True if attached successfully or already attached
    """
    logger.info(f"Checking EBS volume {volume_id} attachment status...")
    
    # Check if volume is already attached
    try:
        response = ec2_manager.client.ec2.describe_volumes(VolumeIds=[volume_id])
        volume = response['Volumes'][0]
        attachments = volume.get('Attachments', [])
        
        if attachments:
            attachment = attachments[0]
            if attachment['InstanceId'] == instance_id:
                logger.success(f"Volume {volume_id} is already attached to {instance_id}")
                return True
            else:
                logger.error(f"Volume is attached to different instance: {attachment['InstanceId']}")
                return False
        
        # Volume is available, attach it
        logger.info(f"Attaching volume {volume_id} to {instance_id}...")
        ec2_manager.client.ec2.attach_volume(
            VolumeId=volume_id,
            InstanceId=instance_id,
            Device=device
        )
        
        # Wait for attachment
        logger.info("Waiting for volume to attach...")
        waiter = ec2_manager.client.ec2.get_waiter('volume_in_use')
        waiter.wait(VolumeIds=[volume_id])
        
        logger.success(f"Volume {volume_id} attached successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to attach volume: {e}")
        return False


def setup_ebs_volume(ssm_manager: SSMManager, instance_id: str, mount_path: str = "/mnt/training"):
    """
    Format (if needed) and mount EBS volume via SSM.
    
    Args:
        ssm_manager: SSMManager instance
        instance_id: EC2 instance ID
        mount_path: Mount path (default: /mnt/training)
        
    Returns:
        True if successful
    """
    logger.info(f"Setting up EBS volume at {mount_path}...")
    
    # Create a proper bash script for EBS setup
    # Target the 100GB checkpoint volume (not instance store or root volume)
    command_script = f"""
export AWS_DEFAULT_REGION=us-east-1
sleep 5
echo '=== Block Devices ==='
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT

# Find the 100GB checkpoint volume (exclude root nvme0n1 and instance store)
# Look for 100G or ~100G volume that's not mounted
DEVICE=$(lsblk -o NAME,SIZE,TYPE,MOUNTPOINT -d -n | grep -E '100G|100.G' | grep 'disk' | grep -v nvme0n1 | awk '{{print "/dev/"$1}}' | head -1)

if [ -z "$DEVICE" ]; then
  echo "ERROR: Could not find 100GB checkpoint volume"
  exit 1
fi

echo "Found checkpoint device: $DEVICE"

# Check if it has LVM or filesystem
echo '=== Checking filesystem ==='
sudo file -s $DEVICE

# If it's an LVM volume, wipe it and create fresh ext4
if sudo file -s $DEVICE | grep -q 'LVM2_member'; then
  echo 'Removing LVM configuration and formatting fresh...'
  sudo wipefs -a $DEVICE
  sudo mkfs.ext4 -F $DEVICE
elif sudo file -s $DEVICE | grep -q 'data$'; then
  echo 'No filesystem found, formatting...'
  sudo mkfs.ext4 -F $DEVICE
else
  echo 'Filesystem already exists'
fi

sudo mkdir -p {mount_path}
sudo mount $DEVICE {mount_path}
sudo chmod 777 {mount_path}

echo '=== Mounted filesystems ==='
df -h | grep -E '(Filesystem|/mnt)'
echo "EBS checkpoint volume ($DEVICE) mounted at {mount_path}"
"""
    
    try:
        logger.info("Sending EBS setup commands via SSM...")
        command_id = ssm_manager.send_command(
            instance_id=instance_id,
            commands=[command_script],
            comment="Setup EBS checkpoint volume"
        )
        
        logger.info(f"Command ID: {command_id}")
        logger.info("Waiting for command to complete...")
        
        # Wait for command completion
        output = ssm_manager.wait_for_command(command_id, instance_id, timeout=180)
        
        if output['status'] == 'Success':
            logger.success("EBS volume setup completed successfully!")
            logger.info("\nCommand Output:")
            logger.info(output['stdout'])
            
            if output['stderr']:
                logger.warning("\nStderr:")
                logger.warning(output['stderr'])
            
            return True
        else:
            logger.error(f"Command failed with status: {output['status']}")
            logger.error(f"Stdout: {output['stdout']}")
            logger.error(f"Stderr: {output['stderr']}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to setup EBS volume: {e}")
        return False


def pull_docker_image(ssm_manager: SSMManager, instance_id: str, ecr_registry: str, repository: str):
    """
    Pull Docker image from ECR via SSM.
    
    Args:
        ssm_manager: SSMManager instance
        instance_id: EC2 instance ID
        ecr_registry: ECR registry URL
        repository: ECR repository name
        
    Returns:
        True if successful
    """
    logger.info(f"Pulling Docker image from ECR: {ecr_registry}/{repository}:latest")
    
    commands = [
        # Set AWS region
        "export AWS_DEFAULT_REGION=us-east-1",
        
        # Login to ECR
        f"aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin {ecr_registry}",
        
        # Pull the image
        f"docker pull {ecr_registry}/{repository}:latest",
        
        # Verify image
        f"docker images | grep {repository}",
        
        # Test GPU access
        "echo '=== Testing GPU Access ==='",
        "docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi || echo 'GPU test with nvidia/cuda failed'",
        
        "echo 'Docker image ready!'"
    ]
    
    command_script = " && ".join(commands)
    
    try:
        logger.info("Sending Docker pull commands via SSM...")
        command_id = ssm_manager.send_command(
            instance_id=instance_id,
            commands=[command_script],
            comment="Pull Docker image from ECR"
        )
        
        logger.info(f"Command ID: {command_id}")
        logger.info("Waiting for Docker pull to complete (this may take several minutes)...")
        
        # Docker pull can take a while (20+ GB image)
        output = ssm_manager.wait_for_command(command_id, instance_id, timeout=600)
        
        if output['status'] == 'Success':
            logger.success("Docker image pulled successfully!")
            logger.info("\nCommand Output:")
            logger.info(output['stdout'])
            
            if output['stderr']:
                logger.debug("\nStderr (may include Docker warnings):")
                logger.debug(output['stderr'])
            
            return True
        else:
            logger.error(f"Command failed with status: {output['status']}")
            logger.error(f"Stdout: {output['stdout']}")
            logger.error(f"Stderr: {output['stderr']}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to pull Docker image: {e}")
        return False


def main():
    """Main execution function."""
    args = parse_args()
    
    try:
        logger.info("=" * 60)
        logger.info("Deploying Training Environment to EC2")
        logger.info("=" * 60)
        
        # Load configurations
        logger.info(f"Loading configurations from '{args.config_dir}'...")
        configs = load_all_configs(args.config_dir, use_ssm=True)
        
        # Get configuration values
        instance_id = configs.get_aws("aws.ec2.instance_id")
        volume_id = configs.get_aws("aws.ebs.volume_id")
        mount_path = configs.get_aws("aws.ebs.mount_path")
        region = configs.get_aws("aws.region")
        ecr_registry = configs.get_aws("aws.ecr.registry")
        ecr_repository = configs.get_aws("aws.ecr.repository")
        
        logger.info(f"Instance ID: {instance_id}")
        logger.info(f"Volume ID: {volume_id}")
        logger.info(f"Mount Path: {mount_path}")
        logger.info(f"Region: {region}")
        logger.info(f"ECR: {ecr_registry}/{ecr_repository}")
        
        # Initialize AWS client and managers
        aws_client = AWSClient(region=region)
        ec2_manager = EC2Manager(client=aws_client)
        ssm_manager = SSMManager(client=aws_client)
        
        # Verify instance is running
        logger.info("\nVerifying instance state...")
        status_info = ec2_manager.get_instance_status(instance_id)
        instance_state = status_info['state']
        
        if instance_state != "running":
            logger.error(f"Instance is not running (state: {instance_state})")
            logger.error("Please run: python scripts/setup/start_ec2.py")
            return 1
        
        # Verify SSM connectivity
        logger.info("Verifying SSM connectivity...")
        try:
            response = aws_client.ssm.describe_instance_information(
                Filters=[{'Key': 'InstanceIds', 'Values': [instance_id]}]
            )
            
            if response['InstanceInformationList']:
                ssm_status = response['InstanceInformationList'][0]['PingStatus']
            else:
                ssm_status = "Not Registered"
        except Exception:
            ssm_status = "Unknown"
        
        if ssm_status != "Online":
            logger.error(f"SSM agent is not online (status: {ssm_status})")
            logger.error("Wait for SSM to come online or restart the instance")
            return 1
        
        logger.success("Instance is running and SSM is online!")
        
        # Step 1: Attach and setup EBS volume
        if not args.skip_volume:
            logger.info("\n" + "=" * 60)
            logger.info("Step 1: Setting up EBS Checkpoint Volume")
            logger.info("=" * 60)
            
            # Attach volume if needed
            if not attach_ebs_volume(ec2_manager, instance_id, volume_id):
                logger.error("Failed to attach EBS volume")
                return 1
            
            # Wait a moment for device to be recognized
            time.sleep(5)
            
            # Format and mount volume
            if not setup_ebs_volume(ssm_manager, instance_id, mount_path):
                logger.error("Failed to setup EBS volume")
                return 1
        else:
            logger.info("\n⏭️  Skipping EBS volume setup (--skip-volume)")
        
        # Step 2: Pull Docker image
        if not args.skip_docker:
            logger.info("\n" + "=" * 60)
            logger.info("Step 2: Pulling Docker Image from ECR")
            logger.info("=" * 60)
            
            if not pull_docker_image(ssm_manager, instance_id, ecr_registry, ecr_repository):
                logger.error("Failed to pull Docker image")
                return 1
        else:
            logger.info("\n⏭️  Skipping Docker image pull (--skip-docker)")
        
        # Success!
        logger.success("\n" + "=" * 60)
        logger.success("✅ Deployment Complete!")
        logger.success("=" * 60)
        logger.info("\nEC2 instance is ready for training!")
        logger.info("\nNext steps:")
        logger.info("  1. Run training: python scripts/finetune/run_training.py")
        logger.info("  2. Monitor progress via CloudWatch Logs")
        logger.info("  3. After training: python scripts/setup/stop_ec2.py")
        logger.info("\nResources:")
        logger.info(f"  - Checkpoint volume: {mount_path} (100 GB)")
        logger.info(f"  - Docker image: {ecr_registry}/{ecr_repository}:latest")
        logger.info("  - GPU: NVIDIA L4 (24 GB VRAM)")
        
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
