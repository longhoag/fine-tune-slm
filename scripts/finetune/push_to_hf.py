#!/usr/bin/env python3
"""
Push trained model to Hugging Face Hub.

This script:
1. Verifies trained model exists on EC2 or S3
2. Merges LoRA weights with base model (optional)
3. Authenticates with Hugging Face
4. Pushes model to Hub
5. Updates model card with training details

Usage:
    # Push model from EC2 checkpoint directory
    poetry run python scripts/finetune/push_to_hf.py
    
    # Push specific checkpoint
    poetry run python scripts/finetune/push_to_hf.py --checkpoint checkpoint-1000
    
    # Push from S3 (if already copied)
    poetry run python scripts/finetune/push_to_hf.py --from-s3

    # Dry run to verify configuration
    poetry run python scripts/finetune/push_to_hf.py --dry-run
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.utils.config import load_all_configs
from src.utils.aws_helpers import AWSClient, EC2Manager, SSMManager, S3Manager, SecretsManager


def verify_ec2_checkpoint(
    ssm_manager: SSMManager,
    instance_id: str,
    checkpoint_dir: str = "/mnt/training/checkpoints"
) -> dict:
    """
    Verify trained model exists on EC2 instance.
    
    Args:
        ssm_manager: SSMManager instance
        instance_id: EC2 instance ID
        checkpoint_dir: Checkpoint directory path
        
    Returns:
        dict with verification results
    """
    logger.info(f"Verifying checkpoint directory on EC2: {checkpoint_dir}")
    
    check_cmd = f"""
export AWS_DEFAULT_REGION=us-east-1
if [ -d "{checkpoint_dir}" ]; then
    echo "Directory exists"
    echo "=== Contents ==="
    ls -lh {checkpoint_dir}
    echo "=== Checkpoint subdirectories ==="
    ls -d {checkpoint_dir}/checkpoint-* 2>/dev/null || echo "No checkpoints found"
    echo "=== Final model ==="
    ls -lh {checkpoint_dir}/adapter_* 2>/dev/null || echo "No adapter files found"
else
    echo "Directory does not exist"
    exit 1
fi
"""
    
    command_id = ssm_manager.send_command(
        instance_id=instance_id,
        commands=[check_cmd],
        comment="Verify checkpoint directory"
    )
    
    output = ssm_manager.wait_for_command(command_id, instance_id, timeout=60)
    
    if output['status'] == 'Success':
        logger.success("Checkpoint directory verified!")
        logger.info(f"\n{output['stdout']}")
        return {"exists": True, "output": output['stdout']}
    else:
        logger.error("Checkpoint verification failed")
        logger.error(f"\n{output['stderr']}")
        return {"exists": False, "error": output['stderr']}


def push_from_ec2(
    ssm_manager: SSMManager,
    instance_id: str,
    ecr_registry: str,
    repository: str,
    hf_repo: str,
    checkpoint_path: str = "/mnt/training/checkpoints",
    dry_run: bool = False
) -> dict:
    """
    Push model to HuggingFace Hub from EC2 instance.
    
    Args:
        ssm_manager: SSMManager instance
        instance_id: EC2 instance ID
        ecr_registry: ECR registry URL
        repository: ECR repository name
        hf_repo: HuggingFace repository name
        checkpoint_path: Path to checkpoint directory
        dry_run: If True, only verify without pushing
        
    Returns:
        Command output dict
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Pushing Model to HuggingFace Hub: {hf_repo}")
    logger.info(f"{'='*60}\n")
    
    # Build push command using Docker container
    docker_cmd_parts = [
        "docker run --rm --gpus all",
        "-v /mnt/training:/mnt/training",
        "-v /home/ubuntu/fine-tune-slm:/workspace",
        "-w /workspace",
        f"{ecr_registry}/{repository}:latest",
        "python scripts/push_model_to_hub.py",  # Script inside container
        f"--checkpoint-dir {checkpoint_path}",
        f"--repo-name {hf_repo}",
        "--use-ssm",  # Get HF token from Secrets Manager
    ]
    
    if dry_run:
        docker_cmd_parts.append("--dry-run")
    
    push_cmd = " ".join(docker_cmd_parts)
    
    logger.info("Push command:")
    logger.info(push_cmd)
    logger.info("")
    
    if dry_run:
        logger.info("DRY RUN - Would execute push command")
        return {"status": "DryRun", "command": push_cmd}
    
    # Send command
    logger.info("Sending push command via SSM...")
    command_id = ssm_manager.send_command(
        instance_id=instance_id,
        commands=[f"export AWS_DEFAULT_REGION=us-east-1 && cd /home/ubuntu/fine-tune-slm && {push_cmd}"],
        comment=f"Push model to HF Hub: {hf_repo}",
        timeout=1800  # 30 minutes
    )
    
    logger.success(f"Command sent! Command ID: {command_id}")
    logger.info("Waiting for push to complete...")
    
    output = ssm_manager.wait_for_command(
        command_id=command_id,
        instance_id=instance_id,
        timeout=1800,
        poll_interval=10
    )
    
    if output['status'] == 'Success':
        logger.success("✅ Model pushed to HuggingFace Hub successfully!")
        logger.info(f"\nOutput:\n{output['stdout']}")
    else:
        logger.error(f"❌ Push failed with status: {output['status']}")
        logger.error(f"\nStdout:\n{output['stdout']}")
        logger.error(f"\nStderr:\n{output['stderr']}")
    
    return output


def copy_checkpoint_to_s3(
    ssm_manager: SSMManager,
    instance_id: str,
    checkpoint_path: str,
    s3_bucket: str,
    s3_prefix: str
) -> dict:
    """
    Copy checkpoint from EC2 to S3.
    
    Args:
        ssm_manager: SSMManager instance
        instance_id: EC2 instance ID
        checkpoint_path: Source path on EC2
        s3_bucket: Destination S3 bucket
        s3_prefix: Destination S3 prefix
        
    Returns:
        Command output dict
    """
    logger.info(f"Copying checkpoint to S3: s3://{s3_bucket}/{s3_prefix}")
    
    copy_cmd = f"""
export AWS_DEFAULT_REGION=us-east-1
if [ -d "{checkpoint_path}" ]; then
    echo "Copying to S3..."
    aws s3 sync {checkpoint_path} s3://{s3_bucket}/{s3_prefix} --no-progress
    echo "Copy completed"
else
    echo "Checkpoint directory not found: {checkpoint_path}"
    exit 1
fi
"""
    
    command_id = ssm_manager.send_command(
        instance_id=instance_id,
        commands=[copy_cmd],
        comment="Copy checkpoint to S3",
        timeout=1800
    )
    
    logger.info("Waiting for S3 copy to complete...")
    output = ssm_manager.wait_for_command(command_id, instance_id, timeout=1800, poll_interval=10)
    
    if output['status'] == 'Success':
        logger.success(f"✅ Checkpoint copied to s3://{s3_bucket}/{s3_prefix}")
    else:
        logger.error("❌ S3 copy failed")
        logger.error(f"\n{output['stderr']}")
    
    return output


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Push trained model to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Push latest checkpoint from EC2
  poetry run python scripts/finetune/push_to_hf.py
  
  # Push specific checkpoint
  poetry run python scripts/finetune/push_to_hf.py --checkpoint checkpoint-1000
  
  # Copy to S3 only (don't push to HF)
  poetry run python scripts/finetune/push_to_hf.py --copy-to-s3-only

  # Dry run to verify configuration
  poetry run python scripts/finetune/push_to_hf.py --dry-run
        """
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Configuration directory (default: config/)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Specific checkpoint to push (e.g., 'checkpoint-1000')"
    )
    parser.add_argument(
        "--copy-to-s3-only",
        action="store_true",
        help="Only copy to S3, don't push to HuggingFace"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify configuration without pushing"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Push Model to HuggingFace Hub")
    logger.info("="*60)
    logger.info(f"Loading configurations from '{args.config_dir}'...")
    
    try:
        # Load configs
        configs = load_all_configs(args.config_dir)
        
        instance_id = configs.get_aws('aws.ec2.instance_id')
        region = configs.get_aws('aws.region')
        ecr_registry = configs.get_aws('aws.ecr.registry')
        repository = configs.get_aws('aws.ecr.repository')
        s3_bucket = configs.get_aws('aws.s3.bucket')
        s3_prefix = configs.get_aws('aws.s3.prefix')
        hf_repo = configs.get_training('output.hf_repo')
        
        logger.info(f"Instance ID: {instance_id}")
        logger.info(f"Region: {region}")
        logger.info(f"HuggingFace Repo: {hf_repo}")
        logger.info(f"S3 Backup: s3://{s3_bucket}/{s3_prefix}")
        
        # Initialize AWS clients
        aws_client = AWSClient(region=region)
        ec2_manager = EC2Manager(aws_client)
        ssm_manager = SSMManager(aws_client)
        
        # Check instance is running
        status_info = ec2_manager.get_instance_status(instance_id)
        if status_info['state'] != "running":
            logger.error(f"Instance is not running. Current state: {status_info['state']}")
            logger.info("Start the instance first: poetry run python scripts/setup/start_ec2.py")
            sys.exit(1)
        
        # Verify checkpoint exists
        checkpoint_path = "/mnt/training/checkpoints"
        if args.checkpoint:
            checkpoint_path = f"/mnt/training/checkpoints/{args.checkpoint}"
        
        result = verify_ec2_checkpoint(ssm_manager, instance_id, checkpoint_path)
        if not result['exists']:
            logger.error("No checkpoint found on EC2 instance")
            sys.exit(1)
        
        # Copy to S3
        logger.info("\n📦 Copying checkpoint to S3 for archival...")
        copy_result = copy_checkpoint_to_s3(
            ssm_manager=ssm_manager,
            instance_id=instance_id,
            checkpoint_path=checkpoint_path,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix
        )
        
        if copy_result['status'] != 'Success':
            logger.error("Failed to copy checkpoint to S3")
            sys.exit(1)
        
        # Push to HuggingFace (unless --copy-to-s3-only)
        if not args.copy_to_s3_only:
            logger.info("\n🚀 Pushing to HuggingFace Hub...")
            push_result = push_from_ec2(
                ssm_manager=ssm_manager,
                instance_id=instance_id,
                ecr_registry=ecr_registry,
                repository=repository,
                hf_repo=hf_repo,
                checkpoint_path=checkpoint_path,
                dry_run=args.dry_run
            )
            
            if push_result['status'] not in ['Success', 'DryRun']:
                logger.error("Failed to push model to HuggingFace Hub")
                sys.exit(1)
        
        logger.success("\n✅ All operations completed successfully!")
        logger.info("\nNext steps:")
        logger.info(f"  1. View model on HuggingFace: https://huggingface.co/{hf_repo}")
        logger.info(f"  2. View S3 backup: s3://{s3_bucket}/{s3_prefix}")
        logger.info("  3. Stop instance: python scripts/setup/stop_ec2.py")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
