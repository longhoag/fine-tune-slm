#!/usr/bin/env python3
"""
Run model testing on EC2 instance via SSM.

This script orchestrates model testing by sending Docker commands to EC2.
The actual testing logic is in src/test_model.py (runs in Docker container).

Prerequisites:
- EC2 instance must be running: poetry run python scripts/setup/start_ec2.py
- Environment must be deployed: poetry run python scripts/setup/deploy_via_ssm.py

Usage:
    # Test latest model with sample medical text
    poetry run python scripts/finetune/run_test_model_ec2.py
    
    # Test specific timestamped model
    poetry run python scripts/finetune/run_test_model_ec2.py --timestamp 20251111_022951
    
    # Test with custom input
    poetry run python scripts/finetune/run_test_model_ec2.py --input "Your medical text here"
    
    # Test different sample (0, 1, or 2)
    poetry run python scripts/finetune/run_test_model_ec2.py --sample-index 1
    
    # List available models
    poetry run python scripts/finetune/run_test_model_ec2.py --list
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.utils.config import load_all_configs
from src.utils.aws_helpers import AWSClient, SSMManager, S3Manager


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run model testing on EC2 via SSM",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Configuration directory"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        help="Specific model timestamp to test (e.g., '20251111_022951')"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Custom medical text to test (overrides --sample-index)"
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Which sample text to use (0-2, default: 0)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Run Model Testing on EC2")
    logger.info("="*60)
    
    try:
        # Load configs
        configs = load_all_configs(args.config_dir, use_ssm=True)
        
        region = configs.get_aws('aws.region')
        instance_id = configs.get_aws('aws.ec2.instance_id')
        s3_bucket = configs.get_training('output.s3_bucket')
        s3_prefix = configs.get_training('output.s3_prefix')
        
        # Initialize AWS clients
        aws_client = AWSClient(region=region)
        ssm_manager = SSMManager(aws_client)
        s3_manager = S3Manager(aws_client)
        
        # Verify EC2 instance is running
        logger.info(f"\n🔍 Checking EC2 instance: {instance_id}")
        
        try:
            instance_state = aws_client.ec2.describe_instances(
                InstanceIds=[instance_id]
            )['Reservations'][0]['Instances'][0]['State']['Name']
            
            logger.info(f"Instance state: {instance_state}")
            
            if instance_state != 'running':
                logger.error(f"❌ EC2 instance is '{instance_state}', not running")
                logger.info("\nPlease start the instance first:")
                logger.info("  poetry run python scripts/setup/start_ec2.py")
                sys.exit(1)
            
            logger.success("✅ Instance is running")
            
        except Exception as e:
            logger.error(f"❌ Failed to check instance state: {e}")
            logger.info("\nMake sure the instance exists and you have permissions")
            sys.exit(1)
        
        # List available models
        logger.info("\n📦 Finding trained models in S3...")
        
        from scripts.finetune.push_to_hf import list_s3_models
        models = list_s3_models(s3_manager, s3_bucket, s3_prefix)
        
        if not models:
            logger.error("No trained models found in S3")
            logger.info(f"Expected: s3://{s3_bucket}/{s3_prefix}/YYYYMMDD_HHMMSS/")
            sys.exit(1)
        
        logger.success(f"Found {len(models)} trained model(s):")
        for i, model in enumerate(models, 1):
            marker = "← LATEST" if i == 1 else ""
            logger.info(f"  {i}. {model['timestamp']} ({model['formatted_date']}) {marker}")
        
        if args.list:
            sys.exit(0)
        
        # Select model
        if args.timestamp:
            selected = None
            for model in models:
                if model['timestamp'] == args.timestamp:
                    selected = model
                    break
            
            if not selected:
                logger.error(f"Timestamp not found: {args.timestamp}")
                sys.exit(1)
        else:
            selected = models[0]
            logger.info(f"\n✅ Using latest: {selected['timestamp']}")
        
        # Get test input
        if args.input:
            test_input = args.input
            logger.info("\n📝 Using custom input")
        else:
            test_input = None
            logger.info(f"\n📝 Using sample text #{args.sample_index + 1}")
        
        # Get ECR image URI
        ecr_registry = configs.get_aws('aws.ecr.registry')
        ecr_repository = configs.get_aws('aws.ecr.repository')
        ecr_image = f"{ecr_registry}/{ecr_repository}:latest"
        
        # Build Docker command
        logger.info("\n🐳 Preparing Docker command...")
        logger.info(f"ECR Image: {ecr_image}")
        
        # Build Docker command (following run_training.py pattern)
        # Override entrypoint since Dockerfile has ENTRYPOINT ["python3"]
        # Note: Code is already in Docker image at /workspace (set by Dockerfile WORKDIR)
        docker_cmd_parts = [
            "docker run --rm --gpus all",
            "--entrypoint python3",
            "-v /mnt/training:/mnt/training",  # Mount training volume for model download
            ecr_image,
        ]
        
        # Python module command (without 'python' since entrypoint is python3)
        python_cmd_parts = [
            "-m src.test_model",
            "--use-ssm",
            f"--timestamp {selected['timestamp']}",
            f"--sample-index {args.sample_index}"
        ]
        
        # Add custom input if provided
        if test_input:
            # Escape input for shell
            escaped_input = test_input.replace('"', '\\"').replace('$', '\\$')
            python_cmd_parts.append(f'--input "{escaped_input}"')
        
        # Combine Docker and Python commands
        docker_cmd = " ".join(docker_cmd_parts) + " " + " ".join(python_cmd_parts)
        
        logger.success("✅ Docker command ready")
        
        # Send SSM command (no bash -c wrapper needed - following run_training.py pattern)
        logger.info("\n📤 Sending command to EC2...")
        
        commands = [docker_cmd]
        
        logger.info("\n🤖 Running model test on EC2...")
        logger.info("="*60)
        
        command_id = ssm_manager.send_command(
            instance_id=instance_id,
            commands=commands,
            comment=f"Test model {selected['timestamp']}",
            working_directory="/home/ubuntu"  # Don't use /workspace to avoid path conflicts
        )
        
        logger.info(f"SSM Command ID: {command_id}")
        logger.info("Waiting for test to complete (this may take a few minutes)...")
        logger.info("="*60)
        logger.info("")
        
        # Wait and stream output
        result = ssm_manager.wait_for_command(command_id, instance_id)
        
        # Display output
        logger.info("\n" + "="*60)
        logger.info(f"TEST COMPLETED - Status: {result['status']}")
        logger.info("="*60)
        
        # All output is in stderr since we use logger.info() in test_model.py
        stderr = result.get('stderr', '').strip()
        
        if stderr:
            # Display the model test output
            print("\n" + stderr + "\n")
        else:
            logger.warning("\n⚠️  No output captured")
            logger.info("\n💡 Check SSM command logs:")
            logger.info(f"    aws ssm get-command-invocation --command-id {command_id} --instance-id {instance_id} --region us-east-1")
        
        # Final status
        if result['status'] == 'Success':
            logger.success("\n✅ Test completed successfully!")
        else:
            logger.error(f"\n❌ Test failed with status: {result['status']}")
            logger.info(f"Exit code: {result.get('exit_code', 'unknown')}")
            logger.info("\n💡 View error details:")
            logger.info(f"    aws ssm get-command-invocation --command-id {command_id} --instance-id {instance_id} --region us-east-1 --query StandardErrorContent --output text")
        
        logger.info("\n💡 To stop the EC2 instance:")
        logger.info("    poetry run python scripts/setup/stop_ec2.py")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
