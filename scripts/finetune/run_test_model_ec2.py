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
        ecr_image = configs.get_aws('aws.ecr.image_uri')
        
        # Build Docker command
        logger.info("\n🐳 Preparing Docker command...")
        
        docker_cmd = (
            f"docker run --rm --gpus all "
            f"-v /mnt/training:/mnt/training "
            f"-v /home/ubuntu/fine-tune-slm:/workspace "
            f"-w /workspace "
            f"{ecr_image} "
            f"python -m src.test_model "
            f"--use-ssm "
            f"--timestamp {selected['timestamp']} "
            f"--sample-index {args.sample_index}"
        )
        
        # Add custom input if provided
        if test_input:
            # Escape input for shell
            escaped_input = test_input.replace('"', '\\"').replace('$', '\\$')
            docker_cmd += f' --input "{escaped_input}"'
        
        logger.success("✅ Docker command ready")
        
        # Send SSM command
        logger.info("\n📤 Sending command to EC2...")
        
        commands = [docker_cmd]
        
        logger.info("\n🤖 Running model test on EC2...")
        logger.info("="*60)
        
        command_id = ssm_manager.send_command(
            instance_id=instance_id,
            commands=commands,
            comment=f"Test model {selected['timestamp']}"
        )
        
        logger.info(f"SSM Command ID: {command_id}")
        logger.info("Waiting for test to complete (this may take a few minutes)...")
        logger.info("="*60)
        logger.info("")
        
        # Wait and stream output
        result = ssm_manager.wait_for_command(command_id, instance_id)
        
        # Display output
        if result['status'] == 'Success':
            logger.info("\n" + "="*60)
            logger.info("EC2 TEST OUTPUT:")
            logger.info("="*60)
            print(result['output'])
            logger.info("="*60)
            logger.success("\n✅ Test completed successfully!")
        else:
            logger.error(f"\n❌ Test failed with status: {result['status']}")
            logger.info("\nOutput:")
            print(result['output'])
            if result.get('error'):
                logger.info("\nError:")
                print(result['error'])
        
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
