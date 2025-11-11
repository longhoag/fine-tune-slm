#!/usr/bin/env python3
"""
Push trained model from S3 to Hugging Face Hub.

This script:
1. Lists available trained models in S3 (timestamped versions)
2. Downloads selected model from S3 to local temp directory
3. Authenticates with Hugging Face using token from Secrets Manager
4. Pushes model to HuggingFace Hub
5. Cleans up temporary files

Note: Models are automatically uploaded to S3 during training with timestamps.
      This script runs locally - no EC2 instance required!

Usage:
    # List available models in S3
    poetry run python scripts/finetune/push_to_hf.py --list
    
    # Push latest model to HuggingFace (default)
    poetry run python scripts/finetune/push_to_hf.py
    
    # Push specific timestamped version
    poetry run python scripts/finetune/push_to_hf.py --timestamp 20251110_174835
    
    # Dry run to verify configuration
    poetry run python scripts/finetune/push_to_hf.py --dry-run
"""

import argparse
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.utils.config import load_all_configs
from src.utils.aws_helpers import AWSClient, S3Manager, SecretsManager


def list_s3_models(s3_manager: S3Manager, bucket: str, prefix: str) -> list:
    """
    List available trained models in S3.
    
    Args:
        s3_manager: S3Manager instance
        bucket: S3 bucket name
        prefix: S3 prefix (e.g., 'models/llama-3.1-8b-medical-ie')
        
    Returns:
        List of dicts with timestamp and path info
    """
    logger.info(f"Listing models in s3://{bucket}/{prefix}/")
    
    try:
        # List all objects under prefix
        response = s3_manager.client.s3.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{prefix}/",
            Delimiter='/'
        )
        
        # Extract timestamp directories
        models = []
        if 'CommonPrefixes' in response:
            for prefix_info in response['CommonPrefixes']:
                full_prefix = prefix_info['Prefix']
                # Extract timestamp from path: models/llama/.../20251110_174835/
                parts = full_prefix.rstrip('/').split('/')
                timestamp = parts[-1]
                
                # Verify it looks like a timestamp
                if '_' in timestamp and len(timestamp) == 15:
                    try:
                        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                        models.append({
                            'timestamp': timestamp,
                            'datetime': dt,
                            's3_prefix': full_prefix.rstrip('/'),
                            'formatted_date': dt.strftime("%Y-%m-%d %H:%M:%S")
                        })
                    except ValueError:
                        # Not a valid timestamp, skip
                        pass
        
        # Sort by datetime (newest first)
        models.sort(key=lambda x: x['datetime'], reverse=True)
        
        return models
        
    except Exception as e:
        logger.error(f"Failed to list S3 models: {e}")
        return []


def download_model_from_s3(
    s3_manager: S3Manager,
    bucket: str,
    s3_prefix: str,
    local_dir: Path
) -> bool:
    """
    Download model from S3 to local directory.
    
    Args:
        s3_manager: S3Manager instance
        bucket: S3 bucket name
        s3_prefix: S3 prefix to model directory
        local_dir: Local directory to download to
        
    Returns:
        True if successful
    """
    logger.info(f"Downloading from s3://{bucket}/{s3_prefix}/final_model")
    logger.info(f"To local directory: {local_dir}")
    
    try:
        # Download all files under the model prefix
        s3_model_prefix = f"{s3_prefix}/final_model"
        
        # List all files
        response = s3_manager.client.s3.list_objects_v2(
            Bucket=bucket,
            Prefix=s3_model_prefix
        )
        
        if 'Contents' not in response:
            logger.error("No files found in S3")
            return False
        
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Download each file
        for obj in response['Contents']:
            s3_key = obj['Key']
            # Get relative path within the model directory
            rel_path = s3_key.replace(s3_model_prefix + '/', '')
            
            if not rel_path or rel_path == s3_model_prefix:
                continue
                
            local_file = local_dir / rel_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"Downloading: {rel_path}")
            s3_manager.client.s3.download_file(bucket, s3_key, str(local_file))
        
        logger.success(f"✅ Downloaded {len(response['Contents'])} files")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False


def push_to_huggingface(
    model_dir: Path,
    repo_name: str,
    hf_token: str,
    dry_run: bool = False
) -> bool:
    """
    Push model to HuggingFace Hub.
    
    Args:
        model_dir: Local directory containing the model
        repo_name: HuggingFace repository (e.g., 'username/model-name')
        hf_token: HuggingFace API token
        dry_run: If True, only verify without pushing
        
    Returns:
        True if successful
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Pushing to HuggingFace Hub: {repo_name}")
    logger.info(f"{'='*60}\n")
    
    if dry_run:
        logger.warning("DRY RUN - Verifying model files only")
        
        # Check required files exist
        required_files = ['adapter_config.json', 'adapter_model.safetensors']
        missing = []
        for file in required_files:
            if not (model_dir / file).exists():
                missing.append(file)
        
        if missing:
            logger.error(f"Missing required files: {missing}")
            return False
        
        logger.success("✅ Model files verified")
        logger.info(f"Would push to: https://huggingface.co/{repo_name}")
        return True
    
    try:
        from huggingface_hub import HfApi, login
        
        # Login to HuggingFace
        logger.info("Logging in to HuggingFace Hub...")
        login(token=hf_token, add_to_git_credential=False)
        logger.success("✅ Logged in to HuggingFace Hub")
        
        # Initialize API
        api = HfApi()
        
        # Check if repo exists, create if not
        try:
            api.repo_info(repo_id=repo_name, repo_type="model")
            logger.info(f"Repository exists: {repo_name}")
        except Exception:
            logger.info(f"Creating repository: {repo_name}")
            api.create_repo(repo_id=repo_name, repo_type="model", private=False)
        
        # Upload files
        logger.info("Uploading model files...")
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_name,
            repo_type="model",
            commit_message=f"Upload model from training run {timestamp_str}"
        )
        
        logger.success(f"✅ Pushed to: https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to push to HuggingFace: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Push trained model from S3 to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models in S3
  poetry run python scripts/finetune/push_to_hf.py --list
  
  # Push latest model to HuggingFace (default)
  poetry run python scripts/finetune/push_to_hf.py
  
  # Push specific timestamped version
  poetry run python scripts/finetune/push_to_hf.py --timestamp 20251110_174835

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
        "--list",
        action="store_true",
        help="List available models in S3 and exit"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        help="Specific timestamp to push (e.g., '20251110_174835')"
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
        configs = load_all_configs(args.config_dir, use_ssm=True)
        
        region = configs.get_aws('aws.region')
        s3_bucket = configs.get_training('output.s3_bucket')
        s3_prefix = configs.get_training('output.s3_prefix')
        hf_repo = configs.get_training('output.hf_repo')
        hf_token_secret = configs.get_aws('aws.secrets_manager.hf_token_secret')
        
        logger.info(f"S3 Location: s3://{s3_bucket}/{s3_prefix}")
        logger.info(f"HuggingFace Repo: {hf_repo}")
        
        # Initialize AWS clients
        aws_client = AWSClient(region=region)
        s3_manager = S3Manager(aws_client)
        secrets_mgr = SecretsManager(aws_client)
        
        # List available models
        logger.info("\n📦 Checking available models in S3...")
        models = list_s3_models(s3_manager, s3_bucket, s3_prefix)
        
        if not models:
            logger.error("No trained models found in S3")
            logger.info(f"Expected: s3://{s3_bucket}/{s3_prefix}/YYYYMMDD_HHMMSS/")
            logger.info("\nHave you run training yet?")
            logger.info("  poetry run python scripts/finetune/run_training.py")
            sys.exit(1)
        
        logger.success(f"Found {len(models)} trained model(s):")
        for i, model in enumerate(models, 1):
            marker = "← LATEST" if i == 1 else ""
            timestamp = model['timestamp']
            formatted = model['formatted_date']
            logger.info(f"  {i}. {timestamp} ({formatted}) {marker}")
        
        # If --list only, exit
        if args.list:
            logger.info("\nUse --timestamp to push a specific version")
            sys.exit(0)
        
        # Select model to push
        if args.timestamp:
            # Find specified timestamp
            selected = None
            for model in models:
                if model['timestamp'] == args.timestamp:
                    selected = model
                    break
            
            if not selected:
                logger.error(f"Timestamp not found: {args.timestamp}")
                logger.info("Available timestamps:")
                for model in models:
                    logger.info(f"  - {model['timestamp']}")
                sys.exit(1)
            
            timestamp = selected['timestamp']
            formatted = selected['formatted_date']
            logger.info(f"\n✅ Selected: {timestamp} ({formatted})")
        else:
            # Use latest
            selected = models[0]
            timestamp = selected['timestamp']
            formatted = selected['formatted_date']
            logger.info(f"\n✅ Using latest: {timestamp} ({formatted})")
        
        # Create temporary directory for download
        temp_dir = Path(tempfile.mkdtemp(prefix="hf_model_"))
        logger.info(f"\n📥 Temporary directory: {temp_dir}")
        
        try:
            # Download model from S3
            success = download_model_from_s3(
                s3_manager=s3_manager,
                bucket=s3_bucket,
                s3_prefix=selected['s3_prefix'],
                local_dir=temp_dir
            )
            
            if not success:
                logger.error("Failed to download model from S3")
                sys.exit(1)
            
            # Get HuggingFace token
            logger.info("\n🔑 Retrieving HuggingFace token...")
            hf_token = secrets_mgr.get_secret(hf_token_secret)
            logger.success("✅ Retrieved HuggingFace token")
            
            # Push to HuggingFace
            logger.info("\n🚀 Pushing to HuggingFace Hub...")
            success = push_to_huggingface(
                model_dir=temp_dir,
                repo_name=hf_repo,
                hf_token=hf_token,
                dry_run=args.dry_run
            )
            
            if not success:
                logger.error("Failed to push to HuggingFace Hub")
                sys.exit(1)
            
            logger.success("\n✅ All operations completed successfully!")
            if not args.dry_run:
                logger.info(f"\n🎉 Live at: https://huggingface.co/{hf_repo}")
                s3_path = f"{s3_bucket}/{selected['s3_prefix']}/final_model"
                logger.info(f"📦 S3 backup: s3://{s3_path}")
            
        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                logger.info(f"\n🧹 Cleaning up...")
                shutil.rmtree(temp_dir)
                logger.success("✅ Cleanup complete")
        
        sys.exit(0)
        
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
