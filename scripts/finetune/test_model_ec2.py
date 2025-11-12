#!/usr/bin/env python3
"""
Test fine-tuned model on EC2 instance via SSM.

This script:
1. Lists available trained models in S3
2. Sends SSM command to download model from S3 and run inference
3. Displays results in terminal

Prerequisites:
- EC2 instance must be running: poetry run python scripts/setup/start_ec2.py
- Environment must be deployed: poetry run python scripts/setup/deploy_via_ssm.py

The actual inference runs on EC2 with GPU, results streamed to your local terminal.

Usage:
    # Test latest model with sample medical text
    poetry run python scripts/finetune/test_model_ec2.py
    
    # Test specific timestamped model
    poetry run python scripts/finetune/test_model_ec2.py --timestamp 20251111_022951
    
    # Test with custom input
    poetry run python scripts/finetune/test_model_ec2.py --input "Your medical text here"
    
    # Test different sample
    poetry run python scripts/finetune/test_model_ec2.py --sample-index 1
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


# Sample medical texts for testing
SAMPLE_MEDICAL_TEXTS = [
    """70-year-old man with widely metastatic cutaneous melanoma involving the brain, 
    liver, and bones. Biopsy showed BRAF V600E mutation. PD-L1 expression was 5% and 
    tumor mutational burden was high. Started on combination nivolumab and ipilimumab. 
    After 3 months, showed mixed response with some lesions shrinking while brain 
    metastases progressed. Underwent stereotactic radiosurgery for brain lesions.""",
    
    """62-year-old woman diagnosed with stage IIIA non-small cell lung cancer (NSCLC). 
    Molecular testing revealed EGFR exon 19 deletion. PD-L1 TPS 80%. Received osimertinib 
    as first-line therapy with excellent initial response. After 18 months, progression 
    noted with new liver metastases. Next-generation sequencing detected EGFR T790M 
    resistance mutation.""",
    
    """55-year-old male with metastatic colorectal cancer to liver and lungs. KRAS wild-type, 
    MSI-stable. Started FOLFOX plus bevacizumab. CEA levels initially 450 ng/mL, decreased 
    to 15 ng/mL after 6 cycles. CT showed partial response in liver but stable disease in 
    lungs. Continued treatment with good tolerance."""
]


def get_test_script_content(
    s3_bucket: str,
    s3_prefix: str,
    timestamp: str,
    test_input: str,
    base_model: str = "meta-llama/Meta-Llama-3.1-8B"
) -> str:
    """
    Generate the Python script to run on EC2 for testing.
    
    This script will be sent via SSM and executed on the EC2 instance.
    """
    # Escape quotes in test_input
    test_input_escaped = test_input.replace('"', '\\"').replace('\n', '\\n')
    
    script = f'''#!/usr/bin/env python3
"""Test fine-tuned model - runs on EC2."""

import json
import sys
import tempfile
from pathlib import Path

print("="*80)
print("🧪 Testing Fine-Tuned Model on EC2")
print("="*80)
print(f"Model: {s3_bucket}/{s3_prefix}/{timestamp}")
print(f"Base: {base_model}")
print()

# Install required packages if not present
print("📦 Checking dependencies...")
import subprocess
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import boto3
except ImportError as e:
    print(f"Installing missing package: {{e}}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                          "transformers", "peft", "torch", "boto3", "accelerate"])
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import boto3

print("✅ Dependencies ready")
print()

# Download model from S3
print("📥 Downloading model from S3...")
s3 = boto3.client('s3')
temp_dir = Path(tempfile.mkdtemp(prefix="model_test_"))
s3_model_prefix = f"{s3_prefix}/{timestamp}/final_model"

try:
    # List all files in S3
    response = s3.list_objects_v2(
        Bucket="{s3_bucket}",
        Prefix=s3_model_prefix
    )
    
    if 'Contents' not in response:
        print("❌ No model files found in S3")
        sys.exit(1)
    
    # Download each file
    for obj in response['Contents']:
        s3_key = obj['Key']
        rel_path = s3_key.replace(s3_model_prefix + '/', '')
        
        if not rel_path or rel_path == s3_model_prefix:
            continue
        
        local_file = temp_dir / rel_path
        local_file.parent.mkdir(parents=True, exist_ok=True)
        
        s3.download_file("{s3_bucket}", s3_key, str(local_file))
    
    print(f"✅ Downloaded {{len(response['Contents'])}} files to {{temp_dir}}")
    print()
    
    # Load model
    print("🔄 Loading base model...")
    print(f"  Model: {base_model}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "{base_model}",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print("✅ Base model loaded")
    print()
    
    print("🔄 Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, str(temp_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(temp_dir))
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ LoRA adapters loaded")
    print()
    
    # Test input
    test_text = """{test_input_escaped}"""
    
    prompt = f"""Extract all cancer-related entities from the text.

{{test_text}}

Output (JSON format):
"""
    
    print("="*80)
    print("📝 INPUT:")
    print("="*80)
    print(test_text)
    print()
    
    # Run inference
    print("="*80)
    print("🤖 RUNNING INFERENCE...")
    print("="*80)
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {{k: v.to(model.device) for k, v in inputs.items()}}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part (after prompt)
    generated = full_output[len(prompt):].strip()
    
    print("="*80)
    print("📤 OUTPUT:")
    print("="*80)
    print(generated)
    print()
    
    # Try to parse as JSON
    print("="*80)
    print("📊 PARSED OUTPUT:")
    print("="*80)
    try:
        # Find JSON block
        if "{{" in generated:
            json_start = generated.index("{{")
            json_end = generated.rindex("}}") + 2
            json_str = generated[json_start:json_end]
            parsed = json.loads(json_str)
            print(json.dumps(parsed, indent=2))
        else:
            print("⚠️  No JSON object found in output")
            print(generated)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️  Could not parse JSON: {{e}}")
        print(generated)
    
    print()
    print("="*80)
    print("✅ Test complete!")
    print("="*80)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print("🧹 Cleaned up temporary files")
    
except Exception as e:
    print(f"❌ Error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    return script


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test fine-tuned model on EC2",
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
        help="Custom medical text to test (otherwise uses sample)"
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
    logger.info("Test Fine-Tuned Model on EC2")
    logger.info("="*60)
    
    try:
        # Load configs
        configs = load_all_configs(args.config_dir, use_ssm=True)
        
        region = configs.get_aws('aws.region')
        instance_id = configs.get_aws('aws.ec2.instance_id')
        s3_bucket = configs.get_training('output.s3_bucket')
        s3_prefix = configs.get_training('output.s3_prefix')
        base_model = configs.get_training('model.base_model_id')
        
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
            test_input = SAMPLE_MEDICAL_TEXTS[args.sample_index]
            logger.info(f"\n📝 Using sample text #{args.sample_index + 1}")
        
        # Generate test script
        logger.info("\n📝 Generating test script...")
        test_script = get_test_script_content(
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            timestamp=selected['timestamp'],
            test_input=test_input,
            base_model=base_model
        )
        
        logger.success("✅ Test script ready")
        
        # Upload script to EC2 and run
        logger.info("\n📤 Uploading test script to EC2...")
        
        # Create script on EC2
        commands = [
            f"cat > /tmp/test_model.py << 'EOFSCRIPT'\\n{test_script}\\nEOFSCRIPT",
            "chmod +x /tmp/test_model.py",
            "cd /home/ubuntu",
            "python3 /tmp/test_model.py"
        ]
        
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
