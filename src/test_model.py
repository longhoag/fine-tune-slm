#!/usr/bin/env python3
"""
Test fine-tuned model - runs on EC2 in Docker container.

This module loads a trained model from S3 and runs inference with sample medical text.
Designed to run inside the Docker container on EC2 with GPU access.

Usage (inside Docker on EC2):
    python -m src.test_model --use-ssm --timestamp 20251111_022951 --sample-index 0
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import boto3
from loguru import logger

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


def download_model_from_s3(bucket: str, s3_prefix: str, timestamp: str, local_dir: Path) -> bool:
    """
    Download model from S3 to local directory.
    
    Args:
        bucket: S3 bucket name
        s3_prefix: S3 prefix base (e.g., 'models/llama-3.1-8b-medical-ie')
        timestamp: Model timestamp (e.g., '20251111_022951')
        local_dir: Local directory to download to
        
    Returns:
        True if successful
    """
    logger.info(f"📥 Downloading model from S3...")
    s3 = boto3.client('s3')
    s3_model_prefix = f"{s3_prefix}/{timestamp}/final_model"
    
    try:
        # List all files in S3
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=s3_model_prefix
        )
        
        if 'Contents' not in response:
            logger.error("❌ No model files found in S3")
            return False
        
        # Download each file
        for obj in response['Contents']:
            s3_key = obj['Key']
            rel_path = s3_key.replace(s3_model_prefix + '/', '')
            
            if not rel_path or rel_path == s3_model_prefix:
                continue
            
            local_file = local_dir / rel_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            s3.download_file(bucket, s3_key, str(local_file))
        
        logger.success(f"✅ Downloaded {len(response['Contents'])} files to {local_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        return False


def load_model(model_dir: Path, base_model_id: str):
    """
    Load base model and LoRA adapters.
    
    Args:
        model_dir: Directory containing LoRA adapters
        base_model_id: Base model identifier (e.g., 'meta-llama/Meta-Llama-3.1-8B')
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("🔄 Loading base model...")
    logger.info(f"  Model: {base_model_id}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    logger.success("✅ Base model loaded")
    
    logger.info("🔄 Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.success("✅ LoRA adapters loaded")
    
    return model, tokenizer


def run_inference(model, tokenizer, text: str) -> str:
    """
    Run inference on medical text.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        text: Input medical text
        
    Returns:
        Generated output text
    """
    prompt = f"""Extract all cancer-related entities from the text.

{text}

Output (JSON format):
"""
    
    logger.info("🤖 Running inference...")
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
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
    
    return generated


def parse_output(output: str) -> dict:
    """
    Parse model output as JSON.
    
    Args:
        output: Generated text output
        
    Returns:
        Parsed dictionary or None if parsing fails
    """
    try:
        # Find JSON block
        if "{" in output:
            json_start = output.index("{")
            json_end = output.rindex("}") + 1
            json_str = output[json_start:json_end]
            return json.loads(json_str)
        else:
            return None
    except (json.JSONDecodeError, ValueError):
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test fine-tuned model on EC2")
    
    parser.add_argument(
        "--use-ssm",
        action="store_true",
        help="Load configuration from SSM Parameter Store"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="Model timestamp to test (e.g., '20251111_022951')"
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Which sample text to use (0-2)"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Custom medical text to test (overrides --sample-index)"
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("🧪 Testing Fine-Tuned Model on EC2")
    logger.info("="*80)
    
    try:
        # Load configuration
        if args.use_ssm:
            from src.utils.config import load_all_configs
            configs = load_all_configs('config', use_ssm=True)
            
            s3_bucket = configs.get_training('output.s3_bucket')
            s3_prefix = configs.get_training('output.s3_prefix')
            base_model_id = configs.get_training('model.base_model_id')
        else:
            # Fallback to hardcoded values (for testing)
            logger.warning("⚠️  Not using SSM, using default values")
            s3_bucket = "fine-tune-llama-models-longhoang"
            s3_prefix = "models/llama-3.1-8b-medical-ie"
            base_model_id = "meta-llama/Meta-Llama-3.1-8B"
        
        logger.info(f"Model: {s3_bucket}/{s3_prefix}/{args.timestamp}")
        logger.info(f"Base: {base_model_id}")
        logger.info("")
        
        # Get test input
        if args.input:
            test_text = args.input
            logger.info("📝 Using custom input")
        else:
            test_text = SAMPLE_MEDICAL_TEXTS[args.sample_index]
            logger.info(f"📝 Using sample text #{args.sample_index + 1}")
        
        logger.info("")
        
        # Download model from S3
        temp_dir = Path(tempfile.mkdtemp(prefix="model_test_"))
        
        success = download_model_from_s3(
            bucket=s3_bucket,
            s3_prefix=s3_prefix,
            timestamp=args.timestamp,
            local_dir=temp_dir
        )
        
        if not success:
            sys.exit(1)
        
        logger.info("")
        
        # Load model
        model, tokenizer = load_model(temp_dir, base_model_id)
        
        logger.info("")
        
        # Display input
        logger.info("="*80)
        logger.info("📝 INPUT:")
        logger.info("="*80)
        print(test_text)
        logger.info("")
        
        # Run inference
        logger.info("="*80)
        logger.info("🤖 RUNNING INFERENCE...")
        logger.info("="*80)
        
        output = run_inference(model, tokenizer, test_text)
        
        logger.info("")
        logger.info("="*80)
        logger.info("📤 OUTPUT:")
        logger.info("="*80)
        print(output)
        logger.info("")
        
        # Parse output
        logger.info("="*80)
        logger.info("📊 PARSED OUTPUT:")
        logger.info("="*80)
        
        parsed = parse_output(output)
        if parsed:
            print(json.dumps(parsed, indent=2))
        else:
            logger.warning("⚠️  Could not parse JSON from output")
            print(output)
        
        logger.info("")
        logger.info("="*80)
        logger.success("✅ Test complete!")
        logger.info("="*80)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        logger.info("🧹 Cleaned up temporary files")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
