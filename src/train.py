#!/usr/bin/env python3
"""
Fine-tuning script for Llama 3.1 8B with QLoRA (4-bit quantization + LoRA)

This script implements:
- Dataset loading from JSONL files
- 4-bit quantization with BitsAndBytes (NF4)
- LoRA configuration with PEFT
- HuggingFace Trainer setup
- Checkpoint saving to EBS mount
- Model upload to S3 and Hugging Face Hub
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.utils import ConfigLoader, MultiConfigLoader, load_all_configs
from src.utils.logger import logger
from src.utils.aws_helpers import S3Manager, SecretsManager, AWSClient


# ============================================================================
# Dataset Loading and Processing
# ============================================================================


def load_jsonl_dataset(file_path: str) -> List[Dict]:
    """Load JSONL file into list of dictionaries.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of parsed JSON objects
    """
    logger.info(f"Loading JSONL dataset from: {file_path}")
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(data)} examples from {file_path}")
    return data


def format_instruction_example(example: Dict) -> str:
    """Format a single example into instruction-following format.
    
    Args:
        example: Dict with 'instruction', 'input', and 'output' keys
        
    Returns:
        Formatted string for training
    """
    instruction = example['instruction']
    input_text = example['input']
    output_data = example['output']
    
    # Format output as structured JSON string
    output_str = json.dumps(output_data, ensure_ascii=False)
    
    # Use Llama 3.1 chat template format
    formatted = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a medical information extraction assistant. Extract structured cancer-related entities from clinical text.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}\n\nText: {input_text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{output_str}<|eot_id|>"
    )
    
    return formatted


def prepare_dataset(
    train_path: str,
    validation_path: str,
    tokenizer,
    max_seq_length: int = 2048,
) -> Tuple[Dataset, Dataset]:
    """Load and prepare datasets for training.
    
    Args:
        train_path: Path to training JSONL file
        validation_path: Path to validation JSONL file
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length
        
    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
    logger.info("="*60)
    logger.info("Preparing Datasets")
    logger.info("="*60)
    
    # Load JSONL files
    train_data = load_jsonl_dataset(train_path)
    val_data = load_jsonl_dataset(validation_path)
    
    # Format examples
    logger.info("Formatting examples for instruction tuning...")
    train_formatted = [format_instruction_example(ex) for ex in train_data]
    val_formatted = [format_instruction_example(ex) for ex in val_data]
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_dict({"text": train_formatted})
    val_dataset = Dataset.from_dict({"text": val_formatted})
    
    # Tokenize
    logger.info("Tokenizing datasets...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,  # Dynamic padding in collator
        )
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing training data",
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing validation data",
    )
    
    logger.info(f"✅ Train dataset: {len(train_dataset)} examples")
    logger.info(f"✅ Validation dataset: {len(val_dataset)} examples")
    
    return train_dataset, val_dataset


# ============================================================================
# Model Setup with QLoRA
# ============================================================================


def setup_model_and_tokenizer(
    model_name: str,
    lora_config: Dict,
    quantization_config: Dict,
    hf_token: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Set up Llama model with 4-bit quantization and LoRA.
    
    Args:
        model_name: HuggingFace model ID (e.g., "meta-llama/Meta-Llama-3.1-8B")
        lora_config: LoRA configuration dict from training_config.yml
        quantization_config: Quantization config dict from training_config.yml
        hf_token: HuggingFace API token for gated models
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("="*60)
    logger.info("Setting Up Model with QLoRA")
    logger.info("="*60)
    logger.info(f"Model: {model_name}")
    logger.info(f"LoRA rank: {lora_config['r']}")
    logger.info(f"LoRA alpha: {lora_config['lora_alpha']}")
    logger.info(f"Quantization: 4-bit NF4")
    
    # Configure 4-bit quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization_config['load_in_4bit'],
        bnb_4bit_compute_dtype=getattr(torch, quantization_config['bnb_4bit_compute_dtype']),
        bnb_4bit_quant_type=quantization_config['bnb_4bit_quant_type'],
        bnb_4bit_use_double_quant=quantization_config['bnb_4bit_use_double_quant'],
    )
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )
    
    # Set pad token (Llama doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
        dtype=torch.float16,
    )
    
    # Prepare model for k-bit training
    logger.info("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    logger.info("Configuring LoRA adapters...")
    peft_config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        bias=lora_config['bias'],
        task_type=lora_config['task_type'],
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params
    
    logger.info("="*60)
    logger.info(f"✅ Trainable parameters: {trainable_params:,}")
    logger.info(f"✅ Total parameters: {total_params:,}")
    logger.info(f"✅ Trainable: {trainable_percent:.2f}%")
    logger.info("="*60)
    
    return model, tokenizer


# ============================================================================
# Training Setup
# ============================================================================


def setup_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    training_config: Dict,
    output_dir: str,
) -> Trainer:
    """Set up HuggingFace Trainer with training configuration.
    
    Args:
        model: PEFT model with LoRA
        tokenizer: HuggingFace tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        training_config: Training config dict from training_config.yml
        output_dir: Directory for checkpoints
        
    Returns:
        Configured Trainer instance
    """
    logger.info("="*60)
    logger.info("Setting Up Trainer")
    logger.info("="*60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build training arguments dict
    training_args_dict = {
        'output_dir': output_dir,
        'per_device_train_batch_size': training_config['per_device_train_batch_size'],
        'per_device_eval_batch_size': training_config['per_device_eval_batch_size'],
        'gradient_accumulation_steps': training_config['gradient_accumulation_steps'],
        'learning_rate': training_config['learning_rate'],
        'warmup_steps': training_config['warmup_steps'],
        'logging_steps': training_config['logging_steps'],
        'save_steps': training_config['save_steps'],
        'eval_steps': training_config['eval_steps'],
        'save_total_limit': training_config['save_total_limit'],
        'fp16': training_config['fp16'],
        'optim': training_config['optim'],
        'eval_strategy': training_config['evaluation_strategy'],
        'load_best_model_at_end': training_config['load_best_model_at_end'],
        'metric_for_best_model': training_config['metric_for_best_model'],
        'report_to': ["tensorboard"],
        'logging_dir': f"{output_dir}/logs",
        'save_safetensors': True,
        'remove_unused_columns': False,
    }
    
    # Handle max_steps vs num_train_epochs
    # When max_steps is set (test mode), use it; otherwise use num_train_epochs
    max_steps = training_config.get('max_steps')
    if max_steps and max_steps > 0:
        training_args_dict['max_steps'] = max_steps
        # Don't set num_train_epochs when using max_steps
    else:
        training_args_dict['num_train_epochs'] = training_config.get('num_train_epochs', 3)
        training_args_dict['max_steps'] = -1  # -1 means use epochs
    
    # Configure training arguments
    training_args = TrainingArguments(**training_args_dict)
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    logger.info(f"✅ Output directory: {output_dir}")
    logger.info(f"✅ Effective batch size: {training_config['per_device_train_batch_size'] * training_config['gradient_accumulation_steps']}")
    logger.info(f"✅ Learning rate: {training_config['learning_rate']}")
    logger.info(f"✅ Optimizer: {training_config['optim']}")
    
    return trainer


# ============================================================================
# Main Training Function
# ============================================================================


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.1 8B with QLoRA")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing config files (default: config/)",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="synthetic-instruction-tuning-dataset/train.jsonl",
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="synthetic-instruction-tuning-dataset/validation.jsonl",
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--use-ssm",
        action="store_true",
        help="Use SSM Parameter Store for config resolution (default: False for local testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without training (useful for local testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps (useful for testing)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start training from scratch, ignoring existing checkpoints",
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("🚀 Llama 3.1 8B Medical IE Fine-Tuning with QLoRA")
    logger.info("="*60)
    logger.info(f"Config directory: {args.config_dir}")
    logger.info(f"Use SSM: {args.use_ssm}")
    logger.info(f"Dry run: {args.dry_run}")
    
    try:
        # ====================================================================
        # 1. Load Configuration
        # ====================================================================
        logger.info("\n📋 Loading configuration...")
        configs = load_all_configs(args.config_dir, use_ssm=args.use_ssm)
        
        # Extract configs
        model_name = configs.get_training('model.name')
        lora_config = configs.get_training('lora')
        quantization_config = configs.get_training('quantization')
        training_config = configs.get_training('training')
        dataset_config = configs.get_training('dataset')
        
        # Override output dir if specified
        output_dir = args.output_dir or training_config['output_dir']
        
        logger.info(f"✅ Model: {model_name}")
        logger.info(f"✅ Output directory: {output_dir}")
        
        # ====================================================================
        # 2. Get HuggingFace Token
        # ====================================================================
        logger.info("\n🔑 Retrieving HuggingFace token...")
        hf_token = None
        
        if args.use_ssm:
            # Get token from AWS Secrets Manager
            aws_client = AWSClient(region=configs.get_aws('aws.region'))
            secrets_mgr = SecretsManager(aws_client)
            hf_token_secret = configs.get_aws('aws.secrets_manager.hf_token_secret')
            hf_token = secrets_mgr.get_secret(hf_token_secret)
            logger.info(f"✅ Retrieved HF token from Secrets Manager: {hf_token_secret}")
        else:
            # Try to get from environment variable for local testing
            hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
            if hf_token:
                logger.info("✅ Using HF token from HUGGING_FACE_HUB_TOKEN environment variable")
            else:
                logger.warning("⚠️  No HF token found. Set HUGGING_FACE_HUB_TOKEN for gated models.")
        
        # Login to HuggingFace Hub (needed for PEFT to access base model configs)
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            logger.info("✅ Logged in to HuggingFace Hub")
        
        # ====================================================================
        # 3. Setup Model and Tokenizer
        # ====================================================================
        logger.info("\n🤖 Setting up model and tokenizer...")
        model, tokenizer = setup_model_and_tokenizer(
            model_name=model_name,
            lora_config=lora_config,
            quantization_config=quantization_config,
            hf_token=hf_token,
        )
        
        if args.dry_run:
            logger.info("\n✅ Dry run: Model setup successful!")
            logger.info("Skipping dataset loading and training...")
            return 0
        
        # ====================================================================
        # 4. Prepare Datasets
        # ====================================================================
        logger.info("\n📚 Preparing datasets...")
        
        # Use dataset paths from config when --use-ssm, otherwise use CLI args
        train_path = dataset_config['train_path'] if args.use_ssm else args.train_data
        val_path = dataset_config['validation_path'] if args.use_ssm else args.val_data
        
        train_dataset, val_dataset = prepare_dataset(
            train_path=train_path,
            validation_path=val_path,
            tokenizer=tokenizer,
            max_seq_length=dataset_config['max_seq_length'],
        )
        
        # ====================================================================
        # 5. Setup Trainer
        # ====================================================================
        logger.info("\n⚙️  Setting up trainer...")
        
        # Override max_steps if specified (for testing)
        if args.max_steps:
            logger.info(f"⚠️  Overriding max_steps to {args.max_steps} (test mode)")
            training_config['max_steps'] = args.max_steps
            # num_train_epochs will be ignored when max_steps > 0
        
        trainer = setup_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_config=training_config,
            output_dir=output_dir,
        )
        
        # ====================================================================
        # 6. Train!
        # ====================================================================
        logger.info("\n🏋️  Starting training...")
        logger.info("="*60)
        
        # Check for existing checkpoints to resume from
        import glob
        import shutil
        from datetime import datetime
        
        checkpoint_dirs = glob.glob(f"{output_dir}/checkpoint-*")
        final_model_path = f"{output_dir}/final_model"
        final_model_exists = Path(final_model_path).exists()
        resume_from_checkpoint = None
        
        if checkpoint_dirs and not args.no_resume:
            # Sort by checkpoint number and get the latest
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
            latest_checkpoint = checkpoint_dirs[-1]
            checkpoint_step = latest_checkpoint.split('-')[-1]
            
            logger.info(f"📂 Found existing checkpoint: {latest_checkpoint}")
            logger.info(f"🔄 Resuming training from step {checkpoint_step}")
            resume_from_checkpoint = latest_checkpoint
            
        elif (checkpoint_dirs or final_model_exists) and args.no_resume:
            # Backup existing checkpoints/models before starting fresh
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{output_dir}/backup_{timestamp}"
            
            if checkpoint_dirs:
                logger.warning(f"⚠️  Found {len(checkpoint_dirs)} checkpoint(s) but --no-resume flag set")
            if final_model_exists:
                logger.warning("⚠️  Found existing final_model but --no-resume flag set")
            
            logger.info(f"📦 Backing up to: {backup_dir}")
            Path(backup_dir).mkdir(parents=True, exist_ok=True)
            
            # Backup checkpoints
            for checkpoint_dir in checkpoint_dirs:
                checkpoint_name = Path(checkpoint_dir).name
                shutil.move(checkpoint_dir, f"{backup_dir}/{checkpoint_name}")
                logger.info(f"📦 Backed up: {checkpoint_name}")
            
            # Backup final_model
            if final_model_exists:
                shutil.move(final_model_path, f"{backup_dir}/final_model")
                logger.info(f"📦 Backed up: final_model")
            
            logger.info("✅ Backup complete. Starting training from scratch...")
            
        elif final_model_exists and not args.no_resume:
            # Found final_model but no checkpoints - likely from previous test run
            # Back it up automatically to prevent overwriting
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{output_dir}/backup_{timestamp}"
            
            logger.warning("⚠️  Found existing final_model from previous run")
            logger.info(f"📦 Auto-backing up to: {backup_dir}/final_model")
            Path(backup_dir).mkdir(parents=True, exist_ok=True)
            shutil.move(final_model_path, f"{backup_dir}/final_model")
            logger.info("✅ Backup complete. Starting training from scratch...")
            
        else:
            logger.info("Starting training from scratch (no checkpoints found)")
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        logger.info("\n✅ Training complete!")
        
        # ====================================================================
        # 7. Save Final Model (Skip in test mode)
        # ====================================================================
        # Detect test mode: if max_steps is very small (< 50), skip saving
        is_test_mode = args.max_steps and args.max_steps < 50
        
        if is_test_mode:
            logger.warning("\n⚠️  Test mode detected (max_steps < 50)")
            logger.warning("⚠️  Skipping final model save and S3 upload")
            logger.info("\n" + "="*60)
            logger.info("🎉 Test run completed successfully!")
            logger.info("="*60)
            return 0
        
        logger.info("\n💾 Saving final model...")
        final_model_path = f"{output_dir}/final_model"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        logger.info(f"✅ Model saved to: {final_model_path}")
        
        # ====================================================================
        # 8. Upload to S3 (if using SSM) with timestamp
        # ====================================================================
        if args.use_ssm:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            logger.info("\n☁️  Uploading model to S3...")
            s3_bucket = configs.get_training('output.s3_bucket')
            s3_prefix_base = configs.get_training('output.s3_prefix')
            
            # Add timestamp to prevent overwriting previous runs
            s3_prefix = f"{s3_prefix_base}/{timestamp}"
            
            s3_mgr = S3Manager(aws_client)
            s3_mgr.upload_directory(
                local_path=final_model_path,
                bucket=s3_bucket,
                prefix=f"{s3_prefix}/final_model",
            )
            
            logger.info(f"✅ Model uploaded to s3://{s3_bucket}/{s3_prefix}/final_model")
        
        logger.info("\n" + "="*60)
        logger.info("🎉 Fine-tuning completed successfully!")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

