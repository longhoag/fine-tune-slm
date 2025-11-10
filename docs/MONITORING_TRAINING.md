# Monitoring Training Progress and Evaluation

This guide explains all the ways to monitor training loss, view graphs, and evaluate your fine-tuned model.

## 📊 Overview: Where Metrics Are Logged

Your training setup logs metrics to **multiple locations**:

```
Training Metrics Flow:
├── 1. Console Output (SSM stdout) - Real-time text logs
├── 2. TensorBoard - Interactive graphs and visualizations
├── 3. Training History - JSON file with all metrics
├── 4. Checkpoint Files - Saved at regular intervals
└── 5. CloudWatch Logs - AWS centralized logging
```

## 1. Real-Time Console Output

### During Training

When you run training, you'll see loss printed every `logging_steps` (configured as 10 steps):

```python
# scripts/finetune/run_training.py output shows:
⏳ Waiting for training to complete...

Output:
{'loss': 2.8453, 'learning_rate': 1.99e-04, 'epoch': 0.0, 'step': 10}
{'loss': 2.7891, 'learning_rate': 1.98e-04, 'epoch': 0.01, 'step': 20}
{'loss': 2.7234, 'learning_rate': 1.97e-04, 'epoch': 0.02, 'step': 30}
{'eval_loss': 2.6891, 'eval_runtime': 12.34, 'step': 500}
{'loss': 2.6523, 'learning_rate': 1.95e-04, 'epoch': 0.1, 'step': 500}
```

**Configuration** (in `config/training_config.yml`):
```yaml
training:
  logging_steps: 10  # Print metrics every 10 steps
  eval_steps: 500    # Run evaluation every 500 steps
```

### View Live During Background Training

If you ran training in background mode:

```bash
# Get the command ID from run_training.py output
# Then check status:
aws ssm get-command-invocation \
  --command-id abc123... \
  --instance-id i-0ad7db4eb23bd2df8 \
  --query 'StandardOutputContent' \
  --output text
```

Or use CloudWatch Logs (see section 5).

## 2. TensorBoard - Interactive Graphs

### What TensorBoard Provides

TensorBoard creates **interactive visualizations** of your training:

- 📉 **Loss curves** (training and validation)
- 📈 **Learning rate schedule**
- 🎯 **Gradient norms**
- ⏱️ **Training speed** (steps/sec)
- 📊 **Evaluation metrics**

### Where TensorBoard Logs Are Saved

**Location**: `/mnt/training/checkpoints/logs/` (on EC2 EBS volume)

**Files created**:
```
/mnt/training/checkpoints/
├── logs/
│   ├── events.out.tfevents.1699200000.ip-xxx  # TensorBoard events
│   └── events.out.tfevents.1699201000.ip-xxx
├── checkpoint-500/
├── checkpoint-1000/
└── final_model/
```

### How to View TensorBoard

#### Option 1: View on EC2 Instance

```bash
# 1. SSH into EC2 (or use SSM Session Manager)
aws ssm start-session --target i-0ad7db4eb23bd2df8

# 2. Navigate to checkpoint directory
cd /mnt/training/checkpoints

# 3. Start TensorBoard
tensorboard --logdir logs --host 0.0.0.0 --port 6006

# 4. Access via EC2 public IP (requires security group rule)
# http://<EC2-PUBLIC-IP>:6006
```

#### Option 2: Download Logs to Local Machine

```bash
# 1. Copy TensorBoard logs from EC2 to local
aws ssm send-command \
  --instance-id i-0ad7db4eb23bd2df8 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["aws s3 sync /mnt/training/checkpoints/logs s3://fine-tune-llama-models-longhoang/tensorboard-logs"]'

# 2. Download from S3 to local
aws s3 sync s3://fine-tune-llama-models-longhoang/tensorboard-logs ./tensorboard-logs

# 3. Run TensorBoard locally
tensorboard --logdir tensorboard-logs

# 4. Open browser to http://localhost:6006
```

#### Option 3: Add TensorBoard Sync to Training Script

Let me create a helper script for this:

```bash
# Create scripts/utils/sync_tensorboard.sh
#!/bin/bash
# Sync TensorBoard logs from EC2 to S3 and download locally

INSTANCE_ID="i-0ad7db4eb23bd2df8"
S3_BUCKET="fine-tune-llama-models-longhoang"
S3_PREFIX="tensorboard-logs"
LOCAL_DIR="./tensorboard-logs"

echo "Syncing TensorBoard logs from EC2 to S3..."
aws ssm send-command \
  --instance-id $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters "commands=[\"aws s3 sync /mnt/training/checkpoints/logs s3://$S3_BUCKET/$S3_PREFIX\"]"

sleep 5

echo "Downloading logs from S3 to local..."
aws s3 sync s3://$S3_BUCKET/$S3_PREFIX $LOCAL_DIR

echo "Starting TensorBoard..."
tensorboard --logdir $LOCAL_DIR --host 0.0.0.0 --port 6006

echo "Open http://localhost:6006 in your browser"
```

**Usage**:
```bash
# Make executable
chmod +x scripts/utils/sync_tensorboard.sh

# Run it
./scripts/utils/sync_tensorboard.sh
```

### TensorBoard Dashboard Overview

Once TensorBoard is running, you'll see:

**SCALARS Tab**:
- `train/loss` - Training loss over time
- `eval/loss` - Validation loss over time
- `train/learning_rate` - Learning rate schedule
- `train/epoch` - Current epoch
- `train/global_step` - Total steps completed

**Example Graph**:
```
Loss
│
3.0 ┤     ●
    │    ●●
2.5 ┤   ●  ●
    │  ●    ●
2.0 ┤ ●      ●●
    │●         ●●●
1.5 ┤            ●●●●
    └─────────────────── Steps
    0   500  1000  1500
```

## 3. Training History JSON

### Where It's Saved

HuggingFace Trainer automatically saves `trainer_state.json`:

**Location**: `/mnt/training/checkpoints/checkpoint-{N}/trainer_state.json`

### What It Contains

```json
{
  "best_metric": 2.456,
  "best_model_checkpoint": "/mnt/training/checkpoints/checkpoint-1500",
  "epoch": 2.8,
  "global_step": 1500,
  "log_history": [
    {
      "epoch": 0.01,
      "learning_rate": 1.99e-04,
      "loss": 2.8453,
      "step": 10
    },
    {
      "epoch": 0.02,
      "learning_rate": 1.98e-04,
      "loss": 2.7891,
      "step": 20
    },
    {
      "epoch": 0.5,
      "eval_loss": 2.6234,
      "eval_runtime": 15.23,
      "step": 500
    }
  ],
  "total_flos": 1.23e15,
  "train_batch_size": 16,
  "trial_name": null,
  "trial_params": null
}
```

### How to Access

```bash
# Download specific checkpoint
aws s3 cp s3://fine-tune-llama-models-longhoang/models/llama-3.1-8b-medical-ie/checkpoint-1500/trainer_state.json .

# Parse and plot locally
python scripts/utils/plot_training_history.py trainer_state.json
```

## 4. Checkpoint Evaluation

### Saved Checkpoints

**Location**: `/mnt/training/checkpoints/checkpoint-{step}/`

**Files in each checkpoint**:
```
checkpoint-500/
├── adapter_config.json        # LoRA configuration
├── adapter_model.safetensors  # LoRA weights (small ~100MB)
├── trainer_state.json         # Training history
├── training_args.bin          # Training arguments
└── optimizer.pt               # Optimizer state (large)
```

**Configuration**:
```yaml
training:
  save_steps: 500         # Save checkpoint every 500 steps
  save_total_limit: 3     # Keep only last 3 checkpoints
  load_best_model_at_end: true
  metric_for_best_model: eval_loss  # Choose best by validation loss
```

### Evaluate Specific Checkpoint

To test a specific checkpoint:

```bash
# Run evaluation on checkpoint
python scripts/finetune/evaluate_checkpoint.py \
  --checkpoint /mnt/training/checkpoints/checkpoint-1500 \
  --test-samples 100
```

(Note: We need to create this script - see implementation below)

## 5. CloudWatch Logs

### Where SSM Command Output Is Logged

All SSM command output goes to CloudWatch Logs:

**Log Group**: `/aws/ssm/AWS-RunShellScript` (default) or custom group

### View Logs

```bash
# Stream logs in real-time
aws logs tail /aws/ssm/AWS-RunShellScript --follow

# Get logs from last hour
aws logs tail /aws/ssm/AWS-RunShellScript --since 1h

# Search for specific term
aws logs filter-log-events \
  --log-group-name /aws/ssm/AWS-RunShellScript \
  --filter-pattern "loss"
```

### CloudWatch Insights Queries

Create advanced queries in CloudWatch Insights:

```sql
# Query 1: Extract all loss values
fields @timestamp, @message
| filter @message like /loss/
| parse @message /'loss': *,/ as loss
| sort @timestamp desc

# Query 2: Count errors
fields @timestamp, @message
| filter @message like /ERROR/
| stats count() by bin(5m)
```

## 6. Evaluation Metrics

### Built-in Evaluation

During training, the model automatically evaluates on validation set every `eval_steps`:

```yaml
training:
  evaluation_strategy: steps
  eval_steps: 500  # Run evaluation every 500 steps
```

**Metrics computed**:
- `eval_loss` - Cross-entropy loss on validation set
- `eval_runtime` - Time taken for evaluation
- `eval_samples_per_second` - Throughput
- `eval_steps_per_second` - Steps throughput

### Custom Evaluation Script

Create `scripts/finetune/evaluate_checkpoint.py`:

```python
#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on test data.

Usage:
    python scripts/finetune/evaluate_checkpoint.py \\
        --checkpoint /mnt/training/checkpoints/checkpoint-1500 \\
        --test-data synthetic-instruction-tuning-dataset/validation.jsonl \\
        --num-samples 100
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logger import logger


def load_checkpoint(checkpoint_path: str, base_model: str = "meta-llama/Meta-Llama-3.1-8B"):
    """Load LoRA checkpoint."""
    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=True,
        device_map="auto"
    )
    
    logger.info(f"Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    
    return model, tokenizer


def evaluate_sample(model, tokenizer, instruction: str, input_text: str, expected_output: Dict) -> Dict:
    """Evaluate single sample."""
    # Format prompt
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical information extraction assistant. Extract structured cancer-related entities from clinical text.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}

Text: {input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract generated output (after last assistant token)
    generated_output = generated.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    
    # Parse JSON
    try:
        predicted = json.loads(generated_output)
    except json.JSONDecodeError:
        predicted = {"error": "Failed to parse JSON", "raw": generated_output}
    
    return {
        "input": input_text,
        "expected": expected_output,
        "predicted": predicted,
        "match": predicted == expected_output
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test JSONL")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to test")
    
    args = parser.parse_args()
    
    # Load checkpoint
    model, tokenizer = load_checkpoint(args.checkpoint)
    
    # Load test data
    logger.info(f"Loading test data: {args.test_data}")
    test_samples = []
    with open(args.test_data, 'r') as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            test_samples.append(json.loads(line))
    
    # Evaluate
    results = []
    correct = 0
    for i, sample in enumerate(test_samples):
        logger.info(f"Evaluating sample {i+1}/{len(test_samples)}")
        result = evaluate_sample(
            model, tokenizer,
            sample['instruction'],
            sample['input'],
            sample['output']
        )
        results.append(result)
        if result['match']:
            correct += 1
    
    # Report
    accuracy = correct / len(results) * 100
    logger.success(f"\\n{'='*60}")
    logger.success(f"Evaluation Results")
    logger.success(f"{'='*60}")
    logger.info(f"Samples evaluated: {len(results)}")
    logger.info(f"Exact matches: {correct}")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    
    # Save results
    output_file = f"{args.checkpoint}/evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
```

## 7. Summary: Complete Monitoring Workflow

### During Training

```bash
# Option 1: Monitor in foreground (see live logs)
python scripts/finetune/run_training.py

# Option 2: Background + CloudWatch monitoring
python scripts/finetune/run_training.py --background
aws logs tail /aws/ssm/AWS-RunShellScript --follow
```

### After Training

```bash
# 1. Download TensorBoard logs
./scripts/utils/sync_tensorboard.sh
# Open http://localhost:6006

# 2. View training history
aws s3 cp s3://fine-tune-llama-models-longhoang/.../trainer_state.json .
python scripts/utils/plot_training_history.py trainer_state.json

# 3. Evaluate best checkpoint
python scripts/finetune/evaluate_checkpoint.py \\
  --checkpoint /mnt/training/checkpoints/checkpoint-1500 \\
  --test-data synthetic-instruction-tuning-dataset/validation.jsonl
```

## 8. Key Metrics to Watch

### Training Loss
- **Good**: Steadily decreasing
- **Concerning**: Plateaus early or increases
- **Target**: Below 2.0 for medical IE task

### Validation Loss
- **Good**: Decreases with training loss
- **Overfitting**: Training loss decreases but validation increases
- **Target**: Close to training loss (gap < 0.3)

### Learning Rate
- **Pattern**: Warmup → plateau → optional decay
- **Check**: Follows schedule from config

### Training Speed
- **Expected**: ~1-2 steps/second on g6.2xlarge
- **Slower**: Check GPU utilization

## 9. Cost-Effective Monitoring

**Problem**: TensorBoard requires instance running  
**Solution**: Sync logs periodically

```bash
# Add to training script or run every 30 minutes
*/30 * * * * aws s3 sync /mnt/training/checkpoints/logs s3://bucket/tensorboard-logs
```

Then view locally without EC2 running! 💰

## 10. Expected Warnings (Non-Critical)

During training, you may see these warnings - **they are normal and don't indicate errors**:

### ✅ Gradient Checkpointing Warning (FIXED in latest version)
```
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
```
- **Status**: Fixed in latest version by explicitly setting `use_cache=False`
- **What it was**: Gradient checkpointing saves memory by recomputing activations
- **Action needed**: None - update to latest Docker image to eliminate warning

### ✅ PyTorch Checkpoint Warning (FIXED in latest version)
```
torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.
```
- **Status**: Fixed in latest version by setting `use_reentrant=False`
- **What it was**: PyTorch recommends explicitly setting this parameter
- **Action needed**: None - update to latest Docker image to eliminate warning

### ✅ Base Model Config Warning (FIXED in latest version)
```
Unable to fetch remote file due to 401 Client Error...
Could not find a config file in meta-llama/Meta-Llama-3.1-8B
```
- **Status**: Fixed in latest version by HuggingFace Hub login
- **What it was**: PEFT couldn't fetch base model config during save
- **Action needed**: None - update to latest Docker image to eliminate warning

### 🚨 What SHOULD Worry You

These are actual errors:
- `CUDA out of memory` - Reduce batch size or max_seq_length
- `Dataset not found` - Check dataset paths in config
- `Training failed:` with exception - See CloudWatch logs for details
- Training stops unexpectedly - Check instance health and disk space

## 11. Quick Reference

| What to Monitor | Where | How |
|----------------|-------|-----|
| Real-time loss | SSM output | `run_training.py` or CloudWatch |
| Loss graphs | TensorBoard | Download logs, run locally |
| Training history | `trainer_state.json` | In checkpoints, download from S3 |
| Evaluation metrics | Validation steps | Every `eval_steps` in logs |
| Custom evaluation | Test script | `evaluate_checkpoint.py` |
| CloudWatch logs | AWS Console | Search "loss", "error", etc. |

**Recommended**: Use TensorBoard for graphs + CloudWatch for troubleshooting!
