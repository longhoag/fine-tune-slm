# Testing Guide for Fine-Tuning Scripts

This guide explains how to test the fine-tuning environment without actually running a full training job.

## Overview

We have **3 testing modes** to verify the environment is ready before committing to a 3-4 hour training run:

1. **Dry Run Mode** (`--dry-run`) - Validates environment without any training
2. **Test Mode** (`--test`) - Runs 5 training steps to verify everything works
3. **Full Training** - Runs complete training (3-4 hours)

## Prerequisites

Before testing, ensure you have:

1. ✅ EC2 instance running: `poetry run python scripts/setup/start_ec2.py`
2. ✅ Environment deployed: `poetry run python scripts/setup/deploy_via_ssm.py`
3. ✅ Docker image pulled and GPU verified (from deploy script)
4. ✅ EBS volume mounted at `/mnt/training`


To stop instance: `poetry run python scripts/setup/stop_ec2.py`


## Testing Modes

### 1. Dry Run Mode (Recommended First Step)

**Purpose**: Validates that all components are configured correctly without training.

**What it checks**:
- ✅ Configuration files load properly
- ✅ AWS credentials and permissions work
- ✅ Dataset files are accessible
- ✅ HuggingFace token is valid
- ✅ Model can be loaded (Llama 3.1 8B)
- ✅ Tokenizer initializes
- ✅ GPU is available and compatible
- ✅ Output directory is writable

**How to run**:
```bash
poetry run python scripts/finetune/run_training.py --dry-run
```

**Expected output**:
```
====================================================
Running Fine-Tuning on EC2 Instance
====================================================
Instance ID: i-0ad7db4eb23bd2df8
Region: us-east-1
ECR Image: 855386719590.dkr.ecr.us-east-1.amazonaws.com/fine-tune-llama:latest

Verifying instance state...
Instance is running ✓
Verifying SSM connectivity...
Instance is running and SSM is online! ✓

============================================================
Starting Fine-Tuning: DRY RUN (Environment Validation Only)
============================================================

Training command:
docker run --rm --gpus all -v /mnt/training:/mnt/training -v /home/ubuntu/fine-tune-slm:/workspace -w /workspace 855386719590.dkr.ecr.us-east-1.amazonaws.com/fine-tune-llama:latest python -m src.train --use-ssm --output-dir /mnt/training/checkpoints --dry-run

Sending training command via SSM...
Command sent! Command ID: abc123...

⏳ Waiting for training to complete (timeout: 600s)...

Training completed with status: Success
✅ Training completed successfully!

Output:
Loading configuration...
✅ Model: meta-llama/Llama-3.1-8B
✅ Output directory: /mnt/training/checkpoints
✅ Retrieved HF token from Secrets Manager
✅ Loading tokenizer...
✅ Loading model with 4-bit quantization...
✅ GPU available: NVIDIA L4 (24GB)
✅ Loading datasets...
✅ Loaded 4500 training examples
✅ Loaded 500 validation examples
✅ Applying LoRA configuration...
✅ All checks passed! Environment is ready for training.
DRY RUN COMPLETE - No training performed
```

**Duration**: ~2-3 minutes

**Cost**: ~$0.03 (3 minutes × $0.7512/hour)

### 2. Test Mode (Verify Training Works)

**Purpose**: Runs a minimal training job (5 steps) to verify the complete training pipeline works.

**What it tests**:
- ✅ Everything from dry run, PLUS:
- ✅ Training loop executes
- ✅ Forward pass works
- ✅ Backward pass and gradient computation works
- ✅ Optimizer updates weights
- ✅ Checkpoint saving works
- ✅ Loss is being calculated
- ✅ GPU memory is sufficient

**How to run**:
```bash
poetry run python scripts/finetune/run_training.py --test
```

**Expected output**:
```
============================================================
Starting Fine-Tuning: TEST MODE (5 steps to verify setup)
============================================================

Training command:
docker run --rm --gpus all -v /mnt/training:/mnt/training -v /home/ubuntu/fine-tune-slm:/workspace -w /workspace 855386719590.dkr.ecr.us-east-1.amazonaws.com/fine-tune-llama:latest python -m src.train --use-ssm --output-dir /mnt/training/checkpoints --max-steps 5

Sending training command via SSM...
Command sent! Command ID: def456...

⏳ Waiting for training to complete (timeout: 600s)...

Output:
Loading configuration...
✅ Model loaded with LoRA
✅ Training dataset: 4500 examples
✅ Validation dataset: 500 examples
Starting training...

{'loss': 2.8453, 'learning_rate': 1.99e-04, 'epoch': 0.0, 'step': 1}
{'loss': 2.7891, 'learning_rate': 1.98e-04, 'epoch': 0.0, 'step': 2}
{'loss': 2.7234, 'learning_rate': 1.97e-04, 'epoch': 0.0, 'step': 3}
{'loss': 2.6891, 'learning_rate': 1.96e-04, 'epoch': 0.0, 'step': 4}
{'loss': 2.6523, 'learning_rate': 1.95e-04, 'epoch': 0.0, 'step': 5}

Saving checkpoint to /mnt/training/checkpoints/checkpoint-5
Training completed successfully!

Next steps:
  1. Check model artifacts (on EC2 instance): ls /mnt/training/checkpoints
  2. Push to HuggingFace (if locally): poetry run python scripts/finetune/push_to_hf.py
  3. Stop instance (if locally): poetry run python scripts/setup/stop_ec2.py
```

**Duration**: ~5-7 minutes

**Cost**: ~$0.05 (7 minutes × $0.7512/hour)

**What to verify**:
- ✅ Loss is decreasing (even slightly)
- ✅ No CUDA out of memory errors
- ✅ Checkpoint directory created at `/mnt/training/checkpoints/checkpoint-5`
- ✅ No Python errors or warnings

### 3. Full Training

**Purpose**: Run the complete fine-tuning job.

**How to run**:
```bash
# Run in foreground (blocks until complete, ~3-4 hours)
poetry run python scripts/finetune/run_training.py

# OR run in background (returns immediately)
poetry run python scripts/finetune/run_training.py --background
```

**Background mode output**:
```
============================================================
Starting Fine-Tuning: FULL TRAINING
============================================================

Command sent! Command ID: ghi789...

🎯 Running in background mode
Command ID: ghi789...

To check status:
  aws ssm get-command-invocation --command-id ghi789... --instance-id i-0ad7db4eb23bd2df8

To view CloudWatch logs:
  aws logs tail /aws/ssm/fine-tune-llama --follow
```

**Duration**: 3-4 hours

**Cost**: ~$2.25 (3 hours × $0.7512/hour)

## Testing Workflow

Here's the recommended step-by-step testing process:

```bash
# 1. Start EC2 instance (~23 seconds)
poetry run python scripts/setup/start_ec2.py

# 2. Deploy environment (~20 seconds if cached)
poetry run python scripts/setup/deploy_via_ssm.py

# 3. DRY RUN - Validate configuration (~3 minutes, $0.03)
poetry run python scripts/finetune/run_training.py --dry-run

# 4. TEST MODE - Verify training works (~7 minutes, $0.05)
poetry run python scripts/finetune/run_training.py --test

# 5. (Optional) Monitor training
aws logs tail /aws/ssm/fine-tune-llama --follow

# 6. Stop instance to save costs
poetry run python scripts/setup/stop_ec2.py
```

## Complete Full-Training  Workflow

Here's the recommended step-by-step full training process:

### 1. Setting up

```bash
# Start EC2 instance (~23 seconds)
poetry run python scripts/setup/start_ec2.py

# Deploy environment (~20 seconds if cached)
poetry run python scripts/setup/deploy_via_ssm.py
```

### 2. Run Full Training

```bash
# If test passed, run full training in background
poetry run python scripts/finetune/run_training.py --background

# (Optional) Monitor training -->  we can also view this on CloudWatch
aws logs tail /aws/ssm/fine-tune-llama --follow

# Stop instance to save costs after training is done
poetry run python scripts/setup/stop_ec2.py
```

### 3. Push Model to HF 

We don't need EC2 to be running, because at the end of training, the final model and training logs get pushed to S3. We can retrieve files from S3 without the need of EC2 running. 

```bash
# After training completes
#    - List all uploaded models on S3
poetry run python scripts/finetune/push_to_hf.py --list

#    - Push the latest model (20251111_022951)
poetry run python scripts/finetune/push_to_hf.py

#    - Or specify the timestamp explicitly
poetry run python scripts/finetune/push_to_hf.py --timestamp 20251111_022951
```

### 4. View Training Logs as Graphs

Option 1: View metrics with plots (recommended)

```bash
poetry run python scripts/utils/view_training_metrics.py

# View specific training run
poetry run python scripts/utils/view_training_metrics.py --timestamp 20251111_022951

# Save plots to file
poetry run python scripts/utils/view_training_metrics.py --save-plots training_metrics.png

# Export metrics to CSV for analysis
poetry run python scripts/utils/view_training_metrics.py --export-csv metrics.csv
```

Option 2: Traditional TensorBoard (more interactive)

```bash
./scripts/utils/sync_tensorboard.sh
# Opens http://localhost:6006

# View specific run in TensorBoard
./scripts/utils/sync_tensorboard.sh 20251111_022951
```

## Troubleshooting

### Issue: Dry run fails with "HuggingFace token invalid"

**Solution**: Check that the HF token is stored in AWS Secrets Manager:
```bash
aws secretsmanager get-secret-value --secret-id /fine-tune-slm/huggingface/token --query SecretString --output text
```

### Issue: Test mode fails with CUDA out of memory

**Solution**: This means the model or batch size is too large. Check:
1. Training config uses 4-bit quantization (`quantization.load_in_4bit: true`)
2. Batch size is reasonable (`training.per_device_train_batch_size: 1`)
3. Gradient accumulation is set (`training.gradient_accumulation_steps: 4`)

### Issue: Checkpoint directory not found

**Solution**: Verify EBS volume is mounted:
```bash
# SSH into instance or use SSM
aws ssm start-session --target i-0ad7db4eb23bd2df8

# Check mount
df -h | grep /mnt/training

# If not mounted, run deploy again
poetry run python scripts/setup/deploy_via_ssm.py
```

### Issue: Training hangs or takes too long

**Solution**: Check CloudWatch logs for errors:
```bash
aws logs tail /aws/ssm/fine-tune-llama --follow --since 10m
```

## Cost Breakdown for Testing

| Test Mode | Duration | Cost | What It Verifies |
|-----------|----------|------|------------------|
| Dry Run | 3 min | $0.03 | Environment setup |
| Test Mode | 7 min | $0.05 | Training pipeline |
| Full Training | 3-4 hrs | $2.25 | Complete fine-tuning |
| **Total Testing** | **10 min** | **$0.08** | **Everything works** |

**Recommendation**: Spend $0.08 on testing to avoid wasting $2.25 on a broken full training run!

## Testing Trained Model

After training completes, you can test the model's inference capabilities on EC2 (uses GPU for fast inference).

### Prerequisites

1. **Start EC2 instance:**
   ```bash
   poetry run python scripts/setup/start_ec2.py
   ```

2. **Verify environment deployed** (usually already done during training)

### Test with Sample Medical Text

```bash
# Test latest model with built-in sample text
poetry run python scripts/finetune/test_model_ec2.py

# Test specific timestamped model
poetry run python scripts/finetune/test_model_ec2.py --timestamp 20251111_022951

# Test with custom medical text
poetry run python scripts/finetune/test_model_ec2.py --input "62-year-old with stage IV lung cancer..."

# Use different sample text (0, 1, or 2)
poetry run python scripts/finetune/test_model_ec2.py --sample-index 1
```

### Stop EC2 When Done

```bash
# Always stop EC2 after testing to save costs
poetry run python scripts/setup/stop_ec2.py
```

**What happens:**
1. ✅ Verifies EC2 instance is running
2. ✅ Downloads model from S3 (specific timestamp)
3. ✅ Loads base Llama 3.1 8B + your LoRA adapters
4. ✅ Runs inference on GPU
5. ✅ Shows extracted entities in terminal

**Important**: You must manually stop EC2 when done with `scripts/setup/stop_ec2.py`

**Example output:**
```
================================================================================
📝 INPUT:
================================================================================
70-year-old man with widely metastatic cutaneous melanoma involving the brain,
liver, and bones. Biopsy showed BRAF V600E mutation. PD-L1 expression was 5%
and tumor mutational burden was high. Started on combination nivolumab and
ipilimumab...

================================================================================
📤 OUTPUT:
================================================================================
{
  "cancer_type": "melanoma (cutaneous)",
  "stage": "IV",
  "gene_mutation": "BRAF V600E",
  "biomarker": "PD-L1 5%; TMB-high",
  "treatment": "nivolumab and ipilimumab",
  "response": "mixed response",
  "metastasis_site": "brain, liver, bones"
}

✅ Test complete!
```

**Duration**: ~5-8 minutes (model download + inference)

**Cost**: ~$0.05 per test

**Why on EC2?**
- ✅ Fast inference with GPU
- ✅ Test different timestamped models from S3
- ✅ No need to download 16GB base model locally
- ✅ Same environment as training

## Next Steps

After successful testing:

1. ✅ Test scripts passed → Run full training
2. ✅ Training completed → Stop EC2 instance
3. ✅ Test model inference → Verify output quality
4. ✅ Push to HuggingFace → Model published
5. ✅ Model published → Celebrate! 🎉

**Total cost for one complete fine-tuning run**:
- Testing: $0.08
- Training: $2.25
- Model Testing: $0.05
- Publishing: $0.05
- **Total: ~$2.43**

Compare to: Always-on instance = $541/month! 💰
