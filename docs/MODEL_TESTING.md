# Model Testing Guide

Quick reference for testing your fine-tuned medical IE model.

## Prerequisites

Before testing, you need:

1. **Rebuild Docker image** (if `src/test_model.py` was recently added/updated):
   ```bash
   # Trigger GitHub Actions workflow to rebuild and push to ECR
   git push origin main
   # Or build locally and push manually (see .github/workflows/build-and-push-ecr.yml)
   ```

2. **Start EC2 instance:**
   ```bash
   poetry run python scripts/setup/start_ec2.py
   ```

3. **Deploy environment** (pulls latest Docker image):
   ```bash
   poetry run python scripts/setup/deploy_via_ssm.py
   ```

4. **Run test:**
   ```bash
   poetry run python scripts/finetune/run_test_model_ec2.py
   ```

5. **Stop EC2 when done:**
   ```bash
   poetry run python scripts/setup/stop_ec2.py
   ```

**Note**: Steps 1 and 3 are only needed when testing code changes. For routine testing of different models, just start EC2 and run tests.

## Overview

The `run_test_model_ec2.py` script:
- ✅ Sends simple Docker command via SSM
- ✅ Runs `src.test_model` inside Docker container
- ✅ Downloads models from S3 (test any timestamped version)
- ✅ Loads base Llama 3.1 8B + your LoRA adapters on GPU
- ✅ Tests with medical text (sample or custom)
- ✅ Shows structured entity extraction
- ✅ Clean, maintainable code (no embedded scripts)

**EC2 must be running before use** - Start with `scripts/setup/start_ec2.py`

**Remember to stop EC2 after testing** - Stop with `scripts/setup/stop_ec2.py`

## Usage Examples

### 1. Test Latest Model

```bash
poetry run python scripts/finetune/run_test_model_ec2.py
```

Uses built-in sample medical text (melanoma case).

### 2. Test Specific Model Version

```bash
# List available models
poetry run python scripts/finetune/run_test_model_ec2.py --list

# Test specific timestamp
poetry run python scripts/finetune/run_test_model_ec2.py --timestamp 20251111_022951
```

### 3. Test with Custom Input

```bash
poetry run python scripts/finetune/run_test_model_ec2.py --input "62-year-old woman with stage IIIA NSCLC. EGFR exon 19 deletion detected. Started on osimertinib with good response..."
```

### 4. Test Different Sample Cases

The script includes 3 built-in medical cases:

```bash
# Sample 0: Melanoma with brain metastases (default)
poetry run python scripts/finetune/run_test_model_ec2.py --sample-index 0

# Sample 1: NSCLC with EGFR mutation
poetry run python scripts/finetune/run_test_model_ec2.py --sample-index 1

# Sample 2: Colorectal cancer with liver mets
poetry run python scripts/finetune/run_test_model_ec2.py --sample-index 2
```

### 5. Multiple Tests (Keep EC2 Running)

```bash
# Start EC2 once
poetry run python scripts/setup/start_ec2.py

# Run multiple tests
poetry run python scripts/finetune/run_test_model_ec2.py --sample-index 0
poetry run python scripts/finetune/run_test_model_ec2.py --sample-index 1
poetry run python scripts/finetune/run_test_model_ec2.py --timestamp 20251110_123456

# Stop EC2 when done
poetry run python scripts/setup/stop_ec2.py
```

## Sample Medical Texts

### Sample 0: Metastatic Melanoma
```
70-year-old man with widely metastatic cutaneous melanoma involving the brain, 
liver, and bones. Biopsy showed BRAF V600E mutation. PD-L1 expression was 5% 
and tumor mutational burden was high. Started on combination nivolumab and 
ipilimumab. After 3 months, showed mixed response with some lesions shrinking 
while brain metastases progressed. Underwent stereotactic radiosurgery for 
brain lesions.
```

**Expected entities:**
- Cancer type: melanoma (cutaneous)
- Stage: IV
- Gene mutation: BRAF V600E
- Biomarker: PD-L1 5%; TMB-high
- Treatment: nivolumab and ipilimumab; stereotactic radiosurgery
- Response: mixed response
- Metastasis site: brain, liver, bones

### Sample 1: NSCLC with EGFR Mutation
```
62-year-old woman diagnosed with stage IIIA non-small cell lung cancer (NSCLC). 
Molecular testing revealed EGFR exon 19 deletion. PD-L1 TPS 80%. Received 
osimertinib as first-line therapy with excellent initial response. After 18 
months, progression noted with new liver metastases. Next-generation sequencing 
detected EGFR T790M resistance mutation.
```

**Expected entities:**
- Cancer type: non-small cell lung cancer (NSCLC)
- Stage: IIIA
- Gene mutation: EGFR exon 19 deletion; EGFR T790M
- Biomarker: PD-L1 TPS 80%
- Treatment: osimertinib
- Response: excellent initial response; progression at 18 months
- Metastasis site: liver

### Sample 2: Metastatic Colorectal Cancer
```
55-year-old male with metastatic colorectal cancer to liver and lungs. 
KRAS wild-type, MSI-stable. Started FOLFOX plus bevacizumab. CEA levels 
initially 450 ng/mL, decreased to 15 ng/mL after 6 cycles. CT showed 
partial response in liver but stable disease in lungs. Continued treatment 
with good tolerance.
```

**Expected entities:**
- Cancer type: colorectal cancer
- Stage: IV (metastatic)
- Gene mutation: KRAS wild-type
- Biomarker: MSI-stable; CEA 450→15 ng/mL
- Treatment: FOLFOX plus bevacizumab
- Response: partial response (liver); stable disease (lungs)
- Metastasis site: liver, lungs

## Output Format

The model extracts structured JSON:

```json
{
  "cancer_type": "string or null",
  "stage": "string or null",
  "gene_mutation": "string or null",
  "biomarker": "string or null",
  "treatment": "string or null",
  "response": "string or null",
  "metastasis_site": "string or null"
}
```

## Typical Test Output

```
================================================================================
🧪 Testing Fine-Tuned Model on EC2
================================================================================
Model: fine-tune-llama-models-longhoang/models/llama-3.1-8b-medical-ie/20251111_022951
Base: meta-llama/Meta-Llama-3.1-8B

📦 Checking dependencies...
✅ Dependencies ready

📥 Downloading model from S3...
✅ Downloaded 7 files to /tmp/model_test_xyz

🔄 Loading base model...
  Model: meta-llama/Meta-Llama-3.1-8B
✅ Base model loaded

🔄 Loading LoRA adapters...
✅ LoRA adapters loaded

================================================================================
📝 INPUT:
================================================================================
70-year-old man with widely metastatic cutaneous melanoma...

================================================================================
🤖 RUNNING INFERENCE...
================================================================================

================================================================================
📤 OUTPUT:
================================================================================
{
  "cancer_type": "melanoma (cutaneous)",
  "stage": "IV",
  "gene_mutation": "BRAF V600E",
  "biomarker": "PD-L1 5%; TMB-high",
  "treatment": "nivolumab and ipilimumab; stereotactic radiosurgery",
  "response": "mixed response",
  "metastasis_site": "brain, liver, bones"
}

================================================================================
✅ Test complete!
================================================================================
🧹 Cleaned up temporary files
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Model size | ~270MB (LoRA adapters) |
| Base model | 16GB (Llama 3.1 8B) |
| Download time | ~30 seconds |
| Model loading | ~2 minutes |
| Inference time | ~5-10 seconds per sample |
| Total test time | ~5-8 minutes |
| Cost per test | ~$0.05 |

## Workflow Integration

### After Training
```bash
# 1. Training complete, model in S3
poetry run python scripts/finetune/run_training.py
# → Model at s3://bucket/prefix/20251111_022951/final_model
# → EC2 auto-stopped after training

# 2. Start EC2 to test model
poetry run python scripts/setup/start_ec2.py

# 3. Test the model
poetry run python scripts/finetune/run_test_model_ec2.py
# → Verify extraction quality

# 4. Stop EC2
poetry run python scripts/setup/stop_ec2.py

# 5. If good, push to HuggingFace (runs locally, no EC2 needed)
poetry run python scripts/finetune/push_to_hf.py
# → Published at huggingface.co/loghoag/llama-3.1-8b-medical-ie
```

### Compare Multiple Training Runs
```bash
# Start EC2 once
poetry run python scripts/setup/start_ec2.py

# List all trained models
poetry run python scripts/finetune/run_test_model_ec2.py --list

# Test older version
poetry run python scripts/finetune/run_test_model_ec2.py --timestamp 20251110_120000

# Test newer version
poetry run python scripts/finetune/run_test_model_ec2.py --timestamp 20251111_022951

# Compare outputs, stop EC2
poetry run python scripts/setup/stop_ec2.py
```

## Cost Optimization

| Scenario | Strategy | Cost |
|----------|----------|------|
| Single test | Start → Test → Stop | $0.05 |
| Multiple tests (3) | Start → Test 3x → Stop | $0.10 |
| Forgot to stop | 1 hour idle | $0.75 😱 |

**Best practice**: Start EC2, run all your tests, then stop:
```bash
poetry run python scripts/setup/start_ec2.py

# Run all tests
poetry run python scripts/finetune/run_test_model_ec2.py --sample-index 0
poetry run python scripts/finetune/run_test_model_ec2.py --sample-index 1
poetry run python scripts/finetune/run_test_model_ec2.py --sample-index 2

# Stop immediately when done
poetry run python scripts/setup/stop_ec2.py
```

**Remember**: EC2 charges by the second, so stop as soon as you're done testing!

## Troubleshooting

### Issue: "EC2 instance is 'stopped', not running"
**Solution**: Start the instance first:
```bash
poetry run python scripts/setup/start_ec2.py
```

### Issue: "No trained models found in S3"
**Solution**: Verify training completed and uploaded to S3:
```bash
aws s3 ls s3://your-bucket/models/llama-3.1-8b-medical-ie/
```

### Issue: "Model files not found"
**Solution**: Check S3 structure:
```bash
aws s3 ls s3://your-bucket/models/llama-3.1-8b-medical-ie/20251111_022951/final_model/
```
Should see: adapter_config.json, adapter_model.safetensors, tokenizer files

### Issue: "CUDA out of memory"
**Solution**: EC2 instance type (g6.2xlarge) has 24GB VRAM, should be sufficient.
Check if other processes are using GPU:
```bash
# Via SSM
aws ssm start-session --target i-xxxxx
nvidia-smi
```

### Issue: Test hangs or takes too long
**Solution**: Check CloudWatch logs:
```bash
aws logs tail /aws/ssm/fine-tune-llama --follow
```

### Issue: JSON parsing fails
**Cause**: Model output might not be properly formatted JSON.

**Solutions**:
1. Check if model needs more training steps
2. Adjust temperature (lower = more consistent format)
3. Review training data format
4. Inspect raw output (shown before parsed JSON)

## Advanced: Custom Test Scripts

You can modify the test script for custom workflows:

```python
# Example: Batch testing on validation set
test_cases = [
    "Case 1 text...",
    "Case 2 text...",
    "Case 3 text...",
]

for i, case in enumerate(test_cases):
    print(f"\nTest {i+1}:")
    result = model.generate(...)
    print(result)
```

## Next Steps

After successful testing:
1. ✅ Verify extraction quality
2. ✅ Test edge cases (missing fields, complex cases)
3. ✅ Compare with baseline/previous versions
4. ✅ Push best model to HuggingFace
5. ✅ Document model performance

See also:
- [Testing Guide](TESTING_GUIDE.md) - Complete testing workflow
- [Training Guide](../README.md) - Full training pipeline
- [Push to HuggingFace](../scripts/finetune/push_to_hf.py) - Publishing models
