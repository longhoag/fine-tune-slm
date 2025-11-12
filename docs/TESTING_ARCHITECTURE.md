# Testing Architecture

## Overview

Model testing uses a clean, maintainable architecture with the test logic in the Docker image.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Local Machine                                                   │
│                                                                 │
│  scripts/finetune/run_test_model_ec2.py                            │
│    │                                                            │
│    ├─ Verifies EC2 is running                                  │
│    ├─ Lists models from S3                                     │
│    ├─ Builds Docker command                                    │
│    └─ Sends via SSM                                            │
└────────────────────┬────────────────────────────────────────────┘
                     │ SSM Command
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ EC2 Instance                                                    │
│                                                                 │
│  Docker Container (ECR Image)                                  │
│    │                                                            │
│    └─ src/test_model.py                                        │
│         │                                                       │
│         ├─ Downloads model from S3                             │
│         ├─ Loads base model (Llama 3.1 8B)                     │
│         ├─ Loads LoRA adapters                                 │
│         ├─ Runs inference on GPU                               │
│         └─ Outputs results                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. `scripts/finetune/run_test_model_ec2.py` (Local)

**Purpose**: Orchestrate testing from local machine

**Responsibilities**:
- Verify EC2 instance is running
- List available models in S3
- Select model to test (by timestamp)
- Build simple Docker command
- Send command via SSM
- Display results

**SSM Command Example**:
```bash
docker run --rm --gpus all \
  -v /mnt/training:/mnt/training \
  -v /home/ubuntu/fine-tune-slm:/workspace \
  -w /workspace \
  855386719590.dkr.ecr.us-east-1.amazonaws.com/fine-tune-llama:latest \
  python -m src.test_model \
  --use-ssm \
  --timestamp 20251111_022951 \
  --sample-index 0
```

**Benefits**:
- ✅ Clean, simple SSM commands
- ✅ No embedded scripts or string escaping
- ✅ Easy to debug and maintain

### 2. `src/test_model.py` (In Docker Image)

**Purpose**: Run actual model testing on EC2

**Responsibilities**:
- Load configuration from SSM Parameter Store
- Download model from S3 (by timestamp)
- Load base model + LoRA adapters
- Run inference with sample or custom text
- Parse and display results
- Cleanup temporary files

**Arguments**:
- `--use-ssm`: Load config from SSM (required on EC2)
- `--timestamp`: Model timestamp to test (required)
- `--sample-index`: Which sample text (0-2)
- `--input`: Custom medical text (optional)

**Benefits**:
- ✅ Proper Python module, not embedded script
- ✅ Can import from `src.utils.*`
- ✅ Easy to test and debug
- ✅ Versioned with Docker image

## Workflow

### First-Time Setup

```bash
# 1. Add new code to src/test_model.py
git add src/test_model.py
git commit -m "Add model testing module"
git push origin main

# 2. GitHub Actions rebuilds Docker image
# (Automatic, takes ~5 minutes)

# 3. Deploy to EC2 (pulls latest image)
poetry run python scripts/setup/start_ec2.py
poetry run python scripts/setup/deploy_via_ssm.py

# 4. Test!
poetry run python scripts/finetune/run_test_model_ec2.py
```

### Routine Testing (No Code Changes)

```bash
# 1. Start EC2
poetry run python scripts/setup/start_ec2.py

# 2. Test (Docker image already cached)
poetry run python scripts/finetune/run_test_model_ec2.py

# 3. Stop EC2
poetry run python scripts/setup/stop_ec2.py
```

### Update Test Logic

```bash
# 1. Edit src/test_model.py locally
vim src/test_model.py

# 2. Commit and push
git add src/test_model.py
git commit -m "Update test logic"
git push origin main

# 3. Wait for GitHub Actions to rebuild (~5 min)

# 4. Re-deploy to EC2 (pulls new image)
poetry run python scripts/setup/deploy_via_ssm.py

# 5. Test with new logic
poetry run python scripts/finetune/run_test_model_ec2.py
```

## Comparison: Old vs New

### Old Approach (Embedded Script)

```python
# run_test_model_ec2.py
script = f'''#!/usr/bin/env python3
# ... 200 lines of embedded Python code ...
# ... string escaping issues ...
# ... hard to maintain ...
'''

commands = [
    f"cat > /tmp/test.py << 'EOF'\n{script}\nEOF",
    "python3 /tmp/test.py"
]
```

**Problems**:
- ❌ SSM command size limits (~48KB)
- ❌ String escaping complexity
- ❌ Hard to debug embedded code
- ❌ Can't import from `src.*`
- ❌ Code duplication

### New Approach (Docker Module)

```python
# run_test_model_ec2.py
commands = [
    f"docker run ... python -m src.test_model "
    f"--timestamp {timestamp} --sample-index {index}"
]
```

**Benefits**:
- ✅ Simple, clean SSM commands
- ✅ No size limits
- ✅ Easy to maintain
- ✅ Can import from `src.*`
- ✅ Follows project patterns

## File Structure

```
fine-tune-slm/
├── src/
│   ├── train.py              # Training module (runs in Docker)
│   ├── test_model.py         # Testing module (runs in Docker) ← NEW
│   └── utils/
│       ├── config.py
│       └── aws_helpers.py
├── scripts/
│   ├── finetune/
│   │   ├── run_training.py   # Local orchestrator for training
│   │   └── run_test_model_ec2.py # Local orchestrator for testing ← UPDATED
│   └── setup/
│       ├── start_ec2.py
│       ├── stop_ec2.py
│       └── deploy_via_ssm.py
├── docker/
│   └── Dockerfile            # Includes src/ directory
└── .github/
    └── workflows/
        └── build-and-push-ecr.yml  # Auto-builds on push
```

## Maintenance

### When to Rebuild Docker

You need to rebuild the Docker image when:
- ✅ `src/test_model.py` changes
- ✅ `src/train.py` changes
- ✅ `src/utils/*` changes
- ✅ `requirements.txt` changes
- ✅ `config/*` changes

**Automatic**: Pushing to `main` branch triggers GitHub Actions workflow

**Manual** (if needed):
```bash
cd docker
docker build -t fine-tune-llama:latest -f Dockerfile ..
docker tag fine-tune-llama:latest 855386719590.dkr.ecr.us-east-1.amazonaws.com/fine-tune-llama:latest
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 855386719590.dkr.ecr.us-east-1.amazonaws.com
docker push 855386719590.dkr.ecr.us-east-1.amazonaws.com/fine-tune-llama:latest
```

### When NOT to Rebuild

You don't need to rebuild when:
- ❌ `scripts/*` changes (local orchestration only)
- ❌ `docs/*` changes
- ❌ Testing different models/timestamps
- ❌ Changing SSM parameters

## Cost Implications

| Operation | Docker Rebuild? | EC2 Time | Cost |
|-----------|----------------|----------|------|
| First test | Yes (automatic) | ~7 min | $0.10 |
| Subsequent tests | No (cached) | ~5 min | $0.05 |
| Update test logic | Yes (automatic) | ~7 min | $0.10 |
| Test different model | No (cached) | ~5 min | $0.05 |

**Docker image caching**: EC2 instance caches the Docker image after first pull, making subsequent tests faster.

## Debugging

### Test Locally (Without EC2)

```python
# On your local machine with GPU
cd fine-tune-slm

# Mock SSM config with environment variables
export S3_BUCKET="fine-tune-llama-models-longhoang"
export S3_PREFIX="models/llama-3.1-8b-medical-ie"
export BASE_MODEL="meta-llama/Meta-Llama-3.1-8B"

# Run directly
python -m src.test_model \
  --timestamp 20251111_022951 \
  --sample-index 0
```

### View Docker Logs

```bash
# SSH into EC2
aws ssm start-session --target i-xxxxx

# View recent Docker containers
docker ps -a

# View logs from last run
docker logs <container-id>
```

### Test SSM Command Manually

```bash
# Get the exact SSM command
poetry run python scripts/finetune/run_test_model_ec2.py --list

# Copy the docker run command from output

# Send manually via SSM
aws ssm send-command \
  --instance-ids i-xxxxx \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["<docker command>"]'
```

## Benefits of This Architecture

1. **Separation of Concerns**
   - Local scripts orchestrate
   - Docker modules execute
   - Clear boundaries

2. **Maintainability**
   - Proper Python modules
   - Easy to test and debug
   - Version controlled with Git

3. **Consistency**
   - Follows same pattern as training
   - `src.train` for training
   - `src.test_model` for testing

4. **Flexibility**
   - Easy to add new test types
   - Can import shared utilities
   - Extensible architecture

5. **Developer Experience**
   - No string escaping hell
   - IDE support for `src/test_model.py`
   - Can use breakpoints/debugging
   - Unit testable

## Future Enhancements

Possible improvements:
- Add `src/evaluate.py` for batch evaluation
- Add `src/export_model.py` for model conversion
- Add `src/benchmark.py` for performance testing
- All following the same clean Docker-based pattern
