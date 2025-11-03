# Implementation Roadmap - Pre-EC2 Phase

This document tracks the implementation progress while waiting for EC2 quota approval.

---

## 📊 Progress Overview

- ✅ AWS Infrastructure Setup Complete
- ✅ GitHub Secrets Configured
- ✅ Docker Environment Implemented
- ✅ AWS Helpers Implemented
- ⏳ Waiting for EC2 Quota Approval
- 🔄 Code Implementation In Progress

---

## Phase 1: Verify AWS Infrastructure Setup ✅

**Status**: COMPLETED

**Goal**: Confirm all AWS services are properly configured

**Tasks Completed**:
- [x] S3 bucket created and accessible
- [x] ECR repository created
- [x] Secrets Manager has 3 secrets (HF token, Docker token, AWS credentials)
- [x] SSM Parameter Store has 17 parameters
- [x] CloudWatch log group created
- [x] IAM roles and policies configured

**Verification Commands**:
```bash
# Test S3
aws s3 ls
aws s3 ls s3://YOUR-BUCKET-NAME/

# Test ECR
aws ecr describe-repositories --repository-names fine-tune-llama

# Test Secrets Manager
aws secretsmanager list-secrets --query 'SecretList[*].Name' --output table

# Test SSM Parameters
aws ssm get-parameters-by-path \
  --path /fine-tune-slm/ \
  --recursive \
  --query 'Parameters[*].[Name,Value]' \
  --output table

# Test CloudWatch
aws logs describe-log-groups \
  --log-group-name-prefix /aws/ssm/fine-tune-llama
```

**Next Steps**: None - Phase complete ✅

---

## Phase 2: Test GitHub Actions Workflow ✅

**Status**: COMPLETED

**Goal**: Verify CI/CD pipeline can build and push Docker images

**Prerequisites**:
- [x] GitHub Secrets configured:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `DOCKER_HUB_USERNAME`
  - `DOCKER_HUB_TOKEN`
- [x] `.github/workflows/build-and-push-ecr.yml` exists
- [x] Dockerfile complete

**Tasks**:
- [ ] Commit current changes to trigger workflow
- [ ] Monitor GitHub Actions execution
- [ ] Verify Docker image in ECR
- [ ] Fix any build errors

**Commands**:
```bash
# Commit and push to trigger workflow
git add .
git commit -m "feat: implement Docker environment and AWS helpers"
git push origin main

# Monitor in browser
# https://github.com/longhoag/fine-tune-slm/actions

# Verify image pushed
aws ecr list-images --repository-name fine-tune-llama
```

**Expected Result**: 
- GitHub Actions workflow completes successfully
- Docker image appears in ECR with `latest` tag

**Troubleshooting**:
- If workflow fails on AWS auth: Check GitHub Secrets are set correctly
- If build fails: Check Dockerfile syntax and dependencies
- If ECR push fails: Verify IAM permissions include ECR access

**Next Steps**: Proceed to Phase 3 after successful build

---

## Phase 3: Implement Docker Training Environment ✅

**Status**: COMPLETED

**Goal**: Complete Dockerfile with all ML dependencies

**Tasks Completed**:
- [x] Create `requirements.txt` with all Python packages
- [x] Update Dockerfile with:
  - PyTorch 2.1.2 with CUDA 12.1 support
  - Transformers, PEFT, Accelerate, BitsAndBytes
  - AWS CLI for S3/SSM operations
  - Loguru, TensorBoard, Weights & Biases
- [x] Set up directory structure in container
- [x] Configure entrypoint for training

**Files Modified**:
- `requirements.txt` - All Python dependencies
- `docker/Dockerfile` - Complete training environment

**Test Locally** (Optional - requires Docker Desktop):
```bash
# Build image locally
docker build -t fine-tune-llama:test -f docker/Dockerfile .

# Check image size
docker images fine-tune-llama:test

# Test image runs
docker run --rm fine-tune-llama:test --help
```

**Next Steps**: Trigger GitHub Actions to build and push to ECR

---

## Phase 4: Implement Core Utility Modules ✅

**Status**: COMPLETED

**Goal**: Complete helper modules for AWS and configuration

**Tasks Completed**:
- [x] `src/utils/logger.py` - Already complete with Loguru setup
- [x] `src/utils/aws_helpers.py` - Implemented with boto3:
  - `AWSClient` - Base client initialization
  - `EC2Manager` - Start/stop instances, get status
  - `SSMManager` - Send commands, get output, wait for completion
  - `S3Manager` - Upload/download directories, verify artifacts
  - `SecretsManager` - Get secrets, parameters, parameter paths

**Features Implemented**:
- Error handling with try/except blocks
- CloudWatch logging integration
- Waiter patterns for async AWS operations
- Pagination for large result sets
- Timeout and polling mechanisms

**Files Modified**:
- `src/utils/aws_helpers.py` - Complete implementation (500+ lines)

**Next Steps**: Implement config.py for SSM parameter resolution

---

## Phase 5: Implement Configuration Loader ✅

**Status**: COMPLETED

**Goal**: Complete config.py with SSM parameter resolution.Implement src/utils/config.py next - it loads YAML configs and resolves SSM parameters

**Tasks**:
- [ ] Implement `ConfigLoader` class
- [ ] Add `_resolve_ssm_value()` method
- [ ] Add `_resolve_nested_config()` recursive resolution
- [ ] Add caching for resolved values
- [ ] Add `get()` and `get_all_resolved()` methods
- [ ] Support fallback to default values

**File to Update**:
- `src/utils/config.py`

**Implementation Pattern**:
```python
config = ConfigLoader('config/aws_config.yml')
instance_id = config.get('ec2.instance_id')  # Resolves SSM parameter
all_config = config.get_all_resolved()  # Returns fully resolved dict
```

**Test Locally**:
```bash
# Test config loading
python -c "
from src.utils.config import ConfigLoader
config = ConfigLoader('config/aws_config.yml')
print(config.get('aws.region'))
print(config.get('ec2.instance_id'))
"
```

**Expected Result**: 
- Configs load without errors
- SSM parameters resolve correctly
- Default values used when SSM param doesn't exist

**Next Steps**: Move to Phase 6 after implementation

---

## Phase 6: Implement Training Script Foundation ✅

**Status**: COMPLETED

**Goal**: Complete src/train.py with QLoRA (4-bit quantization + LoRA) fine-tuning logic

**Why QLoRA (4-bit + LoRA)?**
- **Memory efficiency**: 8B model fits in 24GB VRAM (~12GB used vs ~30GB without quantization)
- **Cost savings**: Works on g6.2xlarge ($0.75/hr) instead of needing g6.12xlarge ($3.00/hr)
- **Quality preserved**: <1% accuracy loss, LoRA adapters trained in full precision (FP16/BF16)
- **Proven technique**: Industry standard for efficient fine-tuning (thousands of models on HF Hub)

**VRAM Breakdown**:
```
With QLoRA (fits g6.2xlarge 24GB):
- Model (4-bit NF4):        ~4 GB
- LoRA adapters (FP16):     ~100 MB
- Gradients (LoRA only):    ~200 MB
- Optimizer states:         ~400 MB
- Activations/batch:        ~6 GB
- Total:                    ~11 GB ✅

Without quantization (needs 28-30GB):
- Model (FP16):             ~16 GB
- Gradients:                ~8 GB
- Optimizer states:         ~4 GB
- Activations/batch:        ~6 GB
- Total:                    ~34 GB ❌ Won't fit on g6.2xlarge!
```

**Tasks**:
- [ ] Implement dataset loading from JSONL
- [ ] Load Llama 3.1 8B model with 4-bit quantization (NF4)
- [ ] Configure LoRA parameters (r=16, alpha=32, target_modules for Llama)
- [ ] Set up Trainer with configuration
- [ ] Implement checkpoint saving to EBS mount
- [ ] Add CloudWatch logging integration
- [ ] Add dry-run mode for testing

**File to Update**:
- `src/train.py`

**Key Components**:
```python
# Dataset loading
def load_dataset(train_path, val_path):
    # Load JSONL files
    # Format for instruction tuning
    # Return datasets.Dataset objects

# Model loading (QLoRA approach)
def load_model_and_tokenizer(model_name, use_4bit=True):
    # Configure 4-bit quantization with BitsAndBytesConfig
    # - compute_dtype: bfloat16 (for computations)
    # - quant_type: nf4 (Normal Float 4-bit)
    # - use_nested_quant: True (double quantization)
    
    # Load model with quantization
    # Apply LoRA with PEFT
    # - r=16 (rank), alpha=32 (scaling)
    # - target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"] (attention layers)
    # - LoRA adapters stay in FP16 for quality
    
    # Return model, tokenizer, peft_config

# Training
def main():
    # Load configs
    # Initialize model with QLoRA
    # Set up Trainer
    # Train and save checkpoints
```

**Dependencies**:
- Requires Phase 5 (config.py) complete
- Requires HuggingFace token in Secrets Manager
- Requires dataset in `synthetic-instruction-tuning-dataset/`

**Next Steps**: Proceed to Phase 7 after implementation

---

## Phase 7: Local Testing (No GPU Required) ✅

**Status**: COMPLETED

**Goal**: Test configuration loading, dataset validation, and model initialization locally

**Prerequisites**:
- Phase 5 (config.py) complete
- Phase 6 (train.py) complete

**Tasks**:
- [ ] Test config loading without AWS credentials
- [ ] Validate JSONL dataset format
- [ ] Test model loading (without full initialization)
- [ ] Verify imports work correctly
- [ ] Test dry-run mode

**Test Commands**:
```bash
# Test config loading (uses default values)
python -c "
from src.utils.config import ConfigLoader
config = ConfigLoader('config/training_config.yml')
print(config.get_all_resolved())
"

# Test dataset loading
python -c "
from src.train import load_dataset
train_ds, val_ds = load_dataset(
    'synthetic-instruction-tuning-dataset/train.jsonl',
    'synthetic-instruction-tuning-dataset/validation.jsonl'
)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}')
print(train_ds[0])
"

# Test dry-run (won't actually train)
python -m src.train --dry-run --config config/training_config.yml
```

**Expected Results**:
- Configs load successfully
- Dataset has 4,500 train + 500 validation entries
- JSONL format validated
- No import errors
- Dry-run completes without errors

**Troubleshooting**:
- Missing dependencies: Check requirements.txt installed
- Config errors: Verify YAML syntax
- Dataset errors: Check JSONL format

**Next Steps**: Fix any issues, then proceed to Phase 8

---

## Phase 8: Prepare EC2 Setup Scripts 🔄

**Status**: PENDING

**Goal**: Complete automation scripts for EC2 lifecycle management

**Tasks**:
- [ ] Implement `scripts/setup/start_ec2.py`
  - Load instance ID from SSM
  - Start instance
  - Wait for status checks
  - Verify SSM connectivity
  
- [ ] Implement `scripts/setup/deploy_via_ssm.py`
  - Pull Docker image from ECR
  - Attach EBS volume
  - Mount EBS to /mnt/training
  - Set up environment
  
- [ ] Implement `scripts/setup/stop_ec2.py`
  - Save any pending data
  - Unmount EBS (optional - keep attached)
  - Stop instance
  - Verify stopped state

**Files to Update**:
- `scripts/setup/start_ec2.py`
- `scripts/setup/deploy_via_ssm.py`
- `scripts/setup/stop_ec2.py`

**Implementation Pattern**:
```python
# start_ec2.py
from src.utils.config import ConfigLoader
from src.utils.aws_helpers import AWSClient, EC2Manager

config = ConfigLoader('config/aws_config.yml')
client = AWSClient(region=config.get('aws.region'))
ec2_mgr = EC2Manager(client)

instance_id = config.get('ec2.instance_id')
ec2_mgr.start_instance(instance_id)
```

**Test** (after EC2 quota approved):
```bash
# Start instance
python scripts/setup/start_ec2.py

# Deploy
python scripts/setup/deploy_via_ssm.py

# Stop instance
python scripts/setup/stop_ec2.py
```

**Next Steps**: Ready for EC2 testing when quota approved

---

## Phase 9: Fine-Tuning Scripts 🔄

**Status**: PENDING

**Goal**: Complete training execution and model publishing scripts

**Tasks**:
- [ ] Implement `scripts/finetune/run_training.py`
  - Send training command via SSM
  - Monitor CloudWatch logs
  - Wait for completion
  - Copy checkpoints from EBS to S3
  
- [ ] Implement `scripts/finetune/push_to_hf.py`
  - Load model from S3
  - Merge LoRA weights
  - Push to Hugging Face Hub
  - Verify upload

**Files to Update**:
- `scripts/finetune/run_training.py`
- `scripts/finetune/push_to_hf.py`

**Workflow**:
```bash
# 1. Start instance
python scripts/setup/start_ec2.py

# 2. Deploy environment
python scripts/setup/deploy_via_ssm.py

# 3. Run training
python scripts/finetune/run_training.py

# 4. Push to Hugging Face
python scripts/finetune/push_to_hf.py

# 5. Stop instance
python scripts/setup/stop_ec2.py
```

**Next Steps**: Requires EC2 instance to test

---

## Phase 10: Document Setup Summary 📝

**Status**: PENDING

**Goal**: Create comprehensive reference document with all resource IDs

**Tasks**:
- [ ] Create `setup-summary.txt` with:
  - AWS Account ID
  - IAM User credentials location
  - EC2 Instance ID
  - EBS Volume ID
  - S3 Bucket name
  - ECR Registry URL
  - Hugging Face username and repo
  - All SSM parameter paths
  - GitHub repository URL

**File to Create**:
- `setup-summary.txt` (Add to .gitignore!)

**Template**:
```
=== fine-tune-slm Setup Summary ===
Date: 2025-10-30

AWS ACCOUNT
-----------
Account ID: [Your account ID]
IAM User: fine-tune-slm-admin
Region: us-east-1

EC2 (Waiting for quota approval)
-----------
Instance ID: [To be created]
Instance Type: g6.2xlarge
AMI: Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)
IAM Role: EC2-FineTune-Role

EBS (Waiting for quota approval)
-----------
Volume ID: [To be created]
Volume Type: gp3
Size: 100 GiB

S3
-----------
Bucket Name: [Your bucket name]
Prefix: models/llama-3.1-8b-medical-ie

ECR
-----------
Registry URL: [Your registry URL]
Repository: fine-tune-llama

GITHUB
-----------
Repository: https://github.com/longhoag/fine-tune-slm
Secrets configured: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, DOCKER_HUB_USERNAME, DOCKER_HUB_TOKEN

SSM PARAMETERS
-----------
17 parameters under /fine-tune-slm/
```

**Next Steps**: Update when EC2 instance created

---

## 🚦 Current Status & Next Actions

### ✅ Completed
1. AWS Infrastructure (S3, ECR, Secrets Manager, SSM, CloudWatch)
2. GitHub Secrets configured
3. Docker environment implemented
4. AWS helper utilities implemented

### 🔄 In Progress
- Waiting for EC2 quota approval
- Configuration loader implementation
- Training script foundation

### ⏭️ Next Immediate Actions

**Option 1: Continue Implementation** (Recommended while waiting)
```bash
# 1. Implement config.py
# 2. Test config loading locally
# 3. Implement basic train.py structure
# 4. Validate dataset loading
```

**Option 2: Test Current Setup**
```bash
# 1. Commit current code
git add .
git commit -m "feat: implement Docker and AWS helpers"
git push origin main

# 2. Verify GitHub Actions builds successfully

# 3. Check ECR for image
aws ecr list-images --repository-name fine-tune-llama
```

**Option 3: Check Quota Status**
```bash
# Check EC2 quota request status
aws service-quotas list-requested-service-quota-change-history \
  --region us-east-1 \
  --query 'RequestedQuotas[?QuotaCode==`L-DB2E81BA`].[Status,DesiredValue,Created]' \
  --output table
```

---

## 📋 Prerequisites Before EC2 Launch

- [x] AWS account configured
- [x] IAM user with necessary permissions
- [x] AWS CLI configured (~/.aws/credentials)
- [x] S3 bucket created
- [x] ECR repository created
- [x] Secrets Manager configured (3 secrets)
- [x] SSM Parameter Store configured (17 parameters)
- [x] CloudWatch log group created
- [x] GitHub Secrets configured
- [x] Docker environment complete
- [x] AWS helper utilities implemented
- [ ] EC2 quota approved (WAITING)
- [ ] EBS volume created (BLOCKED by EC2 quota)
- [ ] Config loader implemented
- [ ] Training script implemented
- [ ] Local testing complete

---

## 🎯 Success Criteria

### Phase 1-4 (Pre-EC2) ✅
- All AWS services accessible via CLI
- GitHub Actions can build and push Docker images
- Utility modules have no syntax errors
- Local imports work correctly

### Phase 5-7 (Code Implementation) 🔄
- Configs load without errors
- Dataset validates (4,500 + 500 entries)
- Model initialization works (at least in dry-run mode)
- No missing dependencies

### Phase 8-10 (EC2 Ready) ⏳
- EC2 instance starts/stops via scripts
- SSM commands execute successfully
- Training runs on EC2 with GPU
- Model artifacts save to S3
- Model publishes to Hugging Face

---

## 📚 Reference Documentation

- [COMPLETE_SETUP_GUIDE.md](./COMPLETE_SETUP_GUIDE.md) - Full AWS setup walkthrough
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Project architecture overview
- [SSM_PARAMETER_GUIDE.md](./SSM_PARAMETER_GUIDE.md) - SSM parameter usage
- [SECRETS_MANAGER_QUICKSTART.md](./SECRETS_MANAGER_QUICKSTART.md) - Credential storage

---

## 🐛 Troubleshooting

### Common Issues

**GitHub Actions fails to push to ECR**:
```bash
# Verify GitHub Secrets are set
# Check IAM user has ECR permissions
# Verify ECR repository exists
aws ecr describe-repositories --repository-names fine-tune-llama
```

**Config loading fails**:
```bash
# Check YAML syntax
python -m yaml config/aws_config.yml

# Verify SSM parameters exist
aws ssm get-parameters-by-path --path /fine-tune-slm/ --recursive
```

**Dataset loading fails**:
```bash
# Verify JSONL format
head -n 1 synthetic-instruction-tuning-dataset/train.jsonl | python -m json.tool

# Check file exists
ls -lh synthetic-instruction-tuning-dataset/
```

**AWS credentials not found**:
```bash
# Verify credentials file
cat ~/.aws/credentials

# Test AWS access
aws sts get-caller-identity
```

---

## 📅 Timeline Estimate

- **Phase 1-4**: ✅ Completed (Oct 30, 2025)
- **Phase 5-7**: ~2-3 hours implementation + testing
- **EC2 Quota Approval**: 15 minutes - 48 hours (AWS dependent)
- **Phase 8-10**: ~2 hours after EC2 available
- **First Training Run**: ~3-4 hours compute time

**Total**: Ready to train within 1-2 days after quota approval

---

*Last Updated: October 30, 2025*
*Status: Phases 1-4 Complete, Waiting for EC2 Quota*
