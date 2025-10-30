# SSM Parameter Store Configuration Guide

## Overview

This project uses AWS SSM Parameter Store for **AWS resource identifiers** (EC2, S3, ECR, etc.), while training hyperparameters are stored directly in `config/training_config.yml` for version control and reproducibility.

## Architecture

```
config/aws_config.yml (in git)
    ↓ defines SSM parameter schema
SSM Parameter Store (in AWS)
    ↓ stores actual AWS resource IDs
Application Runtime
    ↓ resolves parameters

config/training_config.yml (in git)
    ↓ version controlled hyperparameters
Application Runtime
    ↓ loads directly
```

## What Goes Where

### ✅ SSM Parameter Store
- **EC2 Instance IDs** - Environment-specific
- **S3 Bucket Names** - Different per environment
- **ECR Registry URLs** - AWS account-specific
- **EBS Volume IDs** - Environment-specific
- **HuggingFace Repo Names** - Different per environment
- **Secret Names** - References to Secrets Manager

**Why?** These values change between dev/staging/prod and shouldn't be hardcoded.

### ✅ YAML Files (Version Controlled)
- **Model Names** - Reproducibility
- **LoRA Hyperparameters** - r, alpha, dropout
- **Training Settings** - Learning rate, batch size, epochs
- **Quantization Config** - 4-bit settings
- **Dataset Paths** - Container paths

**Why?** These define the training experiment and should be versioned for reproducibility.

## Configuration Pattern

Each configuration value in YAML files follows this pattern:

```yaml
parameter_name:
  ssm_param: /fine-tune-slm/category/key  # SSM parameter path
  default: value                           # Optional fallback value
  description: "Human-readable description" # Optional documentation
```

### Example

```yaml
ec2:
  instance_id:
    ssm_param: /fine-tune-slm/ec2/instance-id
    description: "EC2 instance ID for training"
```

## Setup Workflow

### Prerequisites

**Configure AWS credentials first!** See [AWS_SETUP.md](./AWS_SETUP.md) for detailed instructions.

Quick setup:
```bash
aws configure  # Enter your AWS access key, secret key, and region
```

### 1. Initial Setup (One-time)

```bash
# Preview what parameters will be created
python scripts/setup/init_ssm_parameters.py --dry-run

# Create all SSM parameters with default values
python scripts/setup/init_ssm_parameters.py
```

This will:
- ✅ Create parameters with `default` values
- ⚠️ Skip parameters without defaults (require manual setup)
- 📋 List all parameters needing manual configuration

### 2. Manual Configuration

For parameters without defaults (like instance IDs, bucket names):

```bash
# Using AWS CLI
aws ssm put-parameter \
  --name /fine-tune-slm/ec2/instance-id \
  --value i-1234567890abcdef0 \
  --type String \
  --description "EC2 instance for training"

# Using AWS Console
# Navigate to Systems Manager > Parameter Store > Create parameter
```

### 3. Runtime Resolution

The `Config` class automatically resolves SSM parameters:

```python
from src.utils.config import load_config

# Load config with SSM resolution
config = load_config(config_dir="config", ssm_client=ssm_client)

# Get resolved value
instance_id = config.get('aws.ec2.instance_id')  # Returns actual value from SSM
```

## Parameter Hierarchy

All parameters use the prefix: `/fine-tune-slm/`

```
/fine-tune-slm/
├── aws/
│   └── region
├── ec2/
│   ├── instance-id          # Environment-specific
│   └── instance-type
├── ebs/
│   ├── volume-id            # Environment-specific
│   ├── mount-path
│   ├── volume-type
│   └── size-gb
├── s3/
│   ├── bucket               # Environment-specific
│   └── prefix
├── ecr/
│   ├── repository
│   └── registry             # Environment-specific
├── secrets/
│   ├── hf-token-name        # Reference to Secrets Manager
│   ├── aws-credentials-name
│   └── docker-token-name
├── cloudwatch/
│   ├── log-group
│   └── log-stream-prefix
└── output/
    └── hf-repo              # Environment-specific
```

**Note:** Training hyperparameters (model name, LoRA settings, learning rate, etc.) are NOT in SSM - they're in `config/training_config.yml`.

## Benefits

### 🔄 Environment Flexibility
- Same codebase for dev/staging/prod
- Change AWS resources without code commits
- Point to different S3 buckets/EC2 instances per environment

### 📊 Training Reproducibility
- Hyperparameters in version control (git)
- Track experiment configurations
- Easy to compare different training runs

### 🔒 Security
- Credentials in Secrets Manager (encrypted)
- Resource IDs in Parameter Store (audit logged)
- No secrets in git

### � Best of Both Worlds
- Infrastructure flexibility (SSM)
- Experiment reproducibility (git)

## Common Operations

### View All Parameters

```bash
aws ssm get-parameters-by-path \
  --path /fine-tune-slm/ \
  --recursive
```

### Update Parameter

```bash
aws ssm put-parameter \
  --name /fine-tune-slm/training/learning-rate \
  --value 1.0e-4 \
  --overwrite
```

### Delete Parameter

```bash
aws ssm delete-parameter \
  --name /fine-tune-slm/training/learning-rate
```

### Get Parameter History

```bash
aws ssm get-parameter-history \
  --name /fine-tune-slm/ec2/instance-id
```

## Best Practices

1. **Always use defaults for non-sensitive values** - Makes initialization easier
2. **Use descriptive parameter names** - Follow the hierarchy pattern
3. **Document required manual setup** - Add to `description` field
4. **Tag parameters** - Use `Project: fine-tune-slm` tag
5. **Version control schema only** - Keep actual values out of git
6. **Test with --dry-run first** - Preview before creating

## Troubleshooting

### Parameter Not Found
```python
# Config.get() returns None if parameter doesn't exist and no default
value = config.get('aws.ec2.instance_id', default='fallback-value')
```

### Permission Denied
Ensure IAM role/user has:
- `ssm:GetParameter`
- `ssm:GetParameters`
- `ssm:GetParametersByPath`
- `ssm:PutParameter` (for setup)

### Secrets Manager Integration
For sensitive values, store in Secrets Manager and reference via SSM:
```bash
# Create secret in Secrets Manager
aws secretsmanager create-secret \
  --name huggingface/api-token \
  --secret-string "hf_xxxxx"

# Reference in SSM (optional, for consistency)
aws ssm put-parameter \
  --name /fine-tune-slm/secrets/hf-token-name \
  --value huggingface/api-token \
  --type String
```
