# Project Structure

This document describes the barebone architecture of the fine-tune-slm project.

## Directory Structure

```
fine-tune-slm/
├── .github/
│   ├── workflows/
│   │   └── build-and-push-ecr.yml    # CI/CD: Build Docker → Push to ECR
│   └── copilot-instructions.md       # AI agent guidance
│
├── config/
│   ├── aws_config.yml                # AWS resource configuration
│   └── training_config.yml           # LoRA training hyperparameters
│
├── docker/
│   ├── Dockerfile                    # Training environment image
│   ├── docker-compose.yml            # Local development setup
│   └── .dockerignore                 # Docker build exclusions
│
├── scripts/
│   ├── setup/
│   │   ├── start_ec2.py             # Start EC2 instance
│   │   ├── deploy_via_ssm.py        # Deploy training env via SSM
│   │   └── stop_ec2.py              # Stop EC2 after completion
│   │
│   └── finetune/
│       ├── run_training.py          # Execute training via SSM
│       └── push_to_hf.py            # Push model to HuggingFace Hub
│
├── src/
│   ├── __init__.py
│   ├── train.py                     # Main training script (runs in container)
│   └── utils/
│       ├── __init__.py
│       ├── aws_helpers.py           # AWS SDK wrappers (EC2, SSM, S3, Secrets)
│       ├── config.py                # Configuration loader/validator
│       └── logger.py                # Loguru setup and helpers
│
├── synthetic-instruction-tuning-dataset/
│   ├── train.jsonl                  # 4,500 training examples
│   └── validation.jsonl             # 500 validation examples
│
├── pyproject.toml                   # Poetry dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## Component Descriptions

### GitHub Actions (`.github/workflows/`)
- **build-and-push-ecr.yml**: Automated Docker image build and ECR push on code changes

### Configuration (`config/`)
- **aws_config.yml**: SSM parameter references for EC2, EBS, S3, ECR, Secrets Manager
- **training_config.yml**: Direct values for model, LoRA, training hyperparameters (version controlled)
- **Pattern**: AWS resources use `ssm_param` + optional `default`; training params are direct values
- **Initialization**: Run `scripts/setup/init_ssm_parameters.py` to create AWS resource parameters

### Docker (`docker/`)
- **Dockerfile**: CUDA-enabled training environment with PyTorch, Transformers, PEFT, AWS CLI
- **docker-compose.yml**: Local development/testing setup (not used in production)

### Setup Scripts (`scripts/setup/`)
Run from local terminal to manage EC2 lifecycle:
1. **init_ssm_parameters.py**: Initialize all SSM parameters (run once during setup)
2. **start_ec2.py**: Start instance → Wait for ready state
3. **deploy_via_ssm.py**: Pull Docker image → Mount EBS → Configure env via SSM
4. **stop_ec2.py**: Verify S3 artifacts → Stop instance

### Fine-tuning Scripts (`scripts/finetune/`)
Run from local terminal to execute training workflow:
1. **run_training.py**: Send training command via SSM → Monitor CloudWatch logs → Copy EBS checkpoints to S3
2. **push_to_hf.py**: Download from S3 → Push to HuggingFace Hub

### Training Code (`src/`)
- **train.py**: Main training script executed inside Docker container on EC2
- **utils/**: Reusable modules for AWS operations, config management, logging

## Data Flow

### Training Workflow
```
Local Terminal
    ↓ (start_ec2.py)
EC2 g6.2xlarge starts
    ↓ (deploy_via_ssm.py via SSM)
Pull Docker image from ECR
Mount EBS gp3 volume
    ↓ (run_training.py via SSM)
Execute training in container
Save checkpoints → EBS volume
    ↓ (on completion)
Copy final model → S3
    ↓ (push_to_hf.py)
Download from S3 → Push to HF Hub
    ↓ (stop_ec2.py)
Stop EC2 instance
```

### Secrets Management
```
AWS Secrets Manager (stores credentials)
    ↓ (integrated with)
SSM Parameter Store (stores all config + secret references)
    ↓ (accessed by)
Scripts/Container (retrieve at runtime)
```

### Configuration Philosophy
- **AWS resources in SSM**: Instance IDs, bucket names, registry URLs (environment-specific)
- **Training hyperparameters in git**: LoRA config, learning rates (version controlled for reproducibility)
- **Credentials in Secrets Manager**: API tokens, access keys (encrypted)
- **Audit trail**: All SSM parameter changes logged in CloudWatch

## Implementation Status

All files are **barebone templates** with:
- ✅ Complete directory structure
- ✅ Function signatures and docstrings
- ✅ Configuration schemas
- ✅ Workflow documentation
- ⏳ Implementation details marked with `# TODO:` comments

Ready for incremental implementation of each component.
