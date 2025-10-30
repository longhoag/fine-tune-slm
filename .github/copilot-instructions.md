# AI Agent Instructions for fine-tune-slm

## Project Overview
This project fine-tunes Llama 3.1 8B model using LoRA for medical cancer-specific information extraction (IE). The entire workflow runs remotely on AWS EC2 (g6.2xlarge instance) controlled via AWS SSM—**no SSH or .pem keys required**.

## Architecture & Data Flow

### Dataset Structure (`synthetic-instruction-tuning-dataset/`)
- **Training**: 4,500 entries in `train.jsonl`
- **Validation**: 500 entries in `validation.jsonl`
- **Status**: Dataset is ready for training—no preprocessing required
- **Schema**: Each entry contains `instruction`, `input`, and structured `output` with fields:
  - `cancer_type`, `stage`, `gene_mutation`, `biomarker`, `treatment`, `response`, `metastasis_site`
- **Example**:
  ```json
  {
    "instruction": "Extract all cancer-related entities from the text.",
    "input": "70-year-old man with widely metastatic cutaneous melanoma...",
    "output": {
      "cancer_type": "melanoma (cutaneous)",
      "stage": "IV",
      "biomarker": "PD-L1 5%; TMB-high",
      "treatment": "nivolumab and ipilimumab; stereotactic radiosurgery",
      "response": "mixed response",
      "metastasis_site": "brain"
    }
  }
  ```

### Remote Execution Model
All scripts execute **from local terminal → AWS SSM → EC2 instance**. This project uses:
- **AWS SSM** for remote command execution (no SSH/key management)
- **AWS Secrets Manager** for sensitive credentials (API tokens, access keys)
- **SSM Parameter Store** for ALL configuration values (resource IDs, settings, hyperparameters)
- **CloudWatch Logs** for SSM command output/session logging
- **EC2** (g6.2xlarge instance) for compute
- **EBS gp3** for active checkpoint writes during training (fast, low-latency)
- **S3** for final artifact storage and archival after training
- **ECR** for Docker image storage
- **Hugging Face Hub** for model retrieval and publishing
- **Failsafe measures** with error handling and precautionary checks

## Key Workflows

### CI/CD Pipeline (GitHub Actions)
**Fully automated**: Build Docker image → Push to ECR

### Setup Scripts (Manual from Local)
1. Start EC2 instance
2. Wait for instance status OK
3. Send SSM run command to deploy
4. CloudWatch logs SSM command output

### Fine-Tuning Scripts (Manual from Local)
1. Run fine-tune job → Save checkpoints to EBS gp3 volume during training
2. Copy final model artifacts to S3 for archival
3. Push trained model to Hugging Face Hub
4. Stop EC2 instance

## Project-Specific Conventions

### Configuration Management
- **SSM Parameter Store**: ALL AWS resource identifiers stored in SSM (not hardcoded in YAML)
- **Training hyperparameters**: Stored directly in `config/training_config.yml` (version controlled)
- **Config files**: Define SSM parameter paths with optional defaults for AWS resources
- **Pattern**: `ssm_param: /fine-tune-slm/<category>/<key>` + `default: <value>`
- **Initialize once**: Run `scripts/setup/init_ssm_parameters.py` to create all parameters
- **Environment flexibility**: Change AWS resources without code changes (dev/staging/prod)

### Dependency Management
- **Local development**: Use Poetry for Python dependency management
- **No `.env` files**: Use AWS Secrets Manager for credentials, SSM Parameter Store for configs

### Logging
- **Use `loguru` logger** instead of print statements for all local scripts
- CloudWatch captures remote execution logs automatically

### Required Credentials
Set up in AWS Secrets Manager:
- Hugging Face access token (retrieve Llama 3.1 8B)
- AWS access key + secret access key
- Docker Hub token
- ECR credentials
- Additional credentials may be added as needed during implementation

### EC2 Instance Selection
- **Target**: g6.2xlarge instance
- Balance cost-efficiency with performance for LoRA fine-tuning

## Critical Design Decisions

1. **Remote-first execution**: SSM removes SSH/key management complexity
2. **Secrets centralization**: Secrets Manager + Parameter Store integration prevents credential sprawl
3. **LoRA technique**: Efficient fine-tuning approach for 8B parameter model
4. **Medical IE domain**: Highly structured output schema requires careful prompt engineering

## Next Steps (Planned Implementation)
- [ ] Run `scripts/setup/init_ssm_parameters.py --dry-run` to preview SSM parameters
- [ ] Set up AWS resources (EC2, EBS, S3, ECR) and update SSM parameters with IDs
- [ ] Build GitHub Actions workflow for Docker → ECR
- [ ] Create SSM run command scripts with error handling
- [ ] Implement fine-tuning scripts with CloudWatch logging integration

## When Adding Code
- Wrap remote operations in try/catch with CloudWatch logging
- Reference AWS resource IDs via SSM Parameter Store (avoid hardcoding)
- Training hyperparameters can be directly in YAML (version controlled for reproducibility)
- Test SSM commands locally with `aws ssm send-command` before automation
- Validate JSONL dataset schema changes against existing 5,000 entries
- Use `scripts/setup/init_ssm_parameters.py` to bootstrap new SSM parameters for AWS resources
