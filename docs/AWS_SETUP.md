# AWS Setup Guide

## Prerequisites

Before running any scripts in this project, you need to configure AWS credentials on your **local machine**.

## AWS Credentials Setup

### Option 1: AWS CLI Configuration (Recommended)

```bash
# Install AWS CLI if not already installed
# macOS:
brew install awscli

# Linux:
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure credentials
aws configure

# You'll be prompted for:
# AWS Access Key ID [None]: AKIA...
# AWS Secret Access Key [None]: ...
# Default region name [None]: us-east-1
# Default output format [None]: json
```

This creates `~/.aws/credentials` and `~/.aws/config`:

```ini
# ~/.aws/credentials
[default]
aws_access_key_id = AKIA...
aws_secret_access_key = ...

# ~/.aws/config
[default]
region = us-east-1
output = json
```

### Option 2: Environment Variables

```bash
# Add to ~/.zshrc or ~/.bashrc
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

# Reload shell
source ~/.zshrc

# Verify
echo $AWS_ACCESS_KEY_ID
```

### Option 3: AWS SSO (For Organizations)

```bash
# Configure SSO profile
aws configure sso

# Login
aws sso login --profile my-profile

# Use profile
export AWS_PROFILE=my-profile

# Or specify per command
AWS_PROFILE=my-profile python scripts/setup/init_ssm_parameters.py
```

## Getting AWS Credentials

### 1. Create IAM User

```bash
# Via AWS Console:
# 1. Go to IAM → Users → Create User
# 2. User name: fine-tune-slm-admin
# 3. Enable "Programmatic access"
# 4. Attach policies:
#    - AmazonEC2FullAccess
#    - AmazonS3FullAccess
#    - AmazonSSMFullAccess
#    - SecretsManagerReadWrite
#    - AmazonECRFullAccess
#    - CloudWatchLogsFullAccess
# 5. Download access keys (only shown once!)
```

**⚠️ Security Best Practice**: Create a dedicated IAM user for this project, don't use root account credentials.

### 2. Required IAM Permissions

Minimum permissions needed:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameter",
        "ssm:GetParameters",
        "ssm:GetParametersByPath",
        "ssm:PutParameter",
        "ssm:DescribeParameters",
        "ec2:StartInstances",
        "ec2:StopInstances",
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus",
        "ssm:SendCommand",
        "ssm:GetCommandInvocation",
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket",
        "secretsmanager:GetSecretValue",
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Resource": "*"
    }
  ]
}
```

## Verify Configuration

```bash
# Test AWS CLI access
aws sts get-caller-identity

# Output should show:
# {
#     "UserId": "AIDA...",
#     "Account": "123456789012",
#     "Arn": "arn:aws:iam::123456789012:user/fine-tune-slm-admin"
# }

# Test SSM access
aws ssm describe-parameters --max-results 1

# Test S3 access
aws s3 ls
```

## Using Credentials in Scripts

All scripts in this project use **boto3**, which automatically discovers credentials in this order:

1. **Explicit credentials** (if passed to client)
2. **Environment variables** (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
3. **AWS credentials file** (`~/.aws/credentials`)
4. **AWS config file** (`~/.aws/config`)
5. **IAM role** (if running on EC2)
6. **Container credentials** (if running in ECS)

No code changes needed - just ensure credentials are configured!

## Security Best Practices

### ✅ DO
- Use IAM users with minimal required permissions
- Enable MFA on IAM users
- Rotate access keys regularly (every 90 days)
- Use AWS SSO for organizational accounts
- Store credentials in `~/.aws/credentials` (automatically secured)

### ❌ DON'T
- Use root account credentials
- Commit credentials to git
- Share credentials via email/chat
- Use same credentials across environments
- Grant `*:*` permissions

## Troubleshooting

### "Unable to locate credentials"

```bash
# Check if credentials are configured
aws configure list

# Output should show:
#       Name                    Value             Type    Location
#       ----                    -----             ----    --------
#    profile                <not set>             None    None
# access_key     ****************ABCD shared-credentials-file    
# secret_key     ****************WXYZ shared-credentials-file    
#     region                us-east-1      config-file    ~/.aws/config
```

### "Access Denied" errors

```bash
# Check which user/role you're using
aws sts get-caller-identity

# Verify IAM permissions in AWS Console
# IAM → Users → [your-user] → Permissions
```

### Multiple AWS accounts

```bash
# Use profiles for different accounts
aws configure --profile dev
aws configure --profile prod

# Use specific profile
AWS_PROFILE=dev python scripts/setup/init_ssm_parameters.py
```

## Next Steps

Once credentials are configured:

1. ✅ Run `python scripts/setup/init_ssm_parameters.py --dry-run` to preview
2. ✅ Create AWS resources (EC2, S3, etc.)
3. ✅ Initialize SSM parameters with `python scripts/setup/init_ssm_parameters.py`
4. ✅ Manually configure resource-specific parameters
5. ✅ Run training workflows!
