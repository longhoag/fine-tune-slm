# Quick Reference: Adding Credentials to AWS Secrets Manager

This is a quick reference for adding credentials to AWS Secrets Manager after you've obtained them from external services.

## 📋 Prerequisites

✅ AWS CLI configured (`aws configure` completed)  
✅ You have the credentials to store (HF token, Docker token, etc.)

## 🚀 Quick Setup (AWS CLI Method)

### 1. Store Hugging Face Token

```bash
aws secretsmanager create-secret \
    --name huggingface/api-token \
    --description "Hugging Face API token for Llama 3.1 access" \
    --secret-string "hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890"
    #                 ↑ Replace with your actual token
```

### 2. Store Docker Hub Credentials

```bash
aws secretsmanager create-secret \
    --name docker/hub-token \
    --description "Docker Hub credentials for image push" \
    --secret-string '{
        "username": "yourDockerHubUsername",
        "token": "dckr_pat_1234567890abcdefghijklmnopqrstuvwxyz"
    }'
    # ↑ Replace with your actual username and token
```

### 3. Store AWS Credentials (for GitHub Actions)

```bash
aws secretsmanager create-secret \
    --name aws/credentials \
    --description "AWS credentials for GitHub Actions" \
    --secret-string '{
        "access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    }'
    # ↑ Replace with your IAM user access keys
```

### 4. Verify All Secrets

```bash
aws secretsmanager list-secrets --query 'SecretList[*].Name' --output table
```

**Expected output:**
```
-----------------------
|    ListSecrets     |
+---------------------+
|  aws/credentials   |
|  docker/hub-token  |
|  huggingface/api-token |
+---------------------+
```

## 🔍 Retrieving Secrets (for verification)

```bash
# Retrieve Hugging Face token
aws secretsmanager get-secret-value \
    --secret-id huggingface/api-token \
    --query SecretString \
    --output text

# Retrieve Docker Hub credentials
aws secretsmanager get-secret-value \
    --secret-id docker/hub-token \
    --query SecretString \
    --output text

# Retrieve AWS credentials
aws secretsmanager get-secret-value \
    --secret-id aws/credentials \
    --query SecretString \
    --output text
```

## 🔄 Updating Secrets (if you need to change them)

```bash
# Update Hugging Face token
aws secretsmanager update-secret \
    --secret-id huggingface/api-token \
    --secret-string "hf_NEW_TOKEN_HERE"

# Update Docker Hub token
aws secretsmanager update-secret \
    --secret-id docker/hub-token \
    --secret-string '{
        "username": "yourDockerHubUsername",
        "token": "dckr_pat_NEW_TOKEN_HERE"
    }'
```

## ❌ Deleting Secrets (if needed)

```bash
# Delete a secret (allows 7-30 day recovery window)
aws secretsmanager delete-secret \
    --secret-id huggingface/api-token \
    --recovery-window-in-days 7

# Force delete immediately (no recovery)
aws secretsmanager delete-secret \
    --secret-id huggingface/api-token \
    --force-delete-without-recovery
```

## 🔐 Security Best Practices

### ✅ DO
- Use single quotes around secret values to avoid shell interpretation
- Store JSON secrets for structured data (username + token pairs)
- Use descriptive names with forward slashes for organization
- Verify secrets after creation

### ❌ DON'T
- Echo secrets in terminal (they appear in history)
- Store secrets in plain text files
- Commit secrets to git
- Share secrets via email/chat

## 🐛 Troubleshooting

### Error: "Secret already exists"

```bash
# If you get "ResourceExistsException", update instead:
aws secretsmanager update-secret \
    --secret-id huggingface/api-token \
    --secret-string "hf_YOUR_TOKEN"
```

### Error: "Access Denied"

```bash
# Check your IAM permissions
aws sts get-caller-identity

# You need these permissions:
# - secretsmanager:CreateSecret
# - secretsmanager:GetSecretValue
# - secretsmanager:UpdateSecret
```

### Error: "Invalid JSON"

```bash
# For JSON secrets, validate first:
echo '{
    "username": "test",
    "token": "abc123"
}' | jq .

# If jq shows no errors, it's valid JSON
```

## 📝 Complete Example Workflow

```bash
# 1. Get your Hugging Face token from https://huggingface.co/settings/tokens
HF_TOKEN="hf_YOUR_ACTUAL_TOKEN_HERE"

# 2. Store it
aws secretsmanager create-secret \
    --name huggingface/api-token \
    --secret-string "$HF_TOKEN"

# 3. Verify it was stored
aws secretsmanager get-secret-value \
    --secret-id huggingface/api-token \
    --query SecretString \
    --output text

# 4. Clear the variable from memory
unset HF_TOKEN
```

## 🔗 Related Documentation

- [AWS Secrets Manager Console Method](./COMPLETE_SETUP_GUIDE.md#6-aws-secrets-manager-setup)
- [AWS Secrets Manager Documentation](https://docs.aws.amazon.com/secretsmanager/)
- [AWS CLI Secrets Manager Commands](https://docs.aws.amazon.com/cli/latest/reference/secretsmanager/)

## ⏱️ Time Estimate

- **CLI Method**: ~2 minutes for all 3 secrets
- **Console Method**: ~5-10 minutes for all 3 secrets

Both methods are equally secure - choose based on your preference!
