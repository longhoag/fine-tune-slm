# Complete Setup Guide for fine-tune-slm Project

This guide walks you through **every single setup step** needed for this project, from scratch. Follow each section in order.

---

## 📋 Table of Contents

1. [AWS Account Setup](#1-aws-account-setup)
2. [IAM User Creation & Access Keys](#2-iam-user-creation--access-keys)
3. [AWS CLI Installation & Configuration](#3-aws-cli-installation--configuration)
4. [Docker Hub Account & Token](#4-docker-hub-account--token)
5. [Hugging Face Account & Token](#5-hugging-face-account--token)
6. [AWS Secrets Manager Setup](#6-aws-secrets-manager-setup)
7. [EC2 Instance Setup](#7-ec2-instance-setup)
8. [EBS Volume Setup](#8-ebs-volume-setup)
9. [S3 Bucket Setup](#9-s3-bucket-setup)
10. [ECR Repository Setup](#10-ecr-repository-setup)
11. [CloudWatch Logs Setup](#11-cloudwatch-logs-setup)
12. [SSM Parameter Store Setup](#12-ssm-parameter-store-setup)
13. [Verification & Testing](#13-verification--testing)

---

## 1. AWS Account Setup

### Step 1.1: Create AWS Account (if you don't have one)

1. Go to https://aws.amazon.com/
2. Click **"Create an AWS Account"** (top right)
3. Enter:
   - **Email address**: Your email
   - **AWS account name**: `fine-tune-slm` (or your preferred name)
4. Click **"Verify email address"**
5. Check your email for verification code
6. Enter verification code
7. Create **root user password** (save this securely!)
8. Choose **Personal** account type
9. Enter your contact information
10. Enter payment information (credit/debit card)
    - ⚠️ **Note**: AWS Free Tier covers many services, but EC2 g6.2xlarge will cost ~$0.90/hour
11. Verify your phone number (SMS or voice call)
12. Select **Basic Support - Free** plan
13. Click **"Complete sign up"**

### Step 1.2: Sign in to AWS Console

1. Go to https://console.aws.amazon.com/
2. Select **"Root user"**
3. Enter your root email
4. Click **"Next"**
5. Enter root password
6. Click **"Sign in"**

You're now in the AWS Management Console! ✅

---

## 2. IAM User Creation & Access Keys

**⚠️ IMPORTANT**: Never use root account credentials for API access. Create an IAM user instead.

### Step 2.1: Navigate to IAM

1. In AWS Console, search bar (top): Type **"IAM"**
2. Click **"IAM"** (Identity and Access Management)

### Step 2.2: Create IAM User

1. Left sidebar: Click **"Users"**
2. Click **"Create user"** (orange button, top right)
3. **Step 1 - Specify user details**:
   - **User name**: `fine-tune-slm-admin`
   - Click **"Next"**

4. **Step 2 - Set permissions**:
   - Select **"Attach policies directly"**
   - Search and check these policies:
     - ☑️ `AmazonEC2FullAccess`
     - ☑️ `AmazonS3FullAccess`
     - ☑️ `AmazonSSMFullAccess`
     - ☑️ `SecretsManagerReadWrite`
     - ☑️ `AmazonEC2ContainerRegistryFullAccess`
     - ☑️ `CloudWatchLogsFullAccess`
   - Click **"Next"**

5. **Step 3 - Review and create**:
   - Review the user details
   - Click **"Create user"**

### Step 2.3: Create Access Keys

1. Click on the user you just created: **fine-tune-slm-admin**
2. Click **"Security credentials"** tab
3. Scroll down to **"Access keys"** section
4. Click **"Create access key"**
5. Select use case: **"Command Line Interface (CLI)"**
6. Check ☑️ **"I understand the above recommendation..."**
7. Click **"Next"**
8. Description tag (optional): `fine-tune-slm local CLI`
9. Click **"Create access key"**

### Step 2.4: Download and Save Access Keys

**🚨 CRITICAL: This is the ONLY time you'll see the secret access key!**

1. You'll see:
   - **Access key ID**: `AKIA...` (starts with AKIA)
   - **Secret access key**: `...` (long random string)

2. Click **"Download .csv file"** - Save this file securely!
3. **OR** Copy both values to a secure location (password manager recommended)

Example format:
```
Access key ID: AKIAIOSFODNN7EXAMPLE
Secret access key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

4. Click **"Done"**

**⚠️ Store these securely - treat them like passwords!**

---

## 3. AWS CLI Installation & Configuration

### Step 3.1: Install AWS CLI

**macOS:**
```bash
# Using Homebrew
brew install awscli

# Verify installation
aws --version
# Should show: aws-cli/2.x.x ...
```

**Linux:**
```bash
# Download installer
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

# Unzip
unzip awscliv2.zip

# Install
sudo ./aws/install

# Verify
aws --version
```

**Windows:**
1. Download: https://awscli.amazonaws.com/AWSCLIV2.msi
2. Run the installer
3. Open Command Prompt and verify: `aws --version`

### Step 3.2: Configure AWS CLI

```bash
aws configure
```

You'll be prompted for 4 values:

```
AWS Access Key ID [None]: <paste your AKIA... key>
AWS Secret Access Key [None]: <paste your secret key>
Default region name [None]: us-east-1
Default output format [None]: json
```

**Example:**
```
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: us-east-1
Default output format [None]: json
```

### Step 3.3: Verify Configuration

```bash
# Test connection
aws sts get-caller-identity
```

**Expected output:**
```json
{
    "UserId": "AIDAI...",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/fine-tune-slm-admin"
}
```

✅ If you see this, AWS CLI is configured correctly!

---

## 4. Docker Hub Account & Token

### Step 4.1: Create Docker Hub Account

1. Go to https://hub.docker.com/signup
2. Fill in:
   - **Docker ID**: Choose a username (e.g., `yourname`)
   - **Email**: Your email
   - **Password**: Create strong password
3. Check ☑️ **"I'm not a robot"**
4. Click **"Sign up"**
5. Verify your email (check inbox)
6. Click verification link in email

### Step 4.2: Sign In

1. Go to https://hub.docker.com/
2. Click **"Sign In"**
3. Enter Docker ID and password

### Step 4.3: Create Access Token

1. Click your username (top right)
2. Click **"Account Settings"**
3. Left sidebar: Click **"Security"**
4. Click **"New Access Token"**
5. Fill in:
   - **Access Token Description**: `fine-tune-slm-github-actions`
   - **Access permissions**: Select **"Read, Write, Delete"**
6. Click **"Generate"**

### Step 4.4: Copy and Save Token

**🚨 CRITICAL: You'll only see this token ONCE!**

1. You'll see a token like: `dckr_pat_...` (long random string)
2. Click **"Copy and Close"**
3. Save this in a secure location

Example:
```
Docker Hub Username: yourname
Docker Hub Token: dckr_pat_1234567890abcdefghijklmnopqrstuvwxyz
```

**⚠️ Save both your Docker Hub username AND token!**

---

## 5. Hugging Face Account & Token

### Step 5.1: Create Hugging Face Account

1. Go to https://huggingface.co/join
2. Fill in:
   - **Email**: Your email
   - **Username**: Choose username (e.g., `yourname`)
   - **Password**: Create strong password
3. Click **"Sign up"**
4. Verify your email (check inbox)
5. Click verification link

### Step 5.2: Sign In

1. Go to https://huggingface.co/login
2. Enter email and password
3. Click **"Sign in"**

### Step 5.3: Request Access to Llama 3.1

**Important**: Meta's Llama models require accepting their license.

1. Go to https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
2. You'll see **"Access Llama 3.1"** form
3. Fill in the form:
   - ☑️ Accept license terms
   - Select **"Research"** or **"Commercial"** use case
4. Click **"Submit"**
5. **Wait for approval** (usually takes a few hours to 1 day)
6. You'll receive email when approved

### Step 5.4: Create Access Token

1. Click your profile picture (top right)
2. Click **"Settings"**
3. Left sidebar: Click **"Access Tokens"**
4. Click **"New token"**
5. Fill in:
   - **Name**: `fine-tune-slm`
   - **Role**: Select **"Write"** (needed to push models)
6. Click **"Generate token"**

### Step 5.5: Copy and Save Token

**🚨 CRITICAL: Copy this token immediately!**

1. You'll see a token like: `hf_...` (long random string)
2. Click **"Copy"**
3. Save this securely

Example:
```
Hugging Face Token: hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890
```

---

## 6. AWS Secrets Manager Setup

Now we'll store all the sensitive credentials in AWS Secrets Manager.

**Two methods available:**
- **Method A**: AWS Console (GUI) - Recommended for beginners
- **Method B**: AWS CLI (Command line) - Faster if you're comfortable with terminal

Choose one method and follow those steps.

---

### Method A: Using AWS Console (GUI)

#### Step 6.1: Navigate to Secrets Manager

1. AWS Console search bar: Type **"Secrets Manager"**
2. Click **"Secrets Manager"**
3. Make sure you're in **us-east-1** region (top right dropdown)

### Step 6.2: Store Hugging Face Token

1. Click **"Store a new secret"** (orange button)
2. **Step 1 - Choose secret type**:
   - Select **"Other type of secret"**
   - Under **"Key/value pairs"**:
     - Click **"Plaintext"** tab
     - Delete everything and paste ONLY your Hugging Face token:
       ```
       hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890
       ```
   - **Encryption key**: Leave as default (aws/secretsmanager)
   - Click **"Next"**

3. **Step 2 - Configure secret**:
   - **Secret name**: `huggingface/api-token`
   - **Description**: `Hugging Face API token for Llama 3.1 access`
   - Click **"Next"**

4. **Step 3 - Configure rotation** (optional):
   - Leave **"Disable automatic rotation"** selected
   - Click **"Next"**

5. **Step 4 - Review**:
   - Review the secret
   - Click **"Store"**

✅ Hugging Face token stored!

### Step 6.3: Store Docker Hub Token

1. Click **"Store a new secret"**
2. **Step 1 - Choose secret type**:
   - Select **"Other type of secret"**
   - Click **"Plaintext"** tab
   - Enter in this format (replace with your actual values):
     ```json
     {
       "username": "yourDockerHubUsername",
       "token": "dckr_pat_1234567890abcdefghijklmnopqrstuvwxyz"
     }
     ```
   - Click **"Next"**

3. **Step 2 - Configure secret**:
   - **Secret name**: `docker/hub-token`
   - **Description**: `Docker Hub credentials for image push`
   - Click **"Next"**

4. Skip rotation, click **"Next"**
5. Click **"Store"**

✅ Docker Hub credentials stored!

### Step 6.4: Store AWS Credentials for GitHub Actions

1. Click **"Store a new secret"**
2. **Step 1 - Choose secret type**:
   - Click **"Plaintext"** tab
   - Enter (replace with your IAM user credentials):
     ```json
     {
       "access_key_id": "AKIAIOSFODNN7EXAMPLE",
       "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
     }
     ```
   - Click **"Next"**

3. **Step 2 - Configure secret**:
   - **Secret name**: `aws/credentials`
   - **Description**: `AWS credentials for GitHub Actions`
   - Click **"Next"**

4. Skip rotation, click **"Next"**
5. Click **"Store"**

✅ AWS credentials stored!

### Step 6.5: Verify All Secrets

1. In Secrets Manager, you should now see 3 secrets:
   - ✅ `huggingface/api-token`
   - ✅ `docker/hub-token`
   - ✅ `aws/credentials`

2. Click each one to verify it was stored correctly (click **"Retrieve secret value"** to view)

---

### Method B: Using AWS CLI (Alternative)

If you prefer command line, you can create all secrets with these commands:

#### Step 6.1: Store Hugging Face Token

```bash
# Replace YOUR_HF_TOKEN with your actual token
aws secretsmanager create-secret \
    --name huggingface/api-token \
    --description "Hugging Face API token for Llama 3.1 access" \
    --secret-string "hf_YOUR_TOKEN_HERE"
```

#### Step 6.2: Store Docker Hub Token

```bash
# Replace with your Docker Hub username and token
aws secretsmanager create-secret \
    --name docker/hub-token \
    --description "Docker Hub credentials for image push" \
    --secret-string '{
        "username": "yourDockerHubUsername",
        "token": "dckr_pat_YOUR_TOKEN_HERE"
    }'
```

#### Step 6.3: Store AWS Credentials

```bash
# Replace with your IAM user access keys
aws secretsmanager create-secret \
    --name aws/credentials \
    --description "AWS credentials for GitHub Actions" \
    --secret-string '{
        "access_key_id": "AKIA_YOUR_KEY_HERE",
        "secret_access_key": "YOUR_SECRET_KEY_HERE"
    }'
```

#### Step 6.4: Verify Secrets Created

```bash
# List all secrets
aws secretsmanager list-secrets

# Retrieve a specific secret to verify
aws secretsmanager get-secret-value \
    --secret-id huggingface/api-token \
    --query SecretString \
    --output text
```

**Expected output**: Your Hugging Face token

---

### Either Method: Verification

After using either method, verify you have 3 secrets:

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

✅ All secrets stored in AWS Secrets Manager!

---

## 7. EC2 Instance Setup

### Step 7.1: Navigate to EC2

1. AWS Console search bar: Type **"EC2"**
2. Click **"EC2"**
3. Ensure you're in **us-east-1** region (top right)

### Step 7.2: Launch Instance

1. Click **"Launch instance"** (orange button)

2. **Name and tags**:
   - **Name**: `fine-tune-llama-training`

3. **Application and OS Images (Amazon Machine Image)**:
   - Click **"Browse more AMIs"**
   - Click **"Quick Start AMIs"** tab
   - Search: `deep learning`
   
   **Choose ONE of these options:**
   
   - **Option 1: Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)** ✅ **RECOMMENDED**
     - Select this AMI (make sure it says **"64-bit (x86)"** - NOT ARM)
     - Click **"Select"** → Click **"Continue"** on pricing page
     - ✅ **Best choice**: NVIDIA drivers + CUDA 12.x + Ubuntu 22.04 LTS
     - ✅ **Lean base** - We install exact PyTorch version in Docker container
     - ✅ **Most stable** - Ubuntu 22.04 LTS (supported until 2027)
     - ✅ **Fastest boot** - No unnecessary pre-installed frameworks
   
   - **Option 2: Deep Learning OSS Nvidia Driver AMI GPU TensorFlow 2.18 (Ubuntu 22.04)**
     - Select this AMI (**64-bit x86**)
     - ✅ Ubuntu 22.04 LTS (stable)
     - ⚠️ Has TensorFlow pre-installed (we don't use it, but harmless)
     - ✅ Good alternative if Option 1 unavailable
     - NVIDIA drivers work perfectly for our PyTorch container
   
   - **Option 3: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Ubuntu 24.04)**
     - Select this AMI (**64-bit x86**)
     - ✅ Has PyTorch 2.8 pre-installed
     - ⚠️ **Ubuntu 24.04** - Very new (April 2024 release)
     - ⚠️ Possible minor package compatibility issues
     - ✅ Still works fine, just newer OS
   
   - **Option 4: Deep Learning AMI Neuron (Ubuntu 22.04)**
     - ⚠️ **Not recommended** - Optimized for AWS Inferentia/Trainium chips
     - Works with NVIDIA GPUs but has unnecessary Neuron SDK bloat
     - Only choose if other options unavailable
   
   **⚠️ CRITICAL**: 
   - **Always choose "64-bit (x86)"** architecture (NOT ARM)
   - g6.2xlarge instances require x86/AMD64 architecture
   - ARM AMIs will NOT work
   
   **Final recommendation**: Use **Option 1 (Base AMI with CUDA + Ubuntu 22.04)** because:
   - ✅ Ubuntu 22.04 LTS = proven stability + long-term support
   - ✅ Our Docker container installs the exact PyTorch/Transformers versions we need
   - ✅ CUDA 12.x drivers = full GPU support without bloat
   - ✅ Clean base = faster deployment and less to maintain

4. **Instance type**:
   - Click dropdown
   - Search: `g6.2xlarge`
   - Select **"g6.2xlarge"** (24 GiB RAM, 1 GPU with 24GB VRAM)
   - **Cost**: ~$0.90/hour

5. **Key pair (login)**:
   - Select **"Proceed without a key pair (Not recommended)"**
   - Why? We'll use AWS SSM Session Manager instead (no SSH keys needed!)

6. **Network settings**:
   - Click **"Edit"**
   - **Auto-assign public IP**: **Enable**
   - **Firewall (security groups)**: Create new security group
     - **Security group name**: `fine-tune-llama-sg`
     - **Description**: `Security group for fine-tuning instance`
     - **Inbound rules**: 
       - Remove all rules (we don't need SSH - using SSM!)

7. **Configure storage**:
   - **Root volume**: 
     - Size: `100 GiB` (enough for Docker images and temp files)
     - Volume type: `gp3`
   - Click **"Add new volume"** (we'll add EBS in next section, skip for now)

8. **Advanced details**:
   - Scroll down to **"IAM instance profile"**
   - Click **"Create new IAM instance profile"**
   - This opens a new tab...

### Step 7.3: Create IAM Role for EC2

**In the new tab (IAM):**

1. Click **"Create role"**
2. **Step 1 - Select trusted entity**:
   - **Trusted entity type**: **AWS service**
   - **Use case**: Select **"EC2"**
   - Click **"Next"**

3. **Step 2 - Add permissions**:
   - Search and select these policies:
     - ☑️ `AmazonSSMManagedInstanceCore` (for SSM access)
     - ☑️ `AmazonS3FullAccess` (for S3 operations)
     - ☑️ `SecretsManagerReadWrite` (to read secrets)
     - ☑️ `CloudWatchAgentServerPolicy` (for CloudWatch logs)
   - Click **"Next"**

4. **Step 3 - Name, review, and create**:
   - **Role name**: `EC2-FineTune-Role`
   - **Description**: `Role for EC2 instance to access S3, Secrets Manager, and SSM`
   - Click **"Create role"**

**Go back to EC2 Launch Instance tab:**

5. Refresh the IAM instance profile dropdown
6. Select **"EC2-FineTune-Role"**

### Step 7.4: Review and Launch

1. **Summary** (right panel): 
   - Review: 1 instance, g6.2xlarge, 100 GiB storage
2. Click **"Launch instance"** (orange button, bottom right)
3. Click **"View all instances"**

### Step 7.5: Get Instance ID

1. Find your instance: `fine-tune-llama-training`
2. **Instance state**: Should show "Running" (wait if "Pending")
3. Copy the **Instance ID**: `i-0123456789abcdef0` (starts with `i-`)
4. **Save this ID** - you'll need it for SSM Parameter Store!

**⚠️ STOP the instance to avoid charges while setting up:**
```bash
aws ec2 stop-instances --instance-ids i-0123456789abcdef0
```

✅ EC2 instance created!

---

## 8. EBS Volume Setup

### Step 8.1: Create EBS Volume

1. In EC2 Console, left sidebar: Click **"Volumes"** (under "Elastic Block Store")
2. Click **"Create volume"** (orange button)

3. **Volume settings**:
   - **Volume type**: `gp3`
   - **Size (GiB)**: `100` (enough for model checkpoints)
   - **IOPS**: `3000` (default for gp3)
   - **Throughput (MB/s)**: `125` (default for gp3)
   - **Availability Zone**: **MUST match your EC2 instance!**
     - Go back to EC2 Instances tab
     - Find your instance's **Availability Zone** (e.g., `us-east-1a`)
     - Use the SAME zone for EBS volume
   - **Snapshot ID**: Leave blank
   - **Encryption**: Leave unchecked (or enable if you prefer)

4. **Tags**:
   - Click **"Add tag"**
   - **Key**: `Name`
   - **Value**: `fine-tune-checkpoints`

5. Click **"Create volume"** (bottom right)

### Step 8.2: Get Volume ID

1. Find your volume: `fine-tune-checkpoints`
2. **State**: Should show "available"
3. Copy the **Volume ID**: `vol-0123456789abcdef0` (starts with `vol-`)
4. **Save this ID** - you'll need it for SSM Parameter Store!

**Note**: We'll attach this volume to EC2 when starting training (done via scripts later)

✅ EBS volume created!

---

## 9. S3 Bucket Setup

### Step 9.1: Navigate to S3

1. AWS Console search bar: Type **"S3"**
2. Click **"S3"**

### Step 9.2: Create Bucket

1. Click **"Create bucket"** (orange button)

2. **General configuration**:
   - **Bucket name**: `fine-tune-llama-models-<your-unique-id>`
     - Example: `fine-tune-llama-models-johndoe123`
     - **Must be globally unique!** Add your name or random numbers
     - **Only lowercase, numbers, hyphens allowed**
   - **AWS Region**: `us-east-1`

3. **Object Ownership**:
   - Select **"ACLs disabled (recommended)"**

4. **Block Public Access settings**:
   - ☑️ **"Block all public access"** (checked - keep private!)

5. **Bucket Versioning**:
   - Select **"Disable"** (unless you want versioning)

6. **Default encryption**:
   - **Encryption type**: **"Server-side encryption with Amazon S3 managed keys (SSE-S3)"**

7. Click **"Create bucket"** (bottom)

### Step 9.3: Save Bucket Name

**Copy and save your bucket name**, example:
```
Bucket name: fine-tune-llama-models-johndoe123
```

You'll need this for SSM Parameter Store!

✅ S3 bucket created!

---

## 10. ECR Repository Setup

### Step 10.1: Navigate to ECR

1. AWS Console search bar: Type **"ECR"**
2. Click **"Elastic Container Registry"**
3. Ensure **us-east-1** region

### Step 10.2: Create Repository

1. Click **"Get started"** or **"Create repository"**

2. **General settings**:
   - **Visibility settings**: **Private**
   - **Repository name**: `fine-tune-llama`

3. **Tag immutability**:
   - Select **"Disabled"** (allows overwriting tags like "latest")

4. **Image scan settings**:
   - **Scan on push**: Leave unchecked (optional)

5. **Encryption settings**:
   - Leave as default (AES-256)

6. Click **"Create repository"**

### Step 10.3: Get Repository URI

1. Click on your repository: **fine-tune-llama**
2. Copy the **URI** (top right), looks like:
   ```
   123456789012.dkr.ecr.us-east-1.amazonaws.com/fine-tune-llama
   ```
3. **Save the registry URL** (everything before `/fine-tune-llama`):
   ```
   Registry URL: 123456789012.dkr.ecr.us-east-1.amazonaws.com
   Repository name: fine-tune-llama
   ```

You'll need both for SSM Parameter Store!

✅ ECR repository created!

---

## 11. CloudWatch Logs Setup

### Step 11.1: Navigate to CloudWatch

1. AWS Console search bar: Type **"CloudWatch"**
2. Click **"CloudWatch"**
3. Left sidebar: Click **"Log groups"** (under "Logs")

### Step 11.2: Create Log Group

1. Click **"Create log group"**
2. **Log group name**: `/aws/ssm/fine-tune-llama`
3. **Retention setting**: `1 week` (or your preference)
   - Options: 1 day, 3 days, 1 week, 2 weeks, 1 month, etc.
   - This controls how long logs are kept (affects cost)
4. Click **"Create"**

✅ CloudWatch log group created!

**Note**: SSM will automatically create log streams in this group when commands run.

---

## 12. SSM Parameter Store Setup

Now we'll store all the resource IDs we just created.

### Step 12.1: Navigate to Parameter Store

1. AWS Console search bar: Type **"Systems Manager"**
2. Click **"Systems Manager"**
3. Left sidebar: Scroll down and click **"Parameter Store"** (under "Application Management")

### Step 12.2: Create Parameters

We'll create parameters one by one. For each parameter:

#### Parameter 1: AWS Region

1. Click **"Create parameter"**
2. **Parameter details**:
   - **Name**: `/fine-tune-slm/aws/region`
   - **Description**: `AWS region for resources`
   - **Tier**: **Standard**
   - **Type**: **String**
   - **Data type**: **text**
   - **Value**: `us-east-1`
3. Click **"Create parameter"**

#### Parameter 2: EC2 Instance ID

1. Click **"Create parameter"**
2. **Parameter details**:
   - **Name**: `/fine-tune-slm/ec2/instance-id`
   - **Description**: `EC2 instance ID for training`
   - **Tier**: **Standard**
   - **Type**: **String**
   - **Value**: `i-0123456789abcdef0` ← **YOUR instance ID!**
3. Click **"Create parameter"**

#### Parameter 3: EC2 Instance Type

1. Click **"Create parameter"**
2. **Name**: `/fine-tune-slm/ec2/instance-type`
3. **Description**: `EC2 instance type`
4. **Type**: **String**
5. **Value**: `g6.2xlarge`
6. Click **"Create parameter"**

#### Parameter 4: EBS Volume ID

1. Click **"Create parameter"**
2. **Name**: `/fine-tune-slm/ebs/volume-id`
3. **Description**: `EBS volume ID for checkpoints`
4. **Type**: **String**
5. **Value**: `vol-0123456789abcdef0` ← **YOUR volume ID!**
6. Click **"Create parameter"**

#### Parameter 5: EBS Mount Path

1. Click **"Create parameter"**
2. **Name**: `/fine-tune-slm/ebs/mount-path`
3. **Description**: `EBS volume mount path in container`
4. **Type**: **String**
5. **Value**: `/mnt/training`
6. Click **"Create parameter"**

#### Parameter 6: EBS Volume Type

1. Click **"Create parameter"**
2. **Name**: `/fine-tune-slm/ebs/volume-type`
3. **Type**: **String**
4. **Value**: `gp3`
5. Click **"Create parameter"**

#### Parameter 7: EBS Size

1. Click **"Create parameter"**
2. **Name**: `/fine-tune-slm/ebs/size-gb`
3. **Type**: **String**
4. **Value**: `100`
5. Click **"Create parameter"**

#### Parameter 8: S3 Bucket

1. Click **"Create parameter"**
2. **Name**: `/fine-tune-slm/s3/bucket`
3. **Description**: `S3 bucket for model artifacts`
4. **Type**: **String**
5. **Value**: `fine-tune-llama-models-johndoe123` ← **YOUR bucket name!**
6. Click **"Create parameter"**

#### Parameter 9: S3 Prefix

1. Click **"Create parameter"**
2. **Name**: `/fine-tune-slm/s3/prefix`
3. **Type**: **String**
4. **Value**: `models/llama-3.1-8b-medical-ie`
5. Click **"Create parameter"**

#### Parameter 10: ECR Repository

1. Click **"Create parameter"**
2. **Name**: `/fine-tune-slm/ecr/repository`
3. **Type**: **String**
4. **Value**: `fine-tune-llama`
5. Click **"Create parameter"**

#### Parameter 11: ECR Registry

1. Click **"Create parameter"**
2. **Name**: `/fine-tune-slm/ecr/registry`
3. **Description**: `ECR registry URL`
4. **Type**: **String**
5. **Value**: `123456789012.dkr.ecr.us-east-1.amazonaws.com` ← **YOUR registry URL!**
6. Click **"Create parameter"**

#### Parameter 12: Secrets - HF Token Name

1. Click **"Create parameter"**
2. **Name**: `/fine-tune-slm/secrets/hf-token-name`
3. **Type**: **String**
4. **Value**: `huggingface/api-token`
5. Click **"Create parameter"**

#### Parameter 13: Secrets - AWS Credentials Name

1. Click **"Create parameter"**
2. **Name**: `/fine-tune-slm/secrets/aws-credentials-name`
3. **Type**: **String**
4. **Value**: `aws/credentials`
5. Click **"Create parameter"**

#### Parameter 14: Secrets - Docker Token Name

1. Click **"Create parameter"**
2. **Name**: `/fine-tune-slm/secrets/docker-token-name`
3. **Type**: **String**
4. **Value**: `docker/hub-token`
5. Click **"Create parameter"**

#### Parameter 15: CloudWatch Log Group

1. Click **"Create parameter"**
2. **Name**: `/fine-tune-slm/cloudwatch/log-group`
3. **Type**: **String**
4. **Value**: `/aws/ssm/fine-tune-llama`
5. Click **"Create parameter"**

#### Parameter 16: CloudWatch Log Stream Prefix

1. Click **"Create parameter"**
2. **Name**: `/fine-tune-slm/cloudwatch/log-stream-prefix`
3. **Type**: **String**
4. **Value**: `training`
5. Click **"Create parameter"**

#### Parameter 17: Hugging Face Output Repo

1. Click **"Create parameter"**
2. **Name**: `/fine-tune-slm/output/hf-repo`
3. **Description**: `Hugging Face repository for trained model`
4. **Type**: **String**
5. **Value**: `yourHFusername/llama-3.1-8b-medical-ie` ← **YOUR HF username!**
6. Click **"Create parameter"**

### Step 12.3: Verify All Parameters

1. In Parameter Store, you should see **17 parameters**
2. All should have names starting with `/fine-tune-slm/`
3. Click on a few to verify values are correct

✅ SSM Parameter Store configured!

---

## 13. Verification & Testing

### Step 13.1: Test AWS CLI Access to All Services

```bash
# Test SSM Parameter Store
aws ssm get-parameters-by-path \
  --path /fine-tune-slm/ \
  --recursive

# Should show all 17 parameters

# Test Secrets Manager
aws secretsmanager list-secrets

# Should show 3 secrets

# Test EC2
aws ec2 describe-instances \
  --instance-ids <YOUR_INSTANCE_ID>

# Should show your instance details

# Test S3
aws s3 ls s3://<YOUR_BUCKET_NAME>

# Should list bucket (empty for now)

# Test ECR
aws ecr describe-repositories

# Should show fine-tune-llama repository
```

### Step 13.2: Create Summary Document

Create a file to save all your important IDs and values:

**File: `setup-summary.txt`** (DO NOT commit to git!)

```
=== fine-tune-slm Setup Summary ===
Date: 2025-10-28

AWS ACCOUNT
-----------
Account ID: 123456789012
IAM User: fine-tune-slm-admin
Region: us-east-1

AWS CREDENTIALS (Stored in ~/.aws/credentials)
-----------
Access Key ID: AKIA...
Secret Access Key: (saved in ~/.aws/credentials)

EC2
-----------
Instance ID: i-0123456789abcdef0
Instance Type: g6.2xlarge
IAM Role: EC2-FineTune-Role

EBS
-----------
Volume ID: vol-0123456789abcdef0
Volume Type: gp3
Size: 100 GiB
Availability Zone: us-east-1a

S3
-----------
Bucket Name: fine-tune-llama-models-johndoe123
Prefix: models/llama-3.1-8b-medical-ie

ECR
-----------
Registry URL: 123456789012.dkr.ecr.us-east-1.amazonaws.com
Repository: fine-tune-llama

HUGGING FACE
-----------
Username: yourHFusername
Token: (stored in AWS Secrets Manager: huggingface/api-token)
Output Repo: yourHFusername/llama-3.1-8b-medical-ie
Llama 3.1 Access: Approved ✓

DOCKER HUB
-----------
Username: yourDockerHubUsername
Token: (stored in AWS Secrets Manager: docker/hub-token)

AWS SECRETS MANAGER
-----------
✓ huggingface/api-token
✓ docker/hub-token
✓ aws/credentials

SSM PARAMETER STORE
-----------
✓ 17 parameters created under /fine-tune-slm/

CLOUDWATCH
-----------
Log Group: /aws/ssm/fine-tune-llama
```

### Step 13.3: Verify Project Script Will Work

```bash
# Navigate to project
cd /Volumes/deuxSSD/Developer/fine-tune-slm

# Test the SSM parameter initialization script (dry run)
python scripts/setup/init_ssm_parameters.py --dry-run

# Should show:
# - Loading config files
# - Found X SSM parameters
# - Would create/skip parameters
```

---

## 🎉 Setup Complete!

You've now set up:

✅ AWS Account & IAM User  
✅ AWS CLI configured  
✅ Docker Hub account & token  
✅ Hugging Face account & token (with Llama 3.1 access)  
✅ AWS Secrets Manager (3 secrets)  
✅ EC2 Instance (g6.2xlarge)  
✅ EBS Volume (100 GiB gp3)  
✅ S3 Bucket  
✅ ECR Repository  
✅ CloudWatch Logs  
✅ SSM Parameter Store (17 parameters)  

### Next Steps

1. **Configure GitHub Secrets** (for GitHub Actions):
   - Go to your GitHub repo → Settings → Secrets and variables → Actions
   - Add secrets:
     - `AWS_ACCESS_KEY_ID`: Your IAM user access key
     - `AWS_SECRET_ACCESS_KEY`: Your IAM user secret key

2. **Test Your Setup**:
   ```bash
   # Start EC2 instance
   aws ec2 start-instances --instance-ids <YOUR_INSTANCE_ID>
   
   # Wait for it to be running
   aws ec2 wait instance-running --instance-ids <YOUR_INSTANCE_ID>
   
   # Test SSM connection (no SSH needed!)
   aws ssm start-session --target <YOUR_INSTANCE_ID>
   ```

3. **Ready for Implementation**:
   - All infrastructure is set up
   - All credentials are secured
   - You can now start implementing the actual training code!

### Important Reminders

⚠️ **Cost Management**:
- **STOP EC2 instance** when not using: `aws ec2 stop-instances --instance-ids i-...`
- g6.2xlarge costs ~$0.90/hour when running
- Stopped instances don't charge for compute (only for storage)

⚠️ **Security**:
- Never commit `setup-summary.txt` to git
- Never share your access keys
- Rotate credentials every 90 days

⚠️ **Backups**:
- Save your setup summary in a secure password manager
- Keep a copy of instance IDs and resource ARNs

---

**Questions or issues?** Refer to the specific service documentation or re-read the relevant section above.
