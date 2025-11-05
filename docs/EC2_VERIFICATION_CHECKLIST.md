# EC2 Infrastructure Verification Checklist

**Purpose:** Verify your EC2 instance, EBS volumes, and SSM connectivity are properly configured before implementing Phase 8 automation scripts.

**Prerequisites:** 
- ✅ EC2 quota approved
- ✅ EC2 instance launched
- ✅ EBS checkpoint volume created
- ✅ SSM parameters populated

---

## 📋 Table of Contents

1. [EC2 Instance Verification](#1-ec2-instance-verification)
2. [EBS Volume Verification](#2-ebs-volume-verification)
3. [SSM Connectivity Test](#3-ssm-connectivity-test)
4. [IAM Role Verification](#4-iam-role-verification)
5. [Network Configuration](#5-network-configuration)
6. [SSM Parameter Store Validation](#6-ssm-parameter-store-validation)
7. [Docker and GPU Verification](#7-docker-and-gpu-verification)
8. [Cost Estimation](#8-cost-estimation)

---

## 1. EC2 Instance Verification

### 1.1 Check Instance Details

```bash
# Get your instance ID from SSM
INSTANCE_ID=$(aws ssm get-parameter \
  --name /fine-tune-slm/ec2/instance-id \
  --query 'Parameter.Value' \
  --output text)

echo "Instance ID: $INSTANCE_ID"

# Verify instance details
aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].[InstanceId,InstanceType,State.Name,PublicDnsName,PrivateIpAddress]' \
  --output table
```

**Expected Output:**
```
-----------------------------------------------------------------------------------
|                              DescribeInstances                                  |
+----------------------------+-----------+---------+------------------+------------+
|  i-1234567890abcdef0       | g6.2xlarge| running | ec2-xx-xx.aws... | 10.0.x.x   |
+----------------------------+-----------+---------+------------------+------------+
```

**Verify:**
- ✅ Instance Type: `g6.2xlarge`
- ✅ State: `running`
- ✅ Has private IP address
- ✅ (Optional) Public DNS name if needed

---

### 1.2 Check Instance Specifications

```bash
# Check GPU and memory
aws ec2 describe-instance-types \
  --instance-types g6.2xlarge \
  --query 'InstanceTypes[0].[InstanceType,VCpuInfo.DefaultVCpus,MemoryInfo.SizeInMiB,GpuInfo.Gpus[0].[Name,Count,MemoryInfo.SizeInMiB]]' \
  --output table
```

**Expected Output:**
```
-----------------------------------------------------------------
|                     DescribeInstanceTypes                     |
+------------+-------+---------+--------------+------+----------+
| g6.2xlarge |   8   | 32768   | NVIDIA L4    |  1   |  24576   |
+------------+-------+---------+--------------+------+----------+
```

**Verify:**
- ✅ vCPUs: 8
- ✅ RAM: 32 GB (32768 MiB)
- ✅ GPU: 1x NVIDIA L4
- ✅ VRAM: 24 GB (24576 MiB)

---

### 1.3 Check Instance AMI

```bash
# Check what AMI was used
aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].[ImageId,Tags[?Key==`Name`].Value|[0]]' \
  --output table
```

**Expected:**
- ✅ AMI should be "Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)" or similar
- ✅ Image ID starts with `ami-`

---

## 2. EBS Volume Verification

### 2.1 Check Root Volume

```bash
# Check root volume details
aws ec2 describe-volumes \
  --filters "Name=attachment.instance-id,Values=$INSTANCE_ID" \
            "Name=attachment.device,Values=/dev/sda1,/dev/xvda" \
  --query 'Volumes[0].[VolumeId,Size,VolumeType,State,Iops,Throughput]' \
  --output table
```

**Expected Output:**
```
---------------------------------------------------------------
|                      DescribeVolumes                        |
+------------------------+-----+------+-----------+------+-----+
|  vol-0abc123...        | 100 | gp3  | in-use    | 3000 | 125 |
+------------------------+-----+------+-----------+------+-----+
```

**Verify:**
- ✅ Size: 100 GB
- ✅ Type: gp3
- ✅ State: in-use
- ✅ IOPS: 3000 (default for gp3)
- ✅ Throughput: 125 MB/s

---

### 2.2 Check Checkpoint EBS Volume

```bash
# Get checkpoint volume ID from SSM
CHECKPOINT_VOLUME_ID=$(aws ssm get-parameter \
  --name /fine-tune-slm/ebs/volume-id \
  --query 'Parameter.Value' \
  --output text)

echo "Checkpoint Volume ID: $CHECKPOINT_VOLUME_ID"

# Check checkpoint volume details
aws ec2 describe-volumes \
  --volume-ids $CHECKPOINT_VOLUME_ID \
  --query 'Volumes[0].[VolumeId,Size,VolumeType,State,AvailabilityZone,Attachments[0].[InstanceId,Device,State]]' \
  --output table
```

**Expected Output:**
```
-------------------------------------------------------------------------
|                           DescribeVolumes                             |
+------------------------+-----+------+------------+----------+----------+
|  vol-0def456...        | 100 | gp3  | in-use     | us-east-1a         |
|  i-1234567890abcdef0   | /dev/sdf | attached                          |
+------------------------+-----+------+------------+----------+----------+
```

**Verify:**
- ✅ Volume ID matches SSM parameter
- ✅ Size: 100 GB
- ✅ Type: gp3
- ✅ State: in-use
- ✅ Availability Zone: **SAME as EC2 instance**
- ✅ Attached to correct instance
- ✅ Device: `/dev/sdf` or similar
- ✅ Attachment state: attached

---

### 2.3 Verify EBS Volume Location

```bash
# CRITICAL: Check that EBS volume is in SAME availability zone as EC2
aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].Placement.AvailabilityZone' \
  --output text

aws ec2 describe-volumes \
  --volume-ids $CHECKPOINT_VOLUME_ID \
  --query 'Volumes[0].AvailabilityZone' \
  --output text
```

**Expected:**
Both commands should return **THE SAME** availability zone (e.g., `us-east-1a`)

**⚠️ CRITICAL:** If they're different, the volume cannot attach! You'll need to:
1. Create a snapshot of the volume
2. Create new volume in correct AZ from snapshot
3. Update SSM parameter

---

## 3. SSM Connectivity Test

### 3.1 Test SSM Session Manager

```bash
# Test SSM connectivity
aws ssm describe-instance-information \
  --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
  --query 'InstanceInformationList[0].[InstanceId,PingStatus,PlatformName,PlatformVersion,AgentVersion]' \
  --output table
```

**Expected Output:**
```
-------------------------------------------------------------------------
|                    DescribeInstanceInformation                        |
+----------------------------+--------+----------------+---------+-------+
|  i-1234567890abcdef0       | Online | Ubuntu         | 22.04   | 3.x.x |
+----------------------------+--------+----------------+---------+-------+
```

**Verify:**
- ✅ PingStatus: `Online`
- ✅ PlatformName: Ubuntu
- ✅ PlatformVersion: 22.04
- ✅ AgentVersion: 3.x.x or higher

---

### 3.2 Test Remote Command Execution

```bash
# Send a simple test command via SSM
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --comment "Test SSM connectivity" \
  --parameters 'commands=["echo \"SSM Test: $(date)\"", "whoami", "pwd", "df -h"]' \
  --output text \
  --query 'Command.CommandId'
```

**Save the Command ID, then check output:**

```bash
# Replace COMMAND_ID with the output from above
COMMAND_ID="<your-command-id>"

# Wait 5 seconds for command to complete
sleep 5

# Get command output
aws ssm get-command-invocation \
  --command-id $COMMAND_ID \
  --instance-id $INSTANCE_ID \
  --query '[Status,StandardOutputContent]' \
  --output text
```

**Expected Output:**
```
Success
SSM Test: Mon Nov 4 12:34:56 UTC 2025
root (or ssm-user)
/var/snap/amazon-ssm-agent/xxxxx
Filesystem      Size  Used Avail Use% Mounted on
/dev/root        97G   21G   76G  22% /
/dev/mapper/vg.01-lv_ephemeral  412G   28K  391G   1% /opt/dlami/nvme
...
```

**Verify:**
- ✅ Status: `Success`
- ✅ Commands executed successfully
- ✅ User is `root`, `ubuntu`, or `ssm-user`
- ✅ Root filesystem has space available
- ✅ Instance store visible at `/opt/dlami/nvme` (450GB ephemeral storage)

---

## 4. IAM Role Verification

### 4.1 Check EC2 IAM Instance Profile

```bash
# Check IAM role attached to instance
aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].IamInstanceProfile.Arn' \
  --output text
```

**Expected:**
```
arn:aws:iam::123456789012:instance-profile/EC2-FineTune-Role
```

**Verify:**
- ✅ Instance has an IAM role attached
- ✅ Role name contains something like "FineTune" or "EC2"

---

### 4.2 Verify IAM Role Permissions

```bash
# Get role name
ROLE_NAME=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].IamInstanceProfile.Arn' \
  --output text | cut -d'/' -f2)

echo "IAM Role: $ROLE_NAME"

# List attached policies
aws iam list-attached-role-policies \
  --role-name $ROLE_NAME \
  --query 'AttachedPolicies[*].[PolicyName,PolicyArn]' \
  --output table
```

**Expected Policies:**
- ✅ `AmazonSSMManagedInstanceCore` (for SSM)
- ✅ `AmazonS3FullAccess` or custom S3 policy
- ✅ `AmazonEC2ContainerRegistryReadOnly` or `AmazonEC2ContainerRegistryPowerUser`
- ✅ `SecretsManagerReadWrite` or custom secrets policy

---

## 5. Network Configuration

### 5.1 Check Security Group Rules

```bash
# Get security group ID
SECURITY_GROUP_ID=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' \
  --output text)

echo "Security Group: $SECURITY_GROUP_ID"

# Check inbound rules
aws ec2 describe-security-groups \
  --group-ids $SECURITY_GROUP_ID \
  --query 'SecurityGroups[0].IpPermissions[*].[IpProtocol,FromPort,ToPort,IpRanges[0].CidrIp]' \
  --output table

# Check outbound rules (more important for SSM)
aws ec2 describe-security-groups \
  --group-ids $SECURITY_GROUP_ID \
  --query 'SecurityGroups[0].IpPermissionsEgress[*].[IpProtocol,FromPort,ToPort,IpRanges[0].CidrIp]' \
  --output table
```

**Expected Inbound Rules:**
- ✅ **Empty or no output is CORRECT** - SSM doesn't require any inbound ports!
- ⚠️ SSM Session Manager uses **outbound-only** connections (EC2 → AWS SSM endpoints)
- ⚠️ No SSH (port 22) needed - more secure than traditional SSH access

**Expected Outbound Rules:**
- ✅ All traffic (Protocol: -1) to 0.0.0.0/0, OR
- ✅ HTTPS (Protocol: tcp, Port: 443) to 0.0.0.0/0

**If outbound rules are missing, add them:**
```bash
# Allow all outbound traffic (standard default)
aws ec2 authorize-security-group-egress \
  --group-id $SECURITY_GROUP_ID \
  --ip-permissions IpProtocol=-1,IpRanges='[{CidrIp=0.0.0.0/0}]'
```

**Verify:**
- ✅ **Inbound: Empty is OK** (SSM doesn't need inbound ports)
- ✅ **Outbound: Must allow HTTPS** (for SSM, S3, ECR, Secrets Manager)

---

### 5.2 Check VPC and Subnet

```bash
# Check VPC endpoints for SSM (optional but recommended)
aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].[VpcId,SubnetId]' \
  --output table
```

**Verify:**
- ✅ Instance is in a VPC
- ✅ Instance is in a subnet with internet access (NAT Gateway or Internet Gateway)

---

## 6. SSM Parameter Store Validation

### 6.1 Verify All Required Parameters

```bash
# List all fine-tune-slm parameters
aws ssm get-parameters-by-path \
  --path /fine-tune-slm \
  --recursive \
  --query 'Parameters[*].[Name,Value,Type]' \
  --output table
```

**Expected Parameters (17 total):**

**AWS Resources:**
- ✅ `/fine-tune-slm/aws/region` → us-east-1
- ✅ `/fine-tune-slm/ec2/instance-id` → i-xxxxx
- ✅ `/fine-tune-slm/ec2/instance-type` → g6.2xlarge
- ✅ `/fine-tune-slm/ebs/volume-id` → vol-xxxxx
- ✅ `/fine-tune-slm/ebs/mount-path` → /mnt/training

**S3 and ECR:**
- ✅ `/fine-tune-slm/s3/bucket` → your-bucket-name
- ✅ `/fine-tune-slm/s3/prefix` → models/llama-3.1-8b-medical-ie
- ✅ `/fine-tune-slm/ecr/repository` → fine-tune-llama
- ✅ `/fine-tune-slm/ecr/registry` → xxxxx.dkr.ecr.us-east-1.amazonaws.com

**Secrets:**
- ✅ `/fine-tune-slm/secrets/hf-token-name` → huggingface/api-token
- ✅ `/fine-tune-slm/secrets/aws-credentials-name` → aws/credentials
- ✅ `/fine-tune-slm/secrets/docker-token-name` → docker/hub-token

**CloudWatch:**
- ✅ `/fine-tune-slm/cloudwatch/log-group` → /aws/ssm/fine-tune-llama
- ✅ `/fine-tune-slm/cloudwatch/log-stream-prefix` → training

**Output:**
- ✅ `/fine-tune-slm/output/hf-repo` → username/llama-3.1-8b-medical-ie

---

### 6.2 Test Parameter Retrieval from Python

```bash
# Test config loading with SSM
python3 << 'EOF'
import sys
sys.path.append('/Volumes/deuxSSD/Developer/fine-tune-slm')

from src.utils.config import load_all_configs

# Load configs with SSM enabled
configs = load_all_configs('config', use_ssm=True)

# Test retrieval
print("✅ AWS Region:", configs.get_aws('aws.region'))
print("✅ EC2 Instance ID:", configs.get_aws('aws.ec2.instance_id'))
print("✅ EBS Volume ID:", configs.get_aws('aws.ebs.volume_id'))
print("✅ S3 Bucket:", configs.get_aws('aws.s3.bucket'))
print("✅ ECR Repository:", configs.get_aws('aws.ecr.repository'))
print("\n✅ All SSM parameters retrieved successfully!")
EOF
```

**Expected:** All values should be retrieved without errors.

---

### 6.3 Test Parameter Retrieval on EC2 (via SSM Session Manager)

When running AWS CLI commands **inside the EC2 instance** (via Session Manager), you need to set the region:

```bash
# Start SSM session
aws ssm start-session --target $INSTANCE_ID

# Once connected on EC2, set region and test parameter retrieval:
export AWS_DEFAULT_REGION=us-east-1

# Test retrieving SSM parameters
ECR_REGISTRY=$(aws ssm get-parameter \
  --name /fine-tune-slm/ecr/registry \
  --query 'Parameter.Value' \
  --output text)

echo "ECR Registry: $ECR_REGISTRY"

# Test retrieving secret name
HF_SECRET_NAME=$(aws ssm get-parameter \
  --name /fine-tune-slm/secrets/hf-token-name \
  --query 'Parameter.Value' \
  --output text)

echo "HF Secret Name: $HF_SECRET_NAME"

# Exit SSM session
exit
```

**Expected Output:**
```
ECR Registry: 123456789012.dkr.ecr.us-east-1.amazonaws.com
HF Secret Name: huggingface/api-token
```

**⚠️ Important:** The EC2 instance uses its **IAM instance role** for credentials (automatic), but you must **explicitly set the region** when running AWS CLI commands on the instance.

---

## 7. Docker and GPU Verification

### 7.1 Test Docker on EC2

```bash
# Start SSM session (interactive)
aws ssm start-session --target $INSTANCE_ID

# Once connected, run these commands:
```

**On EC2 instance:**
```bash
# Check Docker installation
docker --version
# Expected: Docker version 24.x.x or higher

# Check Docker daemon status
sudo systemctl status docker
# Expected: active (running)

# Test Docker
sudo docker run hello-world
# Expected: "Hello from Docker!" message

# Check Docker permissions (should work without sudo)
docker ps
# If permission denied, add user to docker group:
# sudo usermod -aG docker $USER
# Then logout and login again
```

---

### 7.2 Test GPU Access

**On EC2 instance:**
```bash
# Check NVIDIA driver
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 580.xx.xx    Driver Version: 580.xx.xx    CUDA Version: 13.0   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |                               |                      |               MIG M. |
# |===============================+======================+======================|
# |   0  NVIDIA L4           Off  | 00000000:00:1E.0 Off |                    0 |
# | N/A   28C    P8    15W /  72W |      0MiB / 23034MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+

# Check if nvcc is available (optional - not needed for PyTorch training)
nvcc --version
# Note: nvcc may not be found - this is NORMAL and expected
# nvcc is only needed for compiling CUDA code, not for running PyTorch
# Your Docker container has all necessary CUDA runtime libraries

# Test GPU with Docker
sudo docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
# Should show same GPU info
```

**Verify:**
- ✅ GPU detected: NVIDIA L4
- ✅ VRAM: ~23 GB available
- ✅ CUDA version: 12.x or 13.0
- ✅ Driver version: 535+ or 580+
- ✅ Docker can access GPU
- ⚠️ `nvcc` not found is **expected and OK** (only needed for compiling CUDA code)

---

### 7.3 Test ECR Login and Image Pull

**On EC2 instance (via SSM Session Manager):**
```bash
# IMPORTANT: Set AWS region first (required for EC2 instance)
export AWS_DEFAULT_REGION=us-east-1

# Get ECR registry URL from SSM Parameter Store
ECR_REGISTRY=$(aws ssm get-parameter \
  --name /fine-tune-slm/ecr/registry \
  --query 'Parameter.Value' \
  --output text)

echo "ECR Registry: $ECR_REGISTRY"

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ECR_REGISTRY

# Pull your training image
docker pull $ECR_REGISTRY/fine-tune-llama:latest

# Verify image
docker images | grep fine-tune-llama
# Should show your image (~9 GB)
```

**Verify:**
- ✅ ECR registry URL retrieved from SSM
- ✅ ECR login successful ("Login Succeeded")
- ✅ Image pulled successfully
- ✅ Image size ~9 GB

**⚠️ Note:** If you get "You must specify a region" error, make sure you ran `export AWS_DEFAULT_REGION=us-east-1` first.

---

## 8. Cost Estimation

### 8.1 Calculate Current Costs

```bash
# Current resource costs (assuming us-east-1)

cat << 'EOF'
EC2 g6.2xlarge (running):
- On-Demand: $0.7512/hour
- Monthly (24/7): ~$547/month ⚠️

EBS Volumes:
- Root (100 GB gp3): $8.00/month
- Checkpoint (100 GB gp3): $8.00/month
- Total: $16/month

S3 Storage (~50 GB models):
- Storage: $1.15/month
- Requests: ~$0.10/month

ECR Storage (~50 GB images):
- Storage: $5.00/month

Total Monthly Cost (EC2 running 24/7):
~$569/month ⚠️

**RECOMMENDATION:** Stop EC2 when not training!
- Training only: ~$15-20 for 3-4 hour session
- Storage only: ~$30/month
EOF
```

---

### 8.2 Cost-Saving Recommendations

```bash
cat << 'EOF'
✅ Stop EC2 instance when not training:
   - Start: scripts/setup/start_ec2.py
   - Train: 3-4 hours
   - Stop: scripts/setup/stop_ec2.py
   - Cost: ~$2.25 per training run

✅ Set up budget alerts:
   aws budgets create-budget --cli-input-json file://budget.json

✅ Use EC2 Auto-stop:
   - Create CloudWatch alarm for idle CPU
   - Automatically stop after 1 hour of inactivity

✅ Delete old EBS snapshots:
   - Keep only last 2-3 snapshots
EOF
```

---

## 9. Final Verification Checklist

Run this comprehensive check:

```bash
cat << 'EOF' > /tmp/verify-ec2-setup.sh
#!/bin/bash
set -e

echo "=========================================="
echo "EC2 Infrastructure Verification"
echo "=========================================="
echo ""

# Get instance ID
INSTANCE_ID=$(aws ssm get-parameter --name /fine-tune-slm/ec2/instance-id --query 'Parameter.Value' --output text)
echo "✓ Instance ID: $INSTANCE_ID"

# Check instance state
STATE=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text)
echo "✓ Instance State: $STATE"

# Check SSM connectivity
SSM_STATUS=$(aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$INSTANCE_ID" --query 'InstanceInformationList[0].PingStatus' --output text)
echo "✓ SSM Status: $SSM_STATUS"

# Check EBS volume
VOLUME_ID=$(aws ssm get-parameter --name /fine-tune-slm/ebs/volume-id --query 'Parameter.Value' --output text)
VOLUME_STATE=$(aws ec2 describe-volumes --volume-ids $VOLUME_ID --query 'Volumes[0].State' --output text)
echo "✓ Checkpoint Volume: $VOLUME_ID ($VOLUME_STATE)"

# Check availability zone match
EC2_AZ=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].Placement.AvailabilityZone' --output text)
EBS_AZ=$(aws ec2 describe-volumes --volume-ids $VOLUME_ID --query 'Volumes[0].AvailabilityZone' --output text)

if [ "$EC2_AZ" == "$EBS_AZ" ]; then
  echo "✓ Availability Zones Match: $EC2_AZ"
else
  echo "✗ ERROR: AZ Mismatch - EC2: $EC2_AZ, EBS: $EBS_AZ"
  exit 1
fi

# Check IAM role
IAM_ROLE=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].IamInstanceProfile.Arn' --output text)
echo "✓ IAM Role: $IAM_ROLE"

# Count SSM parameters
PARAM_COUNT=$(aws ssm get-parameters-by-path --path /fine-tune-slm --recursive --query 'length(Parameters)' --output text)
echo "✓ SSM Parameters: $PARAM_COUNT/17"

echo ""
echo "=========================================="
echo "✅ Verification Complete!"
echo "=========================================="
echo ""
echo "Ready for Phase 8: EC2 Setup Scripts"
EOF

chmod +x /tmp/verify-ec2-setup.sh
/tmp/verify-ec2-setup.sh
```

---

## 10. Troubleshooting

### Issue: SSM Status is "ConnectionLost"

**Solution:**
```bash
# Check if SSM agent is running
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["sudo systemctl status amazon-ssm-agent"]'

# Restart SSM agent if needed
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["sudo systemctl restart amazon-ssm-agent"]'
```

---

### Issue: EBS Volume Not Attached

**Solution:**
```bash
# Check attachment status
aws ec2 describe-volumes --volume-ids $CHECKPOINT_VOLUME_ID

# If "available" (not attached), attach it:
aws ec2 attach-volume \
  --volume-id $CHECKPOINT_VOLUME_ID \
  --instance-id $INSTANCE_ID \
  --device /dev/sdf
```

---

### Issue: AZ Mismatch

**Solution:**
```bash
# Create snapshot
SNAPSHOT_ID=$(aws ec2 create-snapshot \
  --volume-id $CHECKPOINT_VOLUME_ID \
  --description "Checkpoint volume backup" \
  --query 'SnapshotId' \
  --output text)

# Wait for snapshot to complete
aws ec2 wait snapshot-completed --snapshot-ids $SNAPSHOT_ID

# Get correct AZ
CORRECT_AZ=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].Placement.AvailabilityZone' \
  --output text)

# Create new volume in correct AZ
NEW_VOLUME_ID=$(aws ec2 create-volume \
  --snapshot-id $SNAPSHOT_ID \
  --availability-zone $CORRECT_AZ \
  --volume-type gp3 \
  --size 100 \
  --query 'VolumeId' \
  --output text)

# Update SSM parameter
aws ssm put-parameter \
  --name /fine-tune-slm/ebs/volume-id \
  --value $NEW_VOLUME_ID \
  --overwrite
```

---

## ✅ Success Criteria

Before proceeding to Phase 8, verify:

- [ ] EC2 instance is `running` and reachable via SSM
- [ ] SSM PingStatus is `Online`
- [ ] EBS checkpoint volume is `attached` to EC2
- [ ] EC2 and EBS are in the **same availability zone**
- [ ] IAM role has SSM, S3, ECR, and Secrets Manager permissions
- [ ] All 17 SSM parameters are populated
- [ ] GPU is detected (`nvidia-smi` works)
- [ ] Docker is installed and can access GPU
- [ ] Can pull images from ECR
- [ ] Python config loader retrieves SSM parameters successfully

**Once all checks pass, you're ready for Phase 8!** 🚀

---

## Next Steps

After verification is complete:

1. **Stop EC2 instance** to save costs:
   ```bash
   aws ec2 stop-instances --instance-ids $INSTANCE_ID
   ```

2. **Proceed to Phase 8**: Implement EC2 setup scripts
   - `scripts/setup/start_ec2.py`
   - `scripts/setup/deploy_via_ssm.py`
   - `scripts/setup/stop_ec2.py`

3. **Document your setup**: Update `setup-summary.txt` with instance IDs and volume IDs

---

**Last Updated:** November 4, 2025  
**Status:** Pre-Phase 8 Verification
