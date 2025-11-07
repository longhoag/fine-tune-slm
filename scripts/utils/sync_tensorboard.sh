#!/bin/bash
# Sync TensorBoard logs from EC2 to S3 and download locally for visualization
# 
# Usage:
#   ./scripts/utils/sync_tensorboard.sh
#
# Then open http://localhost:6006 in your browser

set -e

# Configuration (loaded from SSM Parameter Store)
echo "Loading configuration from SSM Parameter Store..."
INSTANCE_ID=$(aws ssm get-parameter --name /fine-tune-slm/ec2/instance-id --query 'Parameter.Value' --output text)
S3_BUCKET=$(aws ssm get-parameter --name /fine-tune-slm/s3/bucket --query 'Parameter.Value' --output text)
S3_PREFIX="tensorboard-logs"
LOCAL_DIR="./tensorboard-logs"

echo "Configuration:"
echo "  Instance ID: $INSTANCE_ID"
echo "  S3 Bucket: $S3_BUCKET"
echo "  S3 Prefix: $S3_PREFIX"
echo "  Local Directory: $LOCAL_DIR"
echo ""

# Check if instance is running
echo "Checking instance state..."
INSTANCE_STATE=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text)

if [ "$INSTANCE_STATE" != "running" ]; then
  echo "⚠️  Warning: Instance is $INSTANCE_STATE, not running"
  echo "Only downloading existing logs from S3..."
else
  echo "✓ Instance is running"
  
  # Sync from EC2 to S3
  echo ""
  echo "📤 Syncing TensorBoard logs from EC2 to S3..."
  COMMAND_ID=$(aws ssm send-command \
    --instance-id $INSTANCE_ID \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[\"aws s3 sync /mnt/training/checkpoints/logs s3://$S3_BUCKET/$S3_PREFIX --no-progress\"]" \
    --query 'Command.CommandId' \
    --output text)
  
  echo "SSM Command ID: $COMMAND_ID"
  echo "Waiting for sync to complete..."
  
  # Wait for command to complete
  aws ssm wait command-executed \
    --command-id $COMMAND_ID \
    --instance-id $INSTANCE_ID
  
  echo "✓ Sync to S3 completed"
fi

# Download from S3 to local
echo ""
echo "📥 Downloading logs from S3 to local..."
mkdir -p $LOCAL_DIR
aws s3 sync s3://$S3_BUCKET/$S3_PREFIX $LOCAL_DIR --no-progress

echo "✓ Download completed"

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
  echo ""
  echo "❌ Poetry not found. Please install Poetry first:"
  echo "   curl -sSL https://install.python-poetry.org | python3 -"
  exit 1
fi

# Check if TensorBoard is in Poetry environment
if ! poetry run tensorboard --version &> /dev/null; then
  echo ""
  echo "⚠️  TensorBoard not found in Poetry environment. Installing..."
  poetry add tensorboard --group dev
fi

# Start TensorBoard
echo ""
echo "🚀 Starting TensorBoard..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Open your browser to: http://localhost:6006"
echo "  Press Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Suppress TensorFlow warnings (TensorBoard works fine without TensorFlow for PyTorch logs)
export TF_CPP_MIN_LOG_LEVEL=3
poetry run tensorboard --logdir $LOCAL_DIR --host 0.0.0.0 --port 6006 2>&1 | grep -v "TensorFlow installation not found" | grep -v "pkg_resources"
