#!/bin/bash
# Download TensorBoard logs from S3 and visualize locally
# 
# Usage:
#   # View latest training logs
#   ./scripts/utils/sync_tensorboard.sh
#
#   # View specific training run by timestamp
#   ./scripts/utils/sync_tensorboard.sh 20251111_022951
#
# Then open http://localhost:6006 in your browser

set -e

# Configuration (loaded from SSM Parameter Store)
echo "Loading configuration from SSM Parameter Store..."
S3_BUCKET=$(aws ssm get-parameter --name /fine-tune-slm/s3/bucket --query 'Parameter.Value' --output text)
S3_PREFIX_BASE=$(aws ssm get-parameter --name /fine-tune-slm/s3/prefix --query 'Parameter.Value' --output text)
LOCAL_DIR="./tensorboard-logs"

echo "Configuration:"
echo "  S3 Bucket: $S3_BUCKET"
echo "  S3 Prefix: $S3_PREFIX_BASE"
echo "  Local Directory: $LOCAL_DIR"
echo ""

# Get timestamp argument or find latest
TIMESTAMP="$1"

if [ -z "$TIMESTAMP" ]; then
  echo "🔍 Finding latest training run..."
  
  # List all timestamp directories and get the latest
  TIMESTAMP=$(aws s3 ls s3://$S3_BUCKET/$S3_PREFIX_BASE/ \
    | grep "PRE" \
    | awk '{print $2}' \
    | sed 's/\///' \
    | grep -E "^[0-9]{8}_[0-9]{6}$" \
    | sort -r \
    | head -1)
  
  if [ -z "$TIMESTAMP" ]; then
    echo "❌ No training runs found in S3"
    echo "Expected format: s3://$S3_BUCKET/$S3_PREFIX_BASE/YYYYMMDD_HHMMSS/logs"
    echo ""
    echo "Have you run training yet?"
    echo "  poetry run python scripts/finetune/run_training.py"
    exit 1
  fi
  
  echo "✅ Found latest: $TIMESTAMP"
else
  echo "📌 Using specified timestamp: $TIMESTAMP"
fi

S3_LOGS_PATH="s3://$S3_BUCKET/$S3_PREFIX_BASE/$TIMESTAMP/logs"

echo ""
echo "� Downloading logs from S3..."
echo "  Source: $S3_LOGS_PATH"
echo "  Target: $LOCAL_DIR"

# Check if logs exist in S3
if ! aws s3 ls $S3_LOGS_PATH/ > /dev/null 2>&1; then
  echo ""
  echo "❌ No logs found at: $S3_LOGS_PATH"
  echo ""
  echo "Available training runs:"
  aws s3 ls s3://$S3_BUCKET/$S3_PREFIX_BASE/ | grep "PRE" | awk '{print "  -", $2}' | sed 's/\/$//'
  echo ""
  echo "Note: TensorBoard logs are automatically uploaded after training completes."
  exit 1
fi

# Download from S3 to local
mkdir -p $LOCAL_DIR
aws s3 sync $S3_LOGS_PATH $LOCAL_DIR --delete --no-progress

echo "✅ Download completed"
echo ""

# Count events
EVENT_COUNT=$(find $LOCAL_DIR -name "events.out.tfevents.*" | wc -l | tr -d ' ')
echo "📊 Found $EVENT_COUNT TensorBoard event file(s)"

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
