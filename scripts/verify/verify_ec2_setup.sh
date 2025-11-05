#!/bin/bash
#
# EC2 Infrastructure Quick Verification Script
# 
# This script performs automated verification of your EC2 setup
# before proceeding to Phase 8 automation scripts.
#
# Usage: ./scripts/verify/verify_ec2_setup.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo "EC2 Infrastructure Verification"
echo "==========================================${NC}"
echo ""

# Track failures
FAILURES=0

# Helper functions
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    FAILURES=$((FAILURES + 1))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# 1. Verify Instance ID Parameter
echo -e "${BLUE}[1/10] Checking EC2 Instance ID...${NC}"
INSTANCE_ID=$(aws ssm get-parameter \
    --name /fine-tune-slm/ec2/instance-id \
    --query 'Parameter.Value' \
    --output text 2>/dev/null)

if [ -n "$INSTANCE_ID" ]; then
    check_pass "Instance ID: $INSTANCE_ID"
else
    check_fail "Instance ID not found in SSM Parameter Store"
    exit 1
fi

# 2. Check Instance State
echo -e "\n${BLUE}[2/10] Checking EC2 Instance State...${NC}"
INSTANCE_STATE=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].State.Name' \
    --output text 2>/dev/null)

if [ "$INSTANCE_STATE" == "running" ]; then
    check_pass "Instance State: $INSTANCE_STATE"
elif [ "$INSTANCE_STATE" == "stopped" ]; then
    check_warn "Instance State: $INSTANCE_STATE (needs to be started)"
else
    check_fail "Instance State: $INSTANCE_STATE"
fi

# 3. Verify Instance Type
echo -e "\n${BLUE}[3/10] Checking Instance Type...${NC}"
INSTANCE_TYPE=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].InstanceType' \
    --output text 2>/dev/null)

if [ "$INSTANCE_TYPE" == "g6.2xlarge" ]; then
    check_pass "Instance Type: $INSTANCE_TYPE"
else
    check_fail "Instance Type: $INSTANCE_TYPE (expected: g6.2xlarge)"
fi

# 4. Check SSM Connectivity
echo -e "\n${BLUE}[4/10] Checking SSM Connectivity...${NC}"
SSM_STATUS=$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --query 'InstanceInformationList[0].PingStatus' \
    --output text 2>/dev/null)

if [ "$SSM_STATUS" == "Online" ]; then
    check_pass "SSM Status: $SSM_STATUS"
elif [ "$SSM_STATUS" == "ConnectionLost" ]; then
    check_fail "SSM Status: $SSM_STATUS (check SSM agent on instance)"
else
    check_warn "SSM Status: $SSM_STATUS (instance may need to be running)"
fi

# 5. Check EBS Volume
echo -e "\n${BLUE}[5/10] Checking Checkpoint EBS Volume...${NC}"
VOLUME_ID=$(aws ssm get-parameter \
    --name /fine-tune-slm/ebs/volume-id \
    --query 'Parameter.Value' \
    --output text 2>/dev/null)

if [ -n "$VOLUME_ID" ]; then
    check_pass "Volume ID: $VOLUME_ID"
    
    VOLUME_STATE=$(aws ec2 describe-volumes \
        --volume-ids $VOLUME_ID \
        --query 'Volumes[0].State' \
        --output text 2>/dev/null)
    
    if [ "$VOLUME_STATE" == "in-use" ]; then
        check_pass "Volume State: $VOLUME_STATE"
    elif [ "$VOLUME_STATE" == "available" ]; then
        check_warn "Volume State: $VOLUME_STATE (needs to be attached)"
    else
        check_fail "Volume State: $VOLUME_STATE"
    fi
else
    check_fail "Volume ID not found in SSM Parameter Store"
fi

# 6. Check Availability Zone Match
echo -e "\n${BLUE}[6/10] Checking Availability Zone Match...${NC}"
EC2_AZ=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].Placement.AvailabilityZone' \
    --output text 2>/dev/null)

EBS_AZ=$(aws ec2 describe-volumes \
    --volume-ids $VOLUME_ID \
    --query 'Volumes[0].AvailabilityZone' \
    --output text 2>/dev/null)

if [ "$EC2_AZ" == "$EBS_AZ" ]; then
    check_pass "Availability Zones Match: $EC2_AZ"
else
    check_fail "AZ Mismatch - EC2: $EC2_AZ, EBS: $EBS_AZ"
    echo -e "${YELLOW}   See EC2_VERIFICATION_CHECKLIST.md for fix${NC}"
fi

# 7. Check IAM Role
echo -e "\n${BLUE}[7/10] Checking IAM Instance Profile...${NC}"
IAM_ARN=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].IamInstanceProfile.Arn' \
    --output text 2>/dev/null)

if [ "$IAM_ARN" != "None" ] && [ -n "$IAM_ARN" ]; then
    ROLE_NAME=$(echo $IAM_ARN | cut -d'/' -f2)
    check_pass "IAM Role: $ROLE_NAME"
else
    check_fail "No IAM role attached to instance"
fi

# 8. Check SSM Parameters
echo -e "\n${BLUE}[8/10] Checking SSM Parameters...${NC}"
PARAM_COUNT=$(aws ssm get-parameters-by-path \
    --path /fine-tune-slm \
    --recursive \
    --query 'length(Parameters)' \
    --output text 2>/dev/null | tr -d '\n' | tr -d ' ')

if [ -n "$PARAM_COUNT" ] && [ "$PARAM_COUNT" -ge 15 ] 2>/dev/null; then
    check_pass "SSM Parameters: $PARAM_COUNT found"
else
    check_warn "SSM Parameters: $PARAM_COUNT found (expected ~17)"
fi

# 9. Check S3 Bucket
echo -e "\n${BLUE}[9/10] Checking S3 Bucket...${NC}"
S3_BUCKET=$(aws ssm get-parameter \
    --name /fine-tune-slm/s3/bucket \
    --query 'Parameter.Value' \
    --output text 2>/dev/null)

if aws s3 ls s3://$S3_BUCKET 2>/dev/null; then
    check_pass "S3 Bucket: $S3_BUCKET (accessible)"
else
    check_fail "S3 Bucket: $S3_BUCKET (not accessible)"
fi

# 10. Check ECR Repository
echo -e "\n${BLUE}[10/10] Checking ECR Repository...${NC}"
ECR_REPO=$(aws ssm get-parameter \
    --name /fine-tune-slm/ecr/repository \
    --query 'Parameter.Value' \
    --output text 2>/dev/null)

IMAGE_COUNT=$(aws ecr list-images \
    --repository-name $ECR_REPO \
    --query 'length(imageIds)' \
    --output text 2>/dev/null)

if [ "$IMAGE_COUNT" -gt 0 ]; then
    check_pass "ECR Repository: $ECR_REPO ($IMAGE_COUNT images)"
else
    check_warn "ECR Repository: $ECR_REPO (no images found)"
fi

# Summary
echo ""
echo -e "${BLUE}==========================================${NC}"
if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}✅ All Checks Passed!${NC}"
    echo ""
    echo "Your EC2 infrastructure is ready for Phase 8."
    echo ""
    echo "Next steps:"
    echo "  1. Review EC2_VERIFICATION_CHECKLIST.md for detailed tests"
    echo "  2. Proceed to implement Phase 8 scripts"
    echo "  3. Stop EC2 instance to save costs:"
    echo "     aws ec2 stop-instances --instance-ids $INSTANCE_ID"
else
    echo -e "${RED}❌ $FAILURES Check(s) Failed${NC}"
    echo ""
    echo "Please fix the issues above before proceeding."
    echo "See EC2_VERIFICATION_CHECKLIST.md for troubleshooting."
    exit 1
fi
echo -e "${BLUE}==========================================${NC}"
