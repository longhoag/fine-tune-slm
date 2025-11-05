"""
AWS utilities for EC2, SSM, S3, and Secrets Manager operations.

Complete implementation with boto3 for all AWS service interactions.
"""

import time
import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from loguru import logger
import boto3
from botocore.exceptions import ClientError, WaiterError


class AWSClient:
    """Base AWS client with common functionality."""
    
    def __init__(self, region: str = "us-east-1"):
        """Initialize AWS client with all service clients."""
        self.region = region
        self.ec2 = boto3.client('ec2', region_name=region)
        self.ssm = boto3.client('ssm', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.secrets = boto3.client('secretsmanager', region_name=region)
        self.logs = boto3.client('logs', region_name=region)
        logger.info(f"Initialized AWS clients for region: {region}")


class EC2Manager:
    """Manage EC2 instance lifecycle."""
    
    def __init__(self, client: AWSClient):
        """Initialize EC2 manager."""
        self.client = client
        
    def start_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Start EC2 instance and wait for running state.
        
        Args:
            instance_id: EC2 instance ID
            
        Returns:
            Instance state information
        """
        try:
            logger.info(f"Starting EC2 instance: {instance_id}")
            response = self.client.ec2.start_instances(InstanceIds=[instance_id])
            
            # Wait for instance to be running
            logger.info("Waiting for instance to reach 'running' state...")
            waiter = self.client.ec2.get_waiter('instance_running')
            waiter.wait(InstanceIds=[instance_id])
            
            # Wait for status checks
            logger.info("Waiting for status checks to pass...")
            waiter = self.client.ec2.get_waiter('instance_status_ok')
            waiter.wait(InstanceIds=[instance_id])
            
            logger.info(f"Instance {instance_id} is running and ready")
            return response['StartingInstances'][0]
            
        except ClientError as e:
            logger.error(f"Failed to start instance: {e}")
            raise
        except WaiterError as e:
            logger.error(f"Instance failed to start properly: {e}")
            raise
            
    def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Stop EC2 instance.
        
        Args:
            instance_id: EC2 instance ID
            
        Returns:
            Instance state information
        """
        try:
            logger.info(f"Stopping EC2 instance: {instance_id}")
            response = self.client.ec2.stop_instances(InstanceIds=[instance_id])
            
            # Wait for instance to stop
            logger.info("Waiting for instance to stop...")
            waiter = self.client.ec2.get_waiter('instance_stopped')
            waiter.wait(InstanceIds=[instance_id])
            
            logger.info(f"Instance {instance_id} stopped")
            return response['StoppingInstances'][0]
            
        except ClientError as e:
            logger.error(f"Failed to stop instance: {e}")
            raise
            
    def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """
        Get current instance status.
        
        Args:
            instance_id: EC2 instance ID
            
        Returns:
            Instance status information
        """
        try:
            response = self.client.ec2.describe_instances(InstanceIds=[instance_id])
            instance = response['Reservations'][0]['Instances'][0]
            
            status_info = {
                'instance_id': instance_id,
                'state': instance['State']['Name'],
                'instance_type': instance['InstanceType'],
                'availability_zone': instance['Placement']['AvailabilityZone'],
                'private_ip': instance.get('PrivateIpAddress'),
                'public_ip': instance.get('PublicIpAddress')
            }
            
            logger.info(f"Instance status: {status_info['state']}")
            return status_info
            
        except ClientError as e:
            logger.error(f"Failed to get instance status: {e}")
            raise


class SSMManager:
    """Manage SSM commands and sessions."""
    
    def __init__(self, client: AWSClient):
        """Initialize SSM manager."""
        self.client = client
        
    def send_command(
        self,
        instance_id: str,
        commands: List[str],
        comment: Optional[str] = None,
        working_directory: str = "/home/ubuntu",
        timeout_seconds: int = 3600
    ) -> str:
        """
        Send command to EC2 instance via SSM.
        
        Args:
            instance_id: EC2 instance ID
            commands: List of shell commands to execute
            comment: Optional comment for the command
            working_directory: Directory to run commands in
            timeout_seconds: Command timeout
            
        Returns:
            Command ID for tracking
        """
        try:
            logger.info(f"Sending SSM command to {instance_id}")
            logger.debug(f"Commands: {commands}")
            
            response = self.client.ssm.send_command(
                InstanceIds=[instance_id],
                DocumentName="AWS-RunShellScript",
                Parameters={
                    'commands': commands,
                    'workingDirectory': [working_directory],
                    'executionTimeout': [str(timeout_seconds)]
                },
                Comment=comment or "Fine-tune SLM automated command",
                TimeoutSeconds=timeout_seconds
            )
            
            command_id = response['Command']['CommandId']
            logger.info(f"Command sent successfully. Command ID: {command_id}")
            return command_id
            
        except ClientError as e:
            logger.error(f"Failed to send SSM command: {e}")
            raise
            
    def get_command_output(self, command_id: str, instance_id: str) -> Dict[str, Any]:
        """
        Get command execution output.
        
        Args:
            command_id: SSM command ID
            instance_id: EC2 instance ID
            
        Returns:
            Command output with stdout, stderr, and status
        """
        try:
            response = self.client.ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=instance_id
            )
            
            return {
                'status': response['Status'],
                'stdout': response.get('StandardOutputContent', ''),
                'stderr': response.get('StandardErrorContent', ''),
                'exit_code': response.get('ResponseCode', -1)
            }
            
        except ClientError as e:
            # InvocationDoesNotExist is expected when command is just registered
            if 'InvocationDoesNotExist' in str(e):
                logger.debug(f"Command invocation not yet available (expected): {e}")
            else:
                logger.error(f"Failed to get command output: {e}")
            raise
            
    def wait_for_command(
        self,
        command_id: str,
        instance_id: str,
        timeout: int = 3600,
        poll_interval: int = 5
    ) -> Dict[str, Any]:
        """
        Wait for command completion and return output.
        
        Args:
            command_id: SSM command ID
            instance_id: EC2 instance ID
            timeout: Maximum time to wait in seconds
            poll_interval: Seconds between status checks
            
        Returns:
            Command output
        """
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Command {command_id} did not complete within {timeout}s")
                
            try:
                output = self.get_command_output(command_id, instance_id)
                status = output['status']
                
                if status in ['Success', 'Failed', 'Cancelled', 'TimedOut']:
                    logger.info(f"Command completed with status: {status}")
                    return output
                    
                logger.debug(f"Command status: {status}. Waiting...")
                time.sleep(poll_interval)
                
            except ClientError as e:
                if 'InvocationDoesNotExist' in str(e):
                    logger.debug("Command not yet registered, retrying...")
                    time.sleep(poll_interval)
                else:
                    raise


class S3Manager:
    """Manage S3 operations."""
    
    def __init__(self, client: AWSClient):
        """Initialize S3 manager."""
        self.client = client
        
    def upload_directory(
        self,
        local_path: str,
        bucket: str,
        prefix: str,
        exclude_patterns: Optional[List[str]] = None
    ):
        """
        Upload directory to S3.
        
        Args:
            local_path: Local directory path
            bucket: S3 bucket name
            prefix: S3 key prefix
            exclude_patterns: File patterns to exclude
        """
        local_path = Path(local_path)
        exclude_patterns = exclude_patterns or []
        
        if not local_path.exists():
            raise FileNotFoundError(f"Local path does not exist: {local_path}")
            
        logger.info(f"Uploading {local_path} to s3://{bucket}/{prefix}")
        
        uploaded_count = 0
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                # Check exclusions
                if any(pattern in str(file_path) for pattern in exclude_patterns):
                    continue
                    
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{prefix}/{relative_path}".replace('\\', '/')
                
                try:
                    self.client.s3.upload_file(
                        str(file_path),
                        bucket,
                        s3_key
                    )
                    uploaded_count += 1
                    logger.debug(f"Uploaded: {s3_key}")
                    
                except ClientError as e:
                    logger.error(f"Failed to upload {file_path}: {e}")
                    raise
                    
        logger.info(f"Successfully uploaded {uploaded_count} files to S3")
        
    def download_directory(self, bucket: str, prefix: str, local_path: str):
        """
        Download directory from S3.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix
            local_path: Local directory path
        """
        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading s3://{bucket}/{prefix} to {local_path}")
        
        try:
            paginator = self.client.s3.get_paginator('list_objects_v2')
            downloaded_count = 0
            
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if 'Contents' not in page:
                    logger.warning(f"No objects found with prefix: {prefix}")
                    return
                    
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    relative_path = s3_key[len(prefix):].lstrip('/')
                    
                    if not relative_path:
                        continue
                        
                    file_path = local_path / relative_path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    self.client.s3.download_file(bucket, s3_key, str(file_path))
                    downloaded_count += 1
                    logger.debug(f"Downloaded: {s3_key}")
                    
            logger.info(f"Successfully downloaded {downloaded_count} files from S3")
            
        except ClientError as e:
            logger.error(f"Failed to download from S3: {e}")
            raise
            
    def verify_artifacts(self, bucket: str, prefix: str) -> bool:
        """
        Verify model artifacts exist in S3.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix
            
        Returns:
            True if artifacts exist
        """
        try:
            response = self.client.s3.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=1
            )
            
            exists = 'Contents' in response and len(response['Contents']) > 0
            
            if exists:
                logger.info(f"Artifacts found at s3://{bucket}/{prefix}")
            else:
                logger.warning(f"No artifacts found at s3://{bucket}/{prefix}")
                
            return exists
            
        except ClientError as e:
            logger.error(f"Failed to verify S3 artifacts: {e}")
            return False


class SecretsManager:
    """Manage secrets and parameters retrieval."""
    
    def __init__(self, client: AWSClient):
        """Initialize secrets manager."""
        self.client = client
        
    def get_secret(self, secret_name: str) -> str:
        """
        Retrieve secret from AWS Secrets Manager.
        
        Args:
            secret_name: Name of secret in Secrets Manager
            
        Returns:
            Secret value as string
        """
        try:
            response = self.client.secrets.get_secret_value(SecretId=secret_name)
            secret = response['SecretString']
            logger.info(f"Retrieved secret: {secret_name}")
            return secret
            
        except ClientError as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            raise
            
    def get_secret_json(self, secret_name: str) -> Dict[str, Any]:
        """
        Retrieve secret and parse as JSON.
        
        Args:
            secret_name: Name of secret in Secrets Manager
            
        Returns:
            Parsed JSON secret
        """
        secret_string = self.get_secret(secret_name)
        try:
            return json.loads(secret_string)
        except json.JSONDecodeError as e:
            logger.error(f"Secret {secret_name} is not valid JSON: {e}")
            raise
            
    def get_parameter(self, parameter_name: str, with_decryption: bool = True) -> str:
        """
        Retrieve parameter from SSM Parameter Store.
        
        Args:
            parameter_name: Parameter name (e.g., /fine-tune-slm/ec2/instance-id)
            with_decryption: Decrypt SecureString parameters
            
        Returns:
            Parameter value
        """
        try:
            response = self.client.ssm.get_parameter(
                Name=parameter_name,
                WithDecryption=with_decryption
            )
            value = response['Parameter']['Value']
            logger.debug(f"Retrieved parameter: {parameter_name}")
            return value
            
        except ClientError as e:
            logger.error(f"Failed to retrieve parameter {parameter_name}: {e}")
            raise
            
    def get_parameters_by_path(
        self,
        path: str,
        recursive: bool = True,
        with_decryption: bool = True
    ) -> Dict[str, str]:
        """
        Retrieve all parameters under a path.
        
        Args:
            path: Parameter path (e.g., /fine-tune-slm/)
            recursive: Get all parameters in hierarchy
            with_decryption: Decrypt SecureString parameters
            
        Returns:
            Dictionary of parameter names to values
        """
        parameters = {}
        
        try:
            paginator = self.client.ssm.get_paginator('get_parameters_by_path')
            
            for page in paginator.paginate(
                Path=path,
                Recursive=recursive,
                WithDecryption=with_decryption
            ):
                for param in page['Parameters']:
                    parameters[param['Name']] = param['Value']
                    
            logger.info(f"Retrieved {len(parameters)} parameters from path: {path}")
            return parameters
            
        except ClientError as e:
            logger.error(f"Failed to retrieve parameters from path {path}: {e}")
            raise
            
    def put_parameter(
        self,
        name: str,
        value: str,
        description: str = "",
        parameter_type: str = "String",
        overwrite: bool = False
    ):
        """
        Create or update SSM parameter.
        
        Args:
            name: Parameter name
            value: Parameter value
            description: Parameter description
            parameter_type: String, StringList, or SecureString
            overwrite: Overwrite existing parameter
        """
        try:
            self.client.ssm.put_parameter(
                Name=name,
                Value=value,
                Description=description,
                Type=parameter_type,
                Overwrite=overwrite
            )
            logger.info(f"{'Updated' if overwrite else 'Created'} parameter: {name}")
            
        except ClientError as e:
            logger.error(f"Failed to put parameter {name}: {e}")
            raise
