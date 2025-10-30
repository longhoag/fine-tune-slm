FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 12.1 support
RUN pip3 install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install AWS CLI for S3/SSM operations
RUN apt-get update && apt-get install -y unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip && \
    rm -rf /var/lib/apt/lists/*

# Copy configuration files
COPY config/ /workspace/config/

# Copy training scripts
COPY src/ /workspace/src/

# Copy dataset
COPY synthetic-instruction-tuning-dataset/ /workspace/data/

# Create directories for outputs
RUN mkdir -p /mnt/training/checkpoints /mnt/training/logs

# Set Python path
ENV PYTHONPATH=/workspace

# Default entrypoint
ENTRYPOINT ["python3"]
CMD ["-m", "src.train"]
