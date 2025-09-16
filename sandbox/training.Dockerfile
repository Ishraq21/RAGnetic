# GPU Training Dockerfile for RAGnetic
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core ML libraries
RUN pip3 install \
    transformers>=4.30.0 \
    datasets>=2.12.0 \
    accelerate>=0.20.0 \
    peft>=0.4.0 \
    trl>=0.4.0 \
    bitsandbytes>=0.39.0 \
    scipy \
    scikit-learn \
    numpy \
    pandas \
    tqdm \
    wandb \
    tensorboard

# Install additional utilities
RUN pip3 install \
    requests \
    aiohttp \
    pydantic \
    pyyaml \
    jsonlines \
    huggingface-hub

# Create working directory
WORKDIR /work

# Create directories for data and outputs
RUN mkdir -p /work/data /work/output /work/logs

# Copy the runner script
COPY runner.py /work/runner.py

# Make runner executable
RUN chmod +x /work/runner.py

# Set entrypoint
ENTRYPOINT ["python3", "/work/runner.py"]

# Default command
CMD ["--help"]
