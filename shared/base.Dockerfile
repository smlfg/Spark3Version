# Spark Cluster - Shared Base Image
# Base Docker image for all agents with NVIDIA GPU support

FROM nvcr.io/nvidia/pytorch:24.10-py3

LABEL maintainer="Spark Cluster Team"
LABEL description="Base image for Spark cluster agents with CUDA and PyTorch"
LABEL version="1.0.0"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    # NVIDIA tools
    nvidia-container-toolkit \
    cuda-toolkit-12-2 \
    # Network tools
    openssh-client \
    openssh-server \
    net-tools \
    iputils-ping \
    iperf3 \
    iproute2 \
    # Development tools
    vim \
    nano \
    htop \
    tmux \
    screen \
    # Python development
    python3-dev \
    python3-pip \
    python3-setuptools \
    # Infiniband support (optional)
    infiniband-diags \
    ibverbs-utils \
    rdma-core \
    # Monitoring tools
    sysstat \
    nethogs \
    iotop \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install base Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install common Python dependencies
RUN pip install --no-cache-dir \
    # Core libraries
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    pandas>=2.0.0 \
    # Deep learning
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    torchaudio>=2.0.0 \
    # Distributed training
    horovod[pytorch]>=0.28.0 \
    # Monitoring
    tensorboard>=2.13.0 \
    wandb>=0.15.0 \
    # Utilities
    pyyaml>=6.0 \
    requests>=2.31.0 \
    tqdm>=4.65.0 \
    psutil>=5.9.0 \
    click>=8.1.0 \
    # Configuration management
    python-dotenv>=1.0.0 \
    pydantic>=2.0.0 \
    # Testing
    pytest>=7.4.0 \
    pytest-cov>=4.1.0

# Install NCCL for multi-GPU communication
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnccl2 \
    libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create application directories
RUN mkdir -p /app /data /models /logs /shared

# Create non-root user for running applications
RUN useradd -m -u 1000 -s /bin/bash spark && \
    chown -R spark:spark /app /data /models /logs /shared

# SSH configuration for multi-node communication
RUN mkdir -p /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# Set working directory
WORKDIR /app

# Add health check script
COPY shared/health_check.py /usr/local/bin/health_check.py
RUN chmod +x /usr/local/bin/health_check.py

# Add shared utilities
COPY shared/utils /app/shared/utils
ENV PYTHONPATH=/app:${PYTHONPATH}

# Switch to non-root user
USER spark

# Default command
CMD ["/bin/bash"]

# Expose common ports
# 22: SSH
# 6006: TensorBoard
# 8888: Jupyter
# 8000-8010: Custom services
EXPOSE 22 6006 8888 8000 8001 8002 8003 8004 8005 8006 8007 8008 8009 8010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 /usr/local/bin/health_check.py --self-check || exit 1
