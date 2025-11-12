# Spark Cluster - Minimal Base Image
# Lightweight base image for non-GPU agents

FROM nvidia/cuda:12.2.0-base-ubuntu22.04

LABEL maintainer="Spark Cluster Team"
LABEL description="Minimal base image for Spark cluster agents"
LABEL version="1.0.0"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH=/usr/local/bin:${PATH}

# Install minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install minimal Python packages
RUN pip3 install --no-cache-dir \
    pyyaml>=6.0 \
    requests>=2.31.0 \
    psutil>=5.9.0 \
    pydantic>=2.0.0

# Create directories
RUN mkdir -p /app /data /logs

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash spark && \
    chown -R spark:spark /app /data /logs

WORKDIR /app

# Add health check
COPY shared/health_check.py /usr/local/bin/health_check.py
RUN chmod +x /usr/local/bin/health_check.py

USER spark

CMD ["/bin/bash"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 /usr/local/bin/health_check.py --self-check || exit 1
