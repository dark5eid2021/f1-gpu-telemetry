# F1 GPU Telemetry System - GPU Processor Container
# Multi-stage build for optimized production image

FROM nvcr.io/nvidia/rapidsai/rapidsai:24.02-cuda12.0-runtime-ubuntu22.04-py3.10 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=all \
    NVIDIA_VISIBLE_DEVICES=all

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create application user (security best practice)
RUN useradd --create-home --shell /bin/bash f1user && \
    mkdir -p /app /app/data /app/logs /app/cache && \
    chown -R f1user:f1user /app

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional GPU-specific packages
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Set permissions
RUN chown -R f1user:f1user /app && \
    chmod +x scripts/*.sh

# Switch to non-root user
USER f1user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose ports
EXPOSE 8000 8001

# Set entrypoint
ENTRYPOINT ["python", "src/main.py"]