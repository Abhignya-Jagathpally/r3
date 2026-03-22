# R3-MM Pipeline Docker Container
# Based on Python 3.11-slim for minimal image size

FROM python:3.11-slim

LABEL maintainer="Abhignya Jagathpally <abhignya.j@gmail.com>"
LABEL version="0.1.0"
LABEL description="Multiple Myeloma single-cell computational biology pipeline"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    libhdf5-dev \
    pkg-config \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml setup.py* README.md* ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY workflows/ ./workflows/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .

# Create data and results directories
RUN mkdir -p data/{raw,standardized,analysis_ready} && \
    mkdir -p results logs checkpoints

# Create non-root user for security
RUN useradd -m -u 1000 pipeline && \
    chown -R pipeline:pipeline /app

USER pipeline

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "from src.cli import main; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "src", "--config", "configs/pipeline_config.yaml"]

# Expose ports for MLflow UI
EXPOSE 5000

# Labels
LABEL org.opencontainers.image.created="2026-03-18"
LABEL org.opencontainers.image.source="https://github.com/abhignya-j/r3-mm-pipeline"
LABEL org.opencontainers.image.authors="Abhignya Jagathpally"
