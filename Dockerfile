# CrisisMapper Dockerfile
# Multi-stage build for production-ready AI disaster detection platform

# Stage 1: Base image with system dependencies
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic utilities
    wget \
    curl \
    git \
    vim \
    nano \
    htop \
    tree \
    unzip \
    zip \
    tar \
    gzip \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    # Python development
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    # GDAL and geospatial dependencies
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    libspatialite-dev \
    libsqlite3-dev \
    libnetcdf-dev \
    libhdf5-dev \
    # Image processing libraries
    libopencv-dev \
    libopencv-contrib-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    # Scientific computing
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    # Database clients
    postgresql-client \
    redis-tools \
    # Network utilities
    netcat-openbsd \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r crisismapper && useradd -r -g crisismapper crisismapper

# Stage 2: Python environment setup
FROM base as python-env

# Set working directory
WORKDIR /app

# Upgrade pip and install Python dependencies
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional geospatial packages
RUN pip install --no-cache-dir \
    rasterio[s3] \
    geopandas[all] \
    folium \
    leafmap \
    keplergl

# Stage 3: Application setup
FROM python-env as app

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p \
    /app/data \
    /app/models \
    /app/results \
    /app/uploads \
    /app/logs \
    /app/temp

# Download sample models (placeholder - in production, these would be pre-trained)
RUN mkdir -p /app/models && \
    echo "Downloading YOLOv8 models..." && \
    # In a real implementation, you would download actual model weights
    touch /app/models/yolov8n_disaster.pt && \
    touch /app/models/yolov8s_disaster.pt && \
    touch /app/models/yolov8m_disaster.pt && \
    touch /app/models/yolov8l_disaster.pt && \
    touch /app/models/yolov8x_disaster.pt

# Download sample data
RUN mkdir -p /app/data/sample && \
    echo "Downloading sample data..." && \
    # In a real implementation, you would download actual sample data
    touch /app/data/sample/sentinel2_flood.tif && \
    touch /app/data/sample/landsat8_wildfire.tif && \
    touch /app/data/sample/drone_earthquake.jpg

# Set permissions
RUN chown -R crisismapper:crisismapper /app && \
    chmod -R 755 /app

# Switch to non-root user
USER crisismapper

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "inference_api.py"]

# Stage 4: Production image
FROM app as production

# Additional production optimizations
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn[standard] \
    psycopg2-binary \
    redis

# Production configuration
ENV ENVIRONMENT=production
ENV LOG_LEVEL=info
ENV WORKERS=4

# Copy production configuration
COPY docker/production.ini /app/production.ini

# Production command
CMD ["gunicorn", "--config", "production.ini", "inference_api:app"]

# Stage 5: Development image
FROM app as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    jupyter \
    jupyterlab \
    ipywidgets

# Development configuration
ENV ENVIRONMENT=development
ENV LOG_LEVEL=debug
ENV PYTHONPATH=/app

# Copy development configuration
COPY docker/development.ini /app/development.ini

# Development command
CMD ["python", "inference_api.py", "--reload"]

# Stage 6: Jupyter notebook image
FROM app as jupyter

# Install Jupyter dependencies
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    jupyter-dash \
    voila

# Jupyter configuration
ENV JUPYTER_ENABLE_LAB=yes
ENV JUPYTER_TOKEN=changeme

# Expose Jupyter port
EXPOSE 8888

# Jupyter command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Stage 7: GPU-optimized image
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as gpu

# Copy from base image
COPY --from=app /app /app
COPY --from=app /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Set working directory
WORKDIR /app

# Install CUDA-specific packages
RUN pip install --no-cache-dir \
    cupy-cuda11x \
    numba[cuda]

# GPU configuration
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# GPU command
CMD ["python", "inference_api.py"]

# Build arguments for different stages
ARG BUILD_STAGE=app

# Final stage selection
FROM ${BUILD_STAGE} as final

# Labels
LABEL maintainer="CrisisMapper Team"
LABEL version="2.0.0"
LABEL description="AI-powered disaster detection from satellite and drone imagery"
LABEL org.opencontainers.image.title="CrisisMapper"
LABEL org.opencontainers.image.description="AI-powered disaster detection from satellite and drone imagery"
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.authors="CrisisMapper Team"
LABEL org.opencontainers.image.url="https://github.com/your-username/crisismapper"
LABEL org.opencontainers.image.source="https://github.com/your-username/crisismapper"
LABEL org.opencontainers.image.licenses="MIT"

# Final setup
WORKDIR /app
USER crisismapper

# Default command
CMD ["python", "inference_api.py"]