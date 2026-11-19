FROM python:3.11-slim-bookworm

# Prevent interactive prompts and Python buffering
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install OS libraries required for OpenCV, PyTorch, and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement definitions
COPY requirements.txt .

# Install CPU-optimized PyTorch first (drastically reduces image size on CPU/EC2 instances),
# then install remaining pipeline dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install "numpy==1.26.4" && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copy backend application source code
COPY . .

# Create required directories for persistent storage and model caches
RUN mkdir -p /app/data /app/model_cache /app/temp_downloads /app/debug_bundles

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
