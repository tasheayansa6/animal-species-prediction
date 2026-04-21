# ============================================================
# Dockerfile — Animal Species Prediction
# ============================================================
FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .
RUN pip install -e .

# Create output directories
RUN mkdir -p \
    data/raw \
    data/processed \
    data/metadata \
    models/checkpoints \
    models/final \
    models/logs \
    outputs/figures \
    outputs/reports \
    outputs/predictions

EXPOSE 5000

# Default: run full pipeline
CMD ["python", "scripts/run_pipeline.py", "--help"]

# ── Usage Examples ──────────────────────────────────────────────────
# Build:
#   docker build -t animal-classifier .
#
# Full pipeline:
#   docker run --gpus all \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/outputs:/app/outputs \
#     animal-classifier python scripts/run_pipeline.py
#
# Web app (after training):
#   docker run -p 5000:5000 \
#     -v $(pwd)/models:/app/models \
#     animal-classifier python app/app.py
#
# Predict on images:
#   docker run \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/images:/app/images \
#     animal-classifier python scripts/predict.py --image-dir /app/images
