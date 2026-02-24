FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/app/models/hf \
    TRANSFORMERS_CACHE=/app/models/hf \
    TORCH_HOME=/app/models/torch \
    XDG_CACHE_HOME=/app/models/cache \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# System deps (Whisper + common ML deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy the rest of your app
COPY . /app

# Ensure expected runtime dirs exist (your utils.py expects ./data and ./models) :contentReference[oaicite:1]{index=1}
RUN mkdir -p /app/data/manuals /app/data/images /app/data/index /app/models && \
    mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME"

# Pre-download/cache models at build time (CLIP + Whisper) :contentReference[oaicite:2]{index=2}
RUN python download_models.py

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "ui_app.py", "--server.address=0.0.0.0", "--server.port=8501"]