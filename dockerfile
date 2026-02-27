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

# System deps (Whisper + PIL image decode + common runtime libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libjpeg62-turbo \
    zlib1g \
    libpng16-16 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better caching)
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    \
    # Install CPU torch from PyTorch wheel index (more reliable than torch==...+cpu in requirements)
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.6.0 && \
    \
    # Install the rest of requirements (ignore any torch line if present)
    python - <<'PY'
from pathlib import Path
src = Path("/app/requirements.txt").read_text(encoding="utf-8").splitlines()
filtered = []
for line in src:
    s = line.strip()
    if not s or s.startswith("#"):
        filtered.append(line)
        continue
    # drop torch pins (we install torch separately above)
    if s.lower().startswith("torch==") or s.lower().startswith("torch>=") or s.lower().startswith("torch<"):
        continue
    # drop potential +cpu pin variants
    if s.lower().startswith("torch==") and "+cpu" in s.lower():
        continue
    filtered.append(line)
Path("/app/requirements.filtered.txt").write_text("\n".join(filtered) + "\n", encoding="utf-8")
PY
RUN pip install --no-cache-dir -r /app/requirements.filtered.txt

COPY . /app

# Ensure expected runtime dirs exist (utils.py expects ./data and ./models)
RUN mkdir -p /app/data/manuals /app/data/images /app/data/index /app/models && \
    mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME"

# Pre-download/cache models at build time (CLIP + Whisper)
RUN python download_models.py

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "ui_app.py", "--server.address=0.0.0.0", "--server.port=8501"]