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

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libjpeg62-turbo zlib1g libpng16-16 \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.6.0 && \
    python - <<'PY'
from pathlib import Path
src = Path("/app/requirements.txt").read_text(encoding="utf-8").splitlines()
filtered = [l for l in src if not l.strip().lower().startswith("torch==") and not l.strip().lower().startswith("torch>=") and not l.strip().lower().startswith("torch<")]
Path("/app/requirements.filtered.txt").write_text("\n".join(filtered) + "\n", encoding="utf-8")
PY
RUN pip install --no-cache-dir -r /app/requirements.filtered.txt

COPY . /app

RUN mkdir -p /app/data/manuals /app/data/images /app/data/index /app/models && \
    mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME"

RUN python download_models.py

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "ui_app.py", "--server.address=0.0.0.0", "--server.port=8501"]