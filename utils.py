import os
import logging
from pathlib import Path


# Paths
BASE_DIR = Path(os.getcwd())
DATA_DIR = BASE_DIR / "data"
MANUALS_DIR = DATA_DIR / "manuals"
IMAGES_DIR = DATA_DIR / "images"
INDEX_DIR = DATA_DIR / "index"
MODELS_DIR = BASE_DIR / "models"

for d in [MANUALS_DIR, IMAGES_DIR, INDEX_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

OLLAMA_MODEL_NAME = "mistral:7b"
VLM_MODEL_NAME = "qwen3-vl:8b"
# VLM_MODEL_NAME = "moondream:latest"

TEXT_EMBED_MODEL = "nomic-embed-text"
IMAGE_EMBED_MODEL = "openai/clip-vit-base-patch32"

# Chunking: structure-aware settings
CHUNK_SIZE = 1500       # Max chars per chunk (sections are semantic, so larger is fine)
CHUNK_OVERLAP = 200     # Overlap for recursive fallback splits
MIN_CHUNK_SIZE = 100    # Skip chunks smaller than this
TABLE_MAX_CHARS = 3000  # Tables can be larger since they're self-contained

# To retrieve
TOP_K_TEXT = 5
TOP_K_IMAGE = 3

# Image filtering
IMAGE_COSINE_MIN = 0.15
IMAGE_MAX_RETURN = 3
IMAGE_GAP_MIN = 0.08


def setup_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger