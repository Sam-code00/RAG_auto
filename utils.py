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

OLLAMA_MODEL_NAME = "gemma3:4b"

TEXT_EMBED_MODEL = "nomic-embed-text"
IMAGE_EMBED_MODEL = "openai/clip-vit-base-patch32"

# Chunking
CHUNK_SIZE = 600  # characters approx
CHUNK_OVERLAP = 100

# To retrieve
TOP_K_TEXT = 5
TOP_K_IMAGE = 3

def setup_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
