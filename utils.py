import os
import logging
from pathlib import Path
from dataclasses import dataclass

_SCRIPT_DIR = Path(__file__).parent.resolve() if '__file__' in dir() else Path.cwd()
BASE_DIR = _SCRIPT_DIR
DATA_DIR = BASE_DIR / "data"
MANUALS_DIR = DATA_DIR / "manuals"
IMAGES_DIR = DATA_DIR / "images"
INDEX_DIR = DATA_DIR / "index"
MODELS_DIR = BASE_DIR / "models"

for d in [MANUALS_DIR, IMAGES_DIR, INDEX_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

OLLAMA_MODEL_NAME = "mistral:7b"
VLM_MODEL_NAME = "llava:7b"
TEXT_EMBED_MODEL = "nomic-embed-text"
IMAGE_EMBED_MODEL = "openai/clip-vit-base-patch32"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

TOP_K_TEXT = 5
TOP_K_IMAGE = 3

@dataclass
class RetrievalConfig:
    text_distance_threshold: float = 1000.0
    image_distance_threshold: float = 2.0
    text_weight: float = 0.6
    image_weight: float = 0.4
    enable_page_alignment: bool = True
    page_alignment_bonus: float = 0.2
    min_image_relevance: float = 0.3
    use_rrf: bool = True
    rrf_k: int = 60

RETRIEVAL_CONFIG = RetrievalConfig()

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

def validate_file_exists(filepath: str) -> bool:
    path = Path(filepath)
    return path.exists() and path.is_file()

def normalize_l2_distance_to_score(distance: float, max_distance: float = 2.0) -> float:
    return max(0.0, 1.0 - (distance / max_distance))