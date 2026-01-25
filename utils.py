import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

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
class ImageFilterConfig:
    """Configuration for filtering out low-quality/irrelevant images during ingestion."""
    # Minimum dimensions - filters tiny fragments
    min_width: int = 80
    min_height: int = 80
    
    # Aspect ratio limits (width/height) - filters thin strips from table cells
    min_aspect_ratio: float = 0.2   # No thinner than 1:5
    max_aspect_ratio: float = 5.0   # No wider than 5:1
    
    # Minimum area in pixels - filters tiny icons/bullets
    min_area: int = 10000  # ~100x100 equivalent


@dataclass
class RetrievalConfig:
    # Distance thresholds - LOWER = stricter matching
    text_distance_threshold: float = 1000.0
    image_distance_threshold: float = 1.2  # Tightened from 2.0 for better precision
    
    # Weights for combined scoring
    text_weight: float = 0.6
    image_weight: float = 0.4
    
    # Page alignment settings
    enable_page_alignment: bool = True
    page_alignment_bonus: float = 0.15  # Reduced to prevent weak matches from page alignment
    
    # Minimum scores for inclusion
    min_image_relevance: float = 0.4  # Increased from 0.3
    
    # Reciprocal Rank Fusion
    use_rrf: bool = True
    rrf_k: int = 60


IMAGE_FILTER_CONFIG = ImageFilterConfig()
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
    """Convert L2 distance to a 0-1 score (higher = better match)."""
    return max(0.0, 1.0 - (distance / max_distance))


def check_image_dimensions(width: int, height: int, config: ImageFilterConfig = None) -> bool:
    """Check if image meets dimension/aspect ratio requirements."""
    if config is None:
        config = IMAGE_FILTER_CONFIG
    
    # Check minimum dimensions
    if width < config.min_width or height < config.min_height:
        return False
    
    # Check minimum area
    area = width * height
    if area < config.min_area:
        return False
    
    # Check aspect ratio
    aspect_ratio = width / height if height > 0 else 0
    if aspect_ratio < config.min_aspect_ratio or aspect_ratio > config.max_aspect_ratio:
        return False
    
    return True