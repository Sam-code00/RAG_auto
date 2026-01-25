import fitz
import os
import uuid
import pickle
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from io import BytesIO

from utils import (
    IMAGES_DIR, INDEX_DIR, TEXT_EMBED_MODEL, IMAGE_EMBED_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, setup_logger, validate_file_exists,
    check_image_dimensions, IMAGE_FILTER_CONFIG, ImageFilterConfig
)

logger = setup_logger(__name__)


@dataclass
class TextChunk:
    id: str
    doc_id: str
    page: int
    text: str
    char_start: int = 0
    char_end: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class ImageMetadata:
    id: str
    doc_id: str
    page: int
    filepath: str
    context: str
    width: int = 0
    height: int = 0
    img_index: int = 0
    nearby_text: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PDFProcessor:
    def __init__(self, image_filter_config: ImageFilterConfig = None):
        self.clip_model = CLIPModel.from_pretrained(IMAGE_EMBED_MODEL, local_files_only=True)
        self.clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBED_MODEL, local_files_only=True)
        self.image_filter_config = image_filter_config or IMAGE_FILTER_CONFIG

    def process_pdf(self, pdf_path: str, progress_callback: Callable[[int, int, str], None] = None) -> Tuple[List[Dict], List[Dict]]:
        doc = fitz.open(pdf_path)
        doc_id = os.path.basename(pdf_path)
        total_pages = len(doc)
        
        text_chunks = []
        images_metadata = []
        page_texts = {}
        
        filtered_count = 0

        for page_num, page in enumerate(doc):
            if progress_callback:
                progress_callback(page_num + 1, total_pages, f"Extracting page {page_num + 1}/{total_pages}")
            
            page_number = page_num + 1
            text = page.get_text()
            page_texts[page_number] = text
            
            chunks = self._chunk_text_with_positions(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for chunk_text, char_start, char_end in chunks:
                chunk = TextChunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    page=page_number,
                    text=chunk_text,
                    char_start=char_start,
                    char_end=char_end
                )
                text_chunks.append(chunk.to_dict())

            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                img_meta, was_filtered = self._extract_and_validate_image(
                    doc, img, doc_id, page_number, img_index, text
                )
                if img_meta:
                    images_metadata.append(img_meta.to_dict())
                elif was_filtered:
                    filtered_count += 1

        doc.close()
        
        logger.info(f"Processed {total_pages} pages: {len(text_chunks)} text chunks, "
                   f"{len(images_metadata)} images kept, {filtered_count} images filtered out")
        
        return text_chunks, images_metadata

    def _chunk_text_with_positions(self, text: str, size: int, overlap: int) -> List[Tuple[str, int, int]]:
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + size, text_len)
            chunk = text[start:end]
            if chunk.strip():
                chunks.append((chunk, start, end))
            start += size - overlap
            
        return chunks

    def _extract_and_validate_image(self, doc: fitz.Document, img: tuple, doc_id: str,
                                     page_number: int, img_index: int, page_text: str) -> Tuple[Optional[ImageMetadata], bool]:
        """
        Extract and validate an image from the PDF.
        Returns (ImageMetadata or None, was_filtered: bool)
        """
        try:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            pil_image = Image.open(BytesIO(image_bytes))
            width, height = pil_image.size
            
            # Apply dimension/aspect ratio filtering
            if not check_image_dimensions(width, height, self.image_filter_config):
                logger.debug(f"Filtered image on page {page_number}: {width}x{height} "
                           f"(aspect: {width/height:.2f})")
                return None, True
            
            safe_doc_id = os.path.splitext(doc_id)[0].replace(" ", "_")
            image_filename = f"{safe_doc_id}_p{page_number}_{img_index}.{image_ext}"
            image_path = IMAGES_DIR / image_filename
            
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            if not image_path.exists():
                return None, False
            
            absolute_image_path = str(image_path.resolve())
            nearby_text = page_text[:300].strip() if page_text else ""
            
            return ImageMetadata(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                page=page_number,
                filepath=absolute_image_path,
                context=f"Image {img_index + 1} on page {page_number} of {doc_id}",
                width=width,
                height=height,
                img_index=img_index,
                nearby_text=nearby_text
            ), False
            
        except Exception as e:
            logger.error(f"Failed to extract image on page {page_number}: {e}")
            return None, False

    def embed_text(self, chunks: List[Dict], progress_callback: Callable[[int, int, str], None] = None) -> np.ndarray:
        import ollama
        
        texts = [c["text"] for c in chunks]
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if progress_callback:
                progress_callback(i + 1, total, f"Embedding text chunk {i + 1}/{total}")
            response = ollama.embeddings(model=TEXT_EMBED_MODEL, prompt=text)
            embeddings.append(response["embedding"])
        
        return np.array(embeddings).astype('float32')

    def embed_images(self, images_metadata: List[Dict], progress_callback: Callable[[int, int, str], None] = None) -> Tuple[np.ndarray, List[Dict]]:
        images = []
        valid_metadata = []
        total = len(images_metadata)
        
        for i, meta in enumerate(images_metadata):
            if progress_callback:
                progress_callback(i + 1, total, f"Processing image {i + 1}/{total}")
            
            filepath = meta["filepath"]
            if not validate_file_exists(filepath):
                continue
                
            try:
                img = Image.open(filepath).convert("RGB")
                images.append(img)
                valid_metadata.append(meta)
            except Exception as e:
                logger.error(f"Failed to load image {filepath}: {e}")
        
        if not images:
            return np.array([]).astype('float32'), []

        inputs = self.clip_processor(images=images, return_tensors="pt")
        image_features = self.clip_model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        return image_features.detach().numpy().astype('float32'), valid_metadata


class VectorStore:
    def __init__(self):
        self.text_index: Optional[faiss.Index] = None
        self.image_index: Optional[faiss.Index] = None
        self.text_metadata: List[Dict] = []
        self.image_metadata: List[Dict] = []

    def build_index(self, text_chunks: List[Dict], images_metadata: List[Dict],
                    text_embeddings: np.ndarray, image_embeddings: np.ndarray):
        if len(text_chunks) > 0 and len(text_embeddings) > 0:
            d_text = text_embeddings.shape[1]
            self.text_index = faiss.IndexFlatL2(d_text)
            self.text_index.add(text_embeddings)
            self.text_metadata = text_chunks

        if len(images_metadata) > 0 and len(image_embeddings) > 0:
            d_image = image_embeddings.shape[1]
            self.image_index = faiss.IndexFlatL2(d_image)
            self.image_index.add(image_embeddings)
            self.image_metadata = images_metadata

    def save(self):
        if self.text_index is not None:
            faiss.write_index(self.text_index, str(INDEX_DIR / "text.index"))

        if self.image_index is not None:
            faiss.write_index(self.image_index, str(INDEX_DIR / "image.index"))

        with open(INDEX_DIR / "metadata.pkl", "wb") as f:
            pickle.dump({"text": self.text_metadata, "image": self.image_metadata}, f)

    def load(self) -> bool:
        try:
            text_index_path = INDEX_DIR / "text.index"
            if not text_index_path.exists():
                return False
                
            self.text_index = faiss.read_index(str(text_index_path))
            
            image_index_path = INDEX_DIR / "image.index"
            if image_index_path.exists():
                self.image_index = faiss.read_index(str(image_index_path))
            
            metadata_path = INDEX_DIR / "metadata.pkl"
            with open(metadata_path, "rb") as f:
                meta = pickle.load(f)
                self.text_metadata = meta.get("text", [])
                self.image_metadata = meta.get("image", [])
            
            return True
            
        except Exception as e:
            logger.warning(f"Could not load index: {e}")
            return False

    def get_stats(self) -> Dict:
        return {
            "text_vectors": self.text_index.ntotal if self.text_index else 0,
            "image_vectors": self.image_index.ntotal if self.image_index else 0,
            "text_metadata_count": len(self.text_metadata),
            "image_metadata_count": len(self.image_metadata),
        }