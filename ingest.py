import fitz  # PyMuPDF
import os
import uuid
import pickle
import numpy as np
from PIL import Image
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
from utils import (
    IMAGES_DIR, INDEX_DIR, TEXT_EMBED_MODEL, IMAGE_EMBED_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, setup_logger
)



logger = setup_logger(__name__)

class PDFProcessor:
    def __init__(self):
        # self.text_model = SentenceTransformer(TEXT_EMBED_MODEL)
        
        try:
            self.clip_model = CLIPModel.from_pretrained(IMAGE_EMBED_MODEL, local_files_only=True)
            self.clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBED_MODEL, local_files_only=True)
        except Exception as e:
            # handle exception if model is not found locally
            logger.error(f"Failed to load CLIP model offline: {e}. Please run 'python download_models.py' once with internet.")
            raise e

    def process_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        doc_id = os.path.basename(pdf_path)
        
        text_chunks = []
        images_metadata = []

        logger.info(f"Processing {doc_id} with {len(doc)} pages...")

        for page_num, page in enumerate(doc):
            # Extraction of text and chunking
            text = page.get_text()
            
            chunks = self._chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for chunk in chunks:
                text_chunks.append({
                    "id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "page": page_num + 1,
                    "text": chunk
                })

           # Image Extraction
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                image_filename = f"{os.path.splitext(doc_id)[0]}_p{page_num+1}_{img_index}.{image_ext}"
                image_path = IMAGES_DIR / image_filename

                # Save Image
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                images_metadata.append({
                    "id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "page": page_num + 1,
                    "filepath": str(image_path),
                    "context": f"Image on page {page_num + 1} of {doc_id}" 
                })

        return text_chunks, images_metadata

    def _chunk_text(self, text, size, overlap):
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunk = text[start:end]
            chunks.append(chunk)
            start += size - overlap
        return chunks

    def embed_text(self, chunks):
        texts = [c["text"] for c in chunks]
        # Ollama for embeddings
        embeddings = []
        import ollama
        for text in texts:
            response = ollama.embeddings(model=TEXT_EMBED_MODEL, prompt=text)
            embeddings.append(response["embedding"])
        
        return np.array(embeddings).astype('float32')

    def embed_images(self, images_metadata):
        images = []
        valid_indices = []
        for idx, meta in enumerate(images_metadata):
            try:
                img = Image.open(meta["filepath"])
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                logger.error(f"Failed to load image {meta['filepath']}: {e}")
        
        if not images:
            return np.array([]).astype('float32'), []

        inputs = self.clip_processor(images=images, return_tensors="pt")
        image_features = self.clip_model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True) 
        
       
        valid_metadata = [images_metadata[i] for i in valid_indices]
        
        return image_features.detach().numpy().astype('float32'), valid_metadata

class VectorStore:
    def __init__(self):
        self.text_index = None
        self.image_index = None
        self.text_metadata = []
        self.image_metadata = []

    def build_index(self, text_chunks, images_metadata, text_embeddings, image_embeddings):
        # Text Index
        if len(text_chunks) > 0:
            d_text = text_embeddings.shape[1]
            self.text_index = faiss.IndexFlatL2(d_text)
            self.text_index.add(text_embeddings)
            self.text_metadata = text_chunks

        # Image Index
        if len(images_metadata) > 0 and len(image_embeddings) > 0:
            d_image = image_embeddings.shape[1]
            self.image_index = faiss.IndexFlatL2(d_image)
            self.image_index.add(image_embeddings)
            self.image_metadata = images_metadata

    def save(self):
        # Save 
        if self.text_index:
            faiss.write_index(self.text_index, str(INDEX_DIR / "text.index"))
        if self.image_index:
            faiss.write_index(self.image_index, str(INDEX_DIR / "image.index"))
        
        # Save Metadata
        with open(INDEX_DIR / "metadata.pkl", "wb") as f:
            pickle.dump({
                "text": self.text_metadata,
                "image": self.image_metadata
            }, f)
        logger.info("Index saved successfully.")

    def load(self):
        try:
            self.text_index = faiss.read_index(str(INDEX_DIR / "text.index"))
            if (INDEX_DIR / "image.index").exists():
                self.image_index = faiss.read_index(str(INDEX_DIR / "image.index"))
            
            with open(INDEX_DIR / "metadata.pkl", "rb") as f:
                meta = pickle.load(f)
                self.text_metadata = meta["text"]
                self.image_metadata = meta["image"]
            logger.info("Index loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Could not load index: {e}")
            return False