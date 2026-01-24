import numpy as np
import ollama
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import base64

from ingest import PDFProcessor, VectorStore
from utils import (
    OLLAMA_MODEL_NAME, VLM_MODEL_NAME, TEXT_EMBED_MODEL,
    TOP_K_TEXT, TOP_K_IMAGE,
    RETRIEVAL_CONFIG, RetrievalConfig,
    setup_logger, validate_file_exists, normalize_l2_distance_to_score
)

logger = setup_logger(__name__)


class RAGSystem:
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RETRIEVAL_CONFIG
        self.processor = PDFProcessor()
        self.vector_store = VectorStore()

    def load_models(self):
        try:
            models = ollama.list()
            model_names = []
            
            if 'models' in models:
                for m in models['models']:
                    name = m.get('name') if isinstance(m, dict) else getattr(m, 'model', str(m))
                    model_names.append(name)
                    
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}")

        self.refresh_index()

    def refresh_index(self):
        loaded = self.vector_store.load()
        if loaded:
            stats = self.vector_store.get_stats()
            logger.info(f"Index loaded: {stats}")

    def _image_to_base64(self, image_path: str) -> str:
        data = Path(image_path).read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def describe_user_image(self, image_path: str) -> str:
        if not validate_file_exists(image_path):
            return ""
            
        try:
            img_b64 = self._image_to_base64(image_path)

            messages = [
                {
                    "role": "system",
                    "content": "You are a vision assistant for vehicle maintenance. Describe what you see in the image in 1-3 sentences."
                },
                {
                    "role": "user",
                    "content": "Describe this image for searching a car manual.",
                    "images": [img_b64],
                },
            ]

            resp = ollama.chat(model=VLM_MODEL_NAME, messages=messages)
            return resp["message"]["content"].strip()

        except Exception as e:
            logger.warning(f"VLM image description failed: {e}")
            return ""

    def _embed_text_query(self, query: str) -> np.ndarray:
        response = ollama.embeddings(model=TEXT_EMBED_MODEL, prompt=query)
        return np.array(response["embedding"]).astype('float32').reshape(1, -1)

    def _embed_image_query(self, query: str) -> np.ndarray:
        inputs = self.processor.clip_processor(text=[query], return_tensors="pt", padding=True)
        text_features = self.processor.clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.detach().numpy().astype('float32')

    def retrieve(self, query: str, user_image_path: Optional[str] = None) -> Dict:
        if not self.vector_store.text_index:
            return {"text": [], "images": [], "aligned_images": []}

        augmented_query = query
        if user_image_path:
            image_desc = self.describe_user_image(user_image_path)
            if image_desc:
                augmented_query = f"{query} {image_desc}"

        text_query_embedding = self._embed_text_query(augmented_query)
        image_query_embedding = self._embed_image_query(query)

        text_results, text_distances = self._search_text(text_query_embedding, TOP_K_TEXT)
        image_results, image_distances = self._search_images(image_query_embedding, TOP_K_IMAGE)

        aligned_images = []
        if self.config.enable_page_alignment:
            aligned_images = self._align_images_to_text(text_results, image_results)

        final_images = self._merge_image_results(image_results, image_distances, aligned_images)

        text_scores = [normalize_l2_distance_to_score(d, 200.0) for d in text_distances]
        image_scores = [normalize_l2_distance_to_score(d, 2.0) for d in image_distances]

        return {
            "text": text_results,
            "images": final_images,
            "aligned_images": aligned_images,
            "text_scores": text_scores,
            "image_scores": image_scores
        }

    def _search_text(self, query_embedding: np.ndarray, top_k: int) -> Tuple[List[Dict], List[float]]:
        D, I = self.vector_store.text_index.search(query_embedding, top_k)
        
        results = []
        distances = []
        
        for idx, dist in zip(I[0], D[0]):
            if idx != -1 and idx < len(self.vector_store.text_metadata):
                if dist <= self.config.text_distance_threshold:
                    results.append(self.vector_store.text_metadata[idx])
                    distances.append(float(dist))
                    
        return results, distances

    def _search_images(self, query_embedding: np.ndarray, top_k: int) -> Tuple[List[Dict], List[float]]:
        if self.vector_store.image_index is None or self.vector_store.image_index.ntotal == 0:
            return [], []

        D, I = self.vector_store.image_index.search(query_embedding, top_k)
        
        results = []
        distances = []
        
        for idx, dist in zip(I[0], D[0]):
            if idx != -1 and idx < len(self.vector_store.image_metadata):
                meta = self.vector_store.image_metadata[idx]
                
                if not validate_file_exists(meta.get("filepath", "")):
                    continue
                
                if dist <= self.config.image_distance_threshold:
                    results.append(meta)
                    distances.append(float(dist))
                    
        return results, distances

    def _align_images_to_text(self, text_results: List[Dict], image_results: List[Dict]) -> List[Dict]:
        relevant_pages = set()
        for text in text_results:
            page = text.get("page")
            doc_id = text.get("doc_id")
            if page and doc_id:
                relevant_pages.add((doc_id, page))
        
        if not relevant_pages:
            return []
        
        retrieved_image_ids = {img.get("id") for img in image_results}
        aligned = []
        
        for img_meta in self.vector_store.image_metadata:
            img_page = (img_meta.get("doc_id"), img_meta.get("page"))
            img_id = img_meta.get("id")
            
            if img_page in relevant_pages and img_id not in retrieved_image_ids:
                if validate_file_exists(img_meta.get("filepath", "")):
                    aligned.append(img_meta)
        
        return aligned

    def _merge_image_results(self, semantic_images: List[Dict], semantic_distances: List[float],
                             aligned_images: List[Dict]) -> List[Dict]:
        scored_images = []
        seen_ids = set()
        
        for img, dist in zip(semantic_images, semantic_distances):
            score = normalize_l2_distance_to_score(dist, 2.0)
            img_copy = img.copy()
            img_copy["_retrieval_score"] = score
            img_copy["_retrieval_type"] = "semantic"
            scored_images.append((score, img_copy))
            seen_ids.add(img.get("id"))
        
        for img in aligned_images:
            if img.get("id") not in seen_ids:
                score = self.config.page_alignment_bonus
                img_copy = img.copy()
                img_copy["_retrieval_score"] = score
                img_copy["_retrieval_type"] = "page_aligned"
                scored_images.append((score, img_copy))
                seen_ids.add(img.get("id"))
        
        scored_images.sort(key=lambda x: x[0], reverse=True)
        return [img for _, img in scored_images[:TOP_K_IMAGE + 2]]

    def generate_answer(self, query: str, retrieval_results: Dict) -> str:
        text_results = retrieval_results.get("text", [])
        image_results = retrieval_results.get("images", [])
        
        context_parts = []
        for item in text_results:
            page = item.get("page", "?")
            text = item.get("text", "")
            context_parts.append(f"[Page {page}]: {text}")
        
        context_str = "\n\n".join(context_parts)
        
        image_context = ""
        if image_results:
            image_pages = [str(img.get("page", "?")) for img in image_results]
            image_context = f"\n\nNote: Relevant images from pages {', '.join(image_pages)} are available."
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a technical assistant answering questions using a vehicle manual. "
                    "Use ONLY the provided context to answer. "
                    "If the context doesn't contain the answer, say: 'I don't have enough information in the manual to answer this.' "
                    "Do NOT include page numbers in your answer - just provide the information directly. "
                    "When describing procedures, use clear numbered steps. "
                    "Be concise and accurate."
                )
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context_str}\n{image_context}\n\nQUESTION: {query}"
            }
        ]

        try:
            response = ollama.chat(model=OLLAMA_MODEL_NAME, messages=messages)
            return response['message']['content']
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating answer: {e}"