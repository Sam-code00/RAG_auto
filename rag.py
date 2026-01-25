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
        """Generate a description of user's image for search augmentation."""
        if not validate_file_exists(image_path):
            return ""
            
        try:
            img_b64 = self._image_to_base64(image_path)

            messages = [
                {
                    "role": "user",
                    "content": """Look at this image from a vehicle. Identify:
1. What part of the vehicle is shown (engine bay, dashboard, interior, exterior, etc.)
2. Any specific components visible (battery, fuse box, controls, warning lights, etc.)
3. Any text, labels, or warning symbols visible

Respond in 2-3 concise sentences describing what you see.""",
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
            return {"text": [], "images": [], "aligned_images": [], "user_image_description": ""}

        # Get description of user's image for better retrieval
        user_image_description = ""
        augmented_query = query
        if user_image_path:
            user_image_description = self.describe_user_image(user_image_path)
            if user_image_description:
                augmented_query = f"{query} {user_image_description}"
                logger.info(f"Augmented query with image description: {user_image_description[:100]}...")

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
            "image_scores": image_scores,
            "user_image_description": user_image_description
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
                
                # Apply stricter distance threshold
                if dist <= self.config.image_distance_threshold:
                    results.append(meta)
                    distances.append(float(dist))
                    
        return results, distances

    def _align_images_to_text(self, text_results: List[Dict], image_results: List[Dict]) -> List[Dict]:
        """Find images on the same pages as retrieved text chunks."""
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
        """Merge semantic search results with page-aligned images, scoring appropriately."""
        scored_images = []
        seen_ids = set()
        
        # Score semantic matches
        for img, dist in zip(semantic_images, semantic_distances):
            score = normalize_l2_distance_to_score(dist, 2.0)
            
            # Only include if above minimum relevance threshold
            if score >= self.config.min_image_relevance:
                img_copy = img.copy()
                img_copy["_retrieval_score"] = score
                img_copy["_retrieval_type"] = "semantic"
                scored_images.append((score, img_copy))
                seen_ids.add(img.get("id"))
        
        # Add page-aligned images with lower score
        for img in aligned_images:
            if img.get("id") not in seen_ids:
                score = self.config.page_alignment_bonus
                img_copy = img.copy()
                img_copy["_retrieval_score"] = score
                img_copy["_retrieval_type"] = "page_aligned"
                scored_images.append((score, img_copy))
                seen_ids.add(img.get("id"))
        
        # Sort by score and limit results
        scored_images.sort(key=lambda x: x[0], reverse=True)
        return [img for _, img in scored_images[:TOP_K_IMAGE + 2]]

    def generate_answer(self, query: str, retrieval_results: Dict, 
                       user_image_path: Optional[str] = None) -> str:
        """
        Generate an answer using retrieved context.
        If user uploaded an image, use VLM to provide visual guidance.
        """
        text_results = retrieval_results.get("text", [])
        image_results = retrieval_results.get("images", [])
        user_image_description = retrieval_results.get("user_image_description", "")
        
        # Build context from retrieved text
        context_parts = []
        for item in text_results:
            page = item.get("page", "?")
            text = item.get("text", "")
            context_parts.append(f"[Page {page}]: {text}")
        
        context_str = "\n\n".join(context_parts)
        
        # If user uploaded an image, use VLM for visual guidance
        if user_image_path and validate_file_exists(user_image_path):
            return self._generate_visual_guidance(
                query, context_str, user_image_path, 
                user_image_description, image_results
            )
        
        # Otherwise use standard text-based generation
        return self._generate_text_answer(query, context_str, image_results)

    def _generate_text_answer(self, query: str, context_str: str, 
                              image_results: List[Dict]) -> str:
        """Generate answer using text-only LLM (no user image)."""
        image_context = ""
        if image_results:
            image_pages = [str(img.get("page", "?")) for img in image_results]
            image_context = f"\n\nNote: Relevant diagrams/images from pages {', '.join(image_pages)} are shown below."
        
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
                "content": f"CONTEXT:\n{context_str}{image_context}\n\nQUESTION: {query}"
            }
        ]

        try:
            response = ollama.chat(model=OLLAMA_MODEL_NAME, messages=messages)
            return response['message']['content']
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating answer: {e}"

    def _generate_visual_guidance(self, query: str, context_str: str,
                                  user_image_path: str, user_image_description: str,
                                  manual_images: List[Dict]) -> str:
        """
        Generate answer with visual guidance using VLM.
        The VLM sees the user's uploaded photo and provides specific guidance
        based on what's visible in their actual image.
        """
        try:
            user_img_b64 = self._image_to_base64(user_image_path)
            
            # Build the prompt with context and visual instructions
            system_prompt = """You are a vehicle maintenance assistant with vision capabilities. 
The user has uploaded a photo of their vehicle and is asking for help.

Your job is to:
1. Look at their photo and identify what you can see
2. Use the manual information provided to give them accurate instructions
3. Give SPECIFIC visual guidance referencing what's actually visible in their photo
   - For example: "I can see the battery in your photo - it's the component on the left side. The red terminal cap you need to remove is visible at the top."
   - Or: "Looking at your dashboard, I can see the warning light you mentioned - it's the orange icon in the center cluster."
4. If you can identify the exact component they're asking about in their photo, point it out
5. If you cannot see what they're asking about in their photo, tell them what you CAN see and suggest what they should look for

Be specific and helpful. Reference actual visual elements you can see in their photo."""

            # Add manual image context if available
            manual_image_info = ""
            if manual_images:
                pages = [str(img.get("page", "?")) for img in manual_images]
                manual_image_info = f"\n\nNote: The manual has relevant diagrams on pages {', '.join(pages)} that are shown to the user separately."

            user_prompt = f"""MANUAL INFORMATION:
{context_str}
{manual_image_info}

USER'S QUESTION: {query}

Look at the user's photo and provide helpful guidance. Be specific about what you can see in their image."""

            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [user_img_b64]
                }
            ]

            response = ollama.chat(model=VLM_MODEL_NAME, messages=messages)
            return response['message']['content']

        except Exception as e:
            logger.error(f"VLM visual guidance failed: {e}")
            # Fall back to text-only response
            return self._generate_text_answer(query, context_str, manual_images)