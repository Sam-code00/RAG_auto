import os
import logging
import numpy as np
import ollama
from ingest import PDFProcessor, VectorStore
from utils import (
    OLLAMA_MODEL_NAME, setup_logger, TOP_K_TEXT, TOP_K_IMAGE, TEXT_EMBED_MODEL, VLM_MODEL_NAME
)
import base64
from pathlib import Path


logger = setup_logger(__name__)

class RAGSystem:
    def __init__(self):
        self.processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.text_index = None 

    def load_models(self):
        logger.info(f"Checking Ollama model: {OLLAMA_MODEL_NAME}...")
        try:
            models = ollama.list()
            # Handle both object and dict return types

            model_names = []
            if 'models' in models:
                for m in models['models']:
                    name = m.get('name') if isinstance(m, dict) else m.model
                    model_names.append(name)
            
            # Check for match
            found = any(OLLAMA_MODEL_NAME in name for name in model_names)
            if not found:
                 logger.warning(f"Model '{OLLAMA_MODEL_NAME}' not found in Ollama list: {model_names}. Attempting to pull or run anyway...")
            else:
                logger.info(f"Model '{OLLAMA_MODEL_NAME}' found.")

        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}. Ensure 'ollama serve' is running.")

        # Load Index
        self.refresh_index()

    def refresh_index(self):
        logger.info("Loading Index...")
        loaded = self.vector_store.load()
        if loaded:
            self.text_index = self.vector_store.text_index
            logger.info(f"Index loaded. Text chunks: {self.text_index.ntotal if self.text_index else 0}")
        else:
            logger.warning("No index found. Please ingest documents.")
    def _image_to_base64(self, image_path: str) -> str:
        data = Path(image_path).read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def describe_user_image(self, image_path: str) -> str:
        try:
            img_b64 = self._image_to_base64(image_path)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a vision assistant for vehicle maintenance. "
                        "Describe what you see in the image in 1-3 sentences. "
                        "If you are unsure, say what is unclear. "
                        "Do not guess specific part numbers."
                    )
                },
                {
                    "role": "user",
                    "content": "Describe the image for the purpose of searching a car manual.",
                    "images": [img_b64],
                },
            ]

            resp = ollama.chat(model=VLM_MODEL_NAME, messages=messages)
            return resp["message"]["content"].strip()

        except Exception as e:
            logger.warning(f"VLM image description failed: {e}")
            return ""

    def retrieve(self, query):
        if not self.vector_store.text_index:
            logger.warning("Attempted retrieval without index.")
            return {"text": [], "images": []}

        # query_embedding = self.processor.embed_text([{"text": query}])[0].reshape(1, -1)
        response = ollama.embeddings(model=TEXT_EMBED_MODEL, prompt=query)
        query_embedding = np.array(response["embedding"]).astype('float32').reshape(1, -1)
        
        # Image
        inputs = self.processor.clip_processor(text=[query], return_tensors="pt", padding=True)
        text_features = self.processor.clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        image_query_embedding = text_features.detach().numpy().astype('float32')

        # Search text index
        D, I = self.vector_store.text_index.search(query_embedding, TOP_K_TEXT)
        retrieved_text = []
        for idx in I[0]:
            if idx != -1 and idx < len(self.vector_store.text_metadata):
                retrieved_text.append(self.vector_store.text_metadata[idx])

        #Search Image Index
        retrieved_images = []
        if self.vector_store.image_index:
            D_img, I_img = self.vector_store.image_index.search(image_query_embedding, TOP_K_IMAGE)
            for idx in I_img[0]:
                if idx != -1 and idx < len(self.vector_store.image_metadata):
                    retrieved_images.append(self.vector_store.image_metadata[idx])

        return {
            "text": retrieved_text,
            "images": retrieved_images
        }

    def generate_answer(self, query, retrieval_results):
        context_texts = [item['text'] for item in retrieval_results['text']]
        context_str = "\n\n".join(context_texts)
        
        # Messages for chat
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a technical assistant answering questions strictly using a vehicle or equipment manual. "
                    "You MUST rely only on the provided context. "
                    "If the context does not clearly contain the answer, respond exactly with: "
                    "'I don’t know based on the provided manual.' "
                    "Do NOT guess, infer, or add outside knowledge. "
                    "When the context describes procedures or steps, present them in a clear, numbered format. "
                    "Be concise, accurate, and factual."
                )
            },
            {
                "role": "user",
                "content": (
                    f"MANUAL CONTEXT:\n"
                    f"{context_str}\n\n"
                    f"USER QUESTION:\n"
                    f"{query}\n\n"
                    f"ANSWER:"
                )
            }
        ]

        try:
            response = ollama.chat(model=OLLAMA_MODEL_NAME, messages=messages)
            return response['message']['content']
        except Exception as e:
            return f"Error generation answer with Ollama: {e}"
