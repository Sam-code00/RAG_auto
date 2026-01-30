import os
import logging
import numpy as np
import ollama
from ingest import PDFProcessor, VectorStore
from utils import (
    OLLAMA_MODEL_NAME, setup_logger, TOP_K_TEXT, TOP_K_IMAGE, TEXT_EMBED_MODEL,
    VLM_MODEL_NAME
)

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

            # Check VLM Model
            logger.info(f"Checking Ollama VLM model: {VLM_MODEL_NAME}...")
            try:
                pass 
            except:
                 pass

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
                'role': 'system',
                'content': "You are a helpful assistant answering questions about a manual. Use the provided context to answer the user's question. Do not HALLUCINATE. Do not go off what you think you know. If the answer is not contained in the context (MANUAL), say 'I don't know based on the provided manual.'"
            },
            {
                'role': 'user',
                'content': f"Context:\n{context_str}\n\nQuestion:\n{query}"
            }
        ]
        
        try:
            response = ollama.chat(model=OLLAMA_MODEL_NAME, messages=messages)
            return response['message']['content']
        except Exception as e:
            return f"Error generation answer with Ollama: {e}"

    def analyze_image_intent(self, image_path, user_query):
        """
        Uses VLM to analyze the image and use the provided user query to determine what to search.
        """
        prompt = (
            f"You are an expert automotive assistant that works ONLY with provided vehicle owner's manuals."
            f"Analyze the provided image and identify visible components ONLY if they are described or labeled in the manual."
            f"Do not make assumptions beyond what is given and do not use outside knowledge."
            f"The user asks: '{user_query}'. "
            f"Infer the user's intent (what they are trying to fix or check). "
            f"Provide a specific search query to find relevant instructions in the uploaded manual. "
            f"Return ONLY the search query."
        )
        
        try:
            # For Vision model, image path is passed in the 'images' list
            response = ollama.chat(
                model=VLM_MODEL_NAME,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_path]
                    }
                ]
            )
            return response['message']['content'].strip()
        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            return user_query  # Fallback to original query
