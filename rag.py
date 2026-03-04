import os
import logging
import numpy as np
import ollama

from ingest import PDFProcessor, VectorStore
from utils import (
    OLLAMA_MODEL_NAME,
    setup_logger,
    TOP_K_TEXT,
    TOP_K_IMAGE,
    TEXT_EMBED_MODEL,
    VLM_MODEL_NAME,
    IMAGE_COSINE_MIN,
    IMAGE_GAP_MIN,
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
            model_names = []
            if "models" in models:
                for m in models["models"]:
                    name = m.get("name") if isinstance(m, dict) else m.model
                    model_names.append(name)

            found = any(OLLAMA_MODEL_NAME in name for name in model_names)
            if not found:
                logger.warning(f"Model '{OLLAMA_MODEL_NAME}' not found: {model_names}. Will try anyway...")
            else:
                logger.info(f"Model '{OLLAMA_MODEL_NAME}' found.")

            logger.info(f"Checking VLM model: {VLM_MODEL_NAME}...")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}. Ensure 'ollama serve' is running.")

        self.refresh_index()

    def refresh_index(self):
        logger.info("Loading Index...")
        loaded = self.vector_store.load()
        if loaded:
            self.text_index = self.vector_store.text_index
            logger.info(f"Index loaded. Text chunks: {self.text_index.ntotal if self.text_index else 0}")
        else:
            logger.warning("No index found. Please ingest documents.")

    def retrieve(self, query: str, doc_id: str = None):
        if not self.vector_store.text_index:
            logger.warning("Attempted retrieval without index.")
            return {"text": [], "images": []}

        # Text retrieval
        response = ollama.embeddings(model=TEXT_EMBED_MODEL, prompt=query)
        query_embedding = np.array(response["embedding"], dtype="float32").reshape(1, -1)

        K_PAGES = max(TOP_K_TEXT, 15)
        if doc_id:
            K_PAGES = min(K_PAGES * 3, self.vector_store.text_index.ntotal)
        D, I = self.vector_store.text_index.search(query_embedding, K_PAGES)

        all_text_hits = []
        for idx in I[0]:
            if idx != -1 and idx < len(self.vector_store.text_metadata):
                chunk = self.vector_store.text_metadata[idx]
                if doc_id and chunk.get("doc_id") != doc_id:
                    continue
                all_text_hits.append(chunk)

        retrieved_text = all_text_hits[:TOP_K_TEXT]

        # Detect which manuals the top text results came from
        relevant_doc_ids = set()
        for chunk in retrieved_text:
            did = chunk.get("doc_id")
            if did:
                relevant_doc_ids.add(did)

        relevant_pages = {}  # {doc_id: set of pages}
        for chunk in retrieved_text:
            page = chunk.get("page")
            did = chunk.get("doc_id", "")
            if page is None:
                continue
            page = int(page)
            if did not in relevant_pages:
                relevant_pages[did] = set()
            relevant_pages[did].add(page)
            for dp in (-1, 1):
                pn = page + dp
                if pn >= 1:
                    relevant_pages[did].add(pn)

        # Flatten for backward compat
        all_relevant_pages = set()
        for pages in relevant_pages.values():
            all_relevant_pages.update(pages)

        # Page-filter images, restrict to docs that had text hits
        page_filtered_images = []
        page_filtered_indices = []

        if getattr(self.vector_store, "image_metadata", None):
            for idx, meta in enumerate(self.vector_store.image_metadata):
                img_doc = meta.get("doc_id", "")
                if doc_id and img_doc != doc_id:
                    continue
                # When searching all, only pull images from manuals that had text hits
                if not doc_id and img_doc not in relevant_doc_ids:
                    continue
                p = meta.get("page")
                doc_pages = relevant_pages.get(img_doc, set())
                if p is not None and int(p) in doc_pages:

                    page_filtered_images.append(meta)
                    page_filtered_indices.append(idx)

        retrieved_images = []

        if page_filtered_images and self.vector_store.image_index:
            try:
                # SigLIP2 text embedding
                import torch
                inputs = self.processor.clip_processor(
                    text=[query], return_tensors="pt",
                    padding="max_length", max_length=64
                )
                with torch.no_grad():
                    text_features = self.processor.clip_model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                query_clip = text_features.detach().numpy().astype("float32")

                candidate_embeddings = []
                valid_map = []

                for local_i, faiss_idx in enumerate(page_filtered_indices):
                    if 0 <= faiss_idx < self.vector_store.image_index.ntotal:
                        vec = self.vector_store.image_index.reconstruct(faiss_idx)
                        candidate_embeddings.append(vec)
                        valid_map.append(local_i)
                    else:
                        logger.warning(f"FAISS idx {faiss_idx} out of bounds (ntotal={self.vector_store.image_index.ntotal})")

                if candidate_embeddings:
                    candidate_matrix = np.array(candidate_embeddings, dtype="float32")
                    sims = (query_clip @ candidate_matrix.T).flatten()

                    # Build page prior from text ranking
                    core_pages = set()
                    page_rank = {}
                    for r, chunk in enumerate(retrieved_text):
                        p = chunk.get("page")
                        if p is None:
                            continue

                        core_pages.add(p)
                        if p not in page_rank:
                            page_rank[p] = r

                        for dp in (-2, -1, 1, 2):
                            pn = p + dp
                            if pn >= 1 and pn not in page_rank:
                                page_rank[pn] = r + 0.5


                    def page_score(page_num: int) -> float:
                        r = page_rank.get(page_num, 999)
                        base = 1.0 / (r + 1.0)
                        # Penalize images outside core text pages
                        if page_num not in core_pages:
                            base *= 0.5
                        return base

                    alpha = 0.6
                    gap_min = IMAGE_GAP_MIN

                    fused = []
                    for row_i, sim in enumerate(sims):
                        local_i = valid_map[row_i]
                        meta = dict(page_filtered_images[local_i])

                        cos = float(sim)
                        p = int(meta.get("page", -1))
                        ps = page_score(p)

                        meta["score_cosine"] = cos
                        meta["score_page"] = ps
                        meta["score_fused"] = alpha * ps + (1.0 - alpha) * cos

                        fused.append(meta)

                    fused.sort(key=lambda x: x["score_fused"], reverse=True)
                    fused = [c for c in fused if c["score_cosine"] >= IMAGE_COSINE_MIN]

                    # Penalize images far from the top-scoring image
                    if fused:
                        anchor_page = int(fused[0].get("page", -1))
                        for c in fused[1:]:
                            img_page = int(c.get("page", -1))
                            distance = abs(img_page - anchor_page)
                            if distance > 3:
                                c["score_fused"] *= 0.4
                            elif distance > 1:
                                c["score_fused"] *= 0.7
                        fused.sort(key=lambda x: x["score_fused"], reverse=True)

                    prev = None
                    top_score = fused[0]["score_fused"] if fused else 0
                    for c in fused:
                        # Only keep images within 60% of top score
                        if c["score_fused"] < top_score * 0.6:
                            break
                        retrieved_images.append(c)

                    retrieved_images = retrieved_images[:TOP_K_IMAGE]

                else:
                    retrieved_images = []

            except Exception as e:
                logger.warning(f"CLIP re-ranking failed, using page-filtered order: {e}")
                retrieved_images = page_filtered_images[:TOP_K_IMAGE]
        else:
            retrieved_images = page_filtered_images[:TOP_K_IMAGE]


        if not retrieved_images and page_filtered_images:
            # If CLIP found nothing, just grab the images from the top text result's page
            top_page = retrieved_text[0].get("page")
            retrieved_images = [img for img in page_filtered_images if img.get("page") == top_page][:3]

        logger.info(
            f"Retrieved {len(retrieved_text)} text chunks from docs {sorted(relevant_doc_ids)}, "
            f"{len(retrieved_images)} images (from {len(page_filtered_images)} page-filtered, "
            f"index_total={self.vector_store.image_index.ntotal if self.vector_store.image_index else 0})"
        )

        return {
            "text": retrieved_text,
            "images": retrieved_images,
            "source_docs": sorted(relevant_doc_ids),
        }

    def generate_answer(self, query, retrieval_results):
        # Check if results span multiple manuals
        doc_ids = set()
        for item in retrieval_results['text']:
            did = item.get('doc_id', '')
            if did:
                doc_ids.add(did)

        if len(doc_ids) > 1:
            # Group context by manual
            grouped = {}
            for item in retrieval_results['text']:
                did = item.get('doc_id', 'Unknown')
                if did not in grouped:
                    grouped[did] = []
                grouped[did].append(item['text'])

            context_parts = []
            for did, texts in grouped.items():
                context_parts.append(f"=== From {did} ===\n" + "\n\n".join(texts))
            context_str = "\n\n".join(context_parts)
        else:
            context_texts = [item['text'] for item in retrieval_results['text']]
            context_str = "\n\n".join(context_texts)

        messages = [
    {
        'role': 'system',
        'content': (
            "You are a professional automotive service assistant helping a user "
            "perform maintenance or diagnostics using ONLY their official vehicle manual.\n\n"
            "CRITICAL RULES:\n"
            "- Answer STRICTLY and ONLY using the provided manual context below.\n"
            "- Do NOT use your own knowledge, training data, or general automotive knowledge.\n"
            "- Do NOT improvise, guess, or fill in missing details.\n"
            "- If the context does not contain a clear answer to the question, you MUST respond with EXACTLY:\n"
            "  'I couldn't find this information in the manual.'\n"
            "- Do NOT attempt a partial answer if the procedure is not explicitly described in the context.\n"
            "- If the context mentions the topic but lacks full steps, say what the manual states and note that full details were not found.\n"
            "- If the context contains information from MULTIPLE manuals, clearly label which manual each answer comes from.\n\n"
            "FORMATTING (only when you DO have an answer from the context):\n"
            "- Use clear numbered steps (1, 2, 3) for procedures.\n"
            "- Put each step on its own line.\n"
            "- Use sub-steps (a, b, c) only when there are alternatives within a step.\n"
            "- Add a blank line between major steps for readability.\n"
            "- Bold important warnings with **WARNING:**.\n"
            "- Keep each step concise but complete.\n\n"
            "Example format:\n"
            "1. First step here.\n\n"
            "2. Second step here.\n\n"
            "3. Third step with options:\n"
            "   a. Option one.\n"
            "   b. Option two.\n\n"
            "**WARNING:** Safety note here."
        )
    },
    {
        'role': 'user',
        'content': f"Manual Context:\n{context_str}\n\nUser Question:\n{query}\n\nRemember: Only answer using the manual context above. If the answer is not there, say you couldn't find it."
    }
]
        try:
            response = ollama.chat(model=OLLAMA_MODEL_NAME, messages=messages)
            return response['message']['content']
        except Exception as e:
            return f"Error generating answer with Ollama: {e}"

    def analyze_image_intent(self, img_path, user_query):
        # Debug code
        print(f"DEBUG: Starting VLM analysis for {img_path}")
        prompt = (
            "You are a certified automotive assistant that works ONLY with the user's uploaded vehicle manual. "
            "Analyze the uploaded image and identify visible vehicle components ONLY if they are clearly described "
            "or labeled in the manual. Do NOT use outside automotive knowledge. "
            "Do NOT guess unknown parts.\n\n"
            f"The user asks: '{user_query}'.\n\n"
            "Based on the image and the user's question, infer the user's maintenance or diagnostic intent. "
            "Generate a precise, specific search query that will retrieve the correct procedural instructions "
            "from the manual.\n\n"
            "Return ONLY the refined search query. "
            "Do not explain your reasoning. Do not answer the question. "
            "Do not provide steps. Only return the search query."
        )


        try:
            response = ollama.chat(
                model=VLM_MODEL_NAME,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_path]
                }]
            )


    
            raw_content = response['message']['content'].strip()
            clean_query = raw_content.split('\n')[0]
            fillers = ["here is the search query:", "refined search query:", "search query:"]
            for filler in fillers:
                if clean_query.lower().startswith(filler):
                    clean_query = clean_query.lower().replace(filler, "").strip()
            clean_query = clean_query.replace('"', '').replace("'", "")
            # print(f"DEBUG: Cleaned query: {clean_query}")
            return clean_query



            # return response['message']['content'].strip()
        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            return user_query