import os
import base64
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from ingest import PDFProcessor, VectorStore
from rag import RAGSystem
from utils import MANUALS_DIR, setup_logger
import speech_to_text as stt

logger = setup_logger(__name__)

st.set_page_config(page_title="SMART Assistant", layout="wide", page_icon="🚗")


def load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# Header
col1, col2 = st.columns([5, 1])
with col1:
    st.title("SMART Assistant")
    st.caption("Ask questions about your vehicle's manual. Upload a photo for visual guidance!")
with col2:
    if Path("assets/SMART.png").exists():
        st.image("assets/SMART.png", width=300)


def load_rag_system():
    if "rag_system" not in st.session_state:
        try:
            st.session_state.rag_system = RAGSystem()
            st.session_state.rag_system.load_models()
        except Exception as e:
            st.error(f"Failed to load system: {e}")
            st.stop()


def load_whisper():
    if "whisper_loaded" not in st.session_state:
        try:
            stt.load_model()
            st.session_state.whisper_loaded = True
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            st.session_state.whisper_loaded = False
            logger.warning(f"Whisper not loaded: {e}")


def save_uploaded_file(uploaded_file):
    save_path = MANUALS_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def process_document(file_path):
    processor = PDFProcessor()
    text_chunks, images_meta = processor.process_pdf(str(file_path))
    st.toast(f"Extracted {len(text_chunks)} text chunks and {len(images_meta)} images.", icon="📄")

    text_emb = processor.embed_text(text_chunks)
    img_emb, valid_imgs = processor.embed_images(images_meta)

    store = VectorStore()
    store.build_index(text_chunks, valid_imgs, text_emb, img_emb)
    store.save()

    st.session_state.rag_system.refresh_index()
    st.toast("Indexing Complete.", icon="✅")


def display_chat_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # ✅ show user's attached image directly under their question
            if msg.get("uploaded_image_path"):
                try:
                    st.image(msg["uploaded_image_path"], caption="Attached Image", width=260)
                except Exception:
                    pass

            if msg.get("vlm_analysis"):
                st.info(f"**VLM Analysis:** {msg['vlm_analysis']}")

            if msg.get("images"):
                st.write("**Supporting Visuals:**")
                imgs = msg["images"][:3]
                num_imgs = len(imgs)
                if num_imgs:
                    img_cols = st.columns(num_imgs, gap="medium")
                    for i, img_meta in enumerate(imgs):
                        with img_cols[i]:
                            try:
                                st.image(
                                    Image.open(img_meta["filepath"]),
                                    caption=f"Page {img_meta['page']}",
                                    width=300,
                                )
                            except Exception:
                                pass


def _save_bytes_to_temp_file(raw_bytes: bytes, filename: str) -> str:
    suffix = "." + filename.split(".")[-1] if "." in filename else ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        return tmp.name


def queue_query(prompt_text: str, image_file=None, image_bytes: bytes = None, image_name: str = "upload.png"):
    """
    Queue a user query for processing. Supports:
      - image_file: Streamlit UploadedFile (from st.file_uploader)
      - image_bytes: raw bytes (from multimodal component base64 decode)
    Also ensures the image shows under the user's chat bubble via uploaded_image_path.
    """
    user_msg = {"role": "user", "content": prompt_text}
    img_path = None

    if image_file is not None:
        try:
            img_path = _save_bytes_to_temp_file(image_file.getbuffer(), image_file.name)
        except Exception:
            img_path = None

    elif image_bytes is not None:
        try:
            img_path = _save_bytes_to_temp_file(image_bytes, image_name)
        except Exception:
            img_path = None

    if img_path:
        user_msg["uploaded_image_path"] = img_path

    st.session_state.messages.append(user_msg)
    st.session_state.pending_query = {"prompt": prompt_text, "img_path": img_path}


def run_pending_query():
    pending = st.session_state.pop("pending_query", None)
    if not pending:
        return

    prompt_text = pending["prompt"]
    img_path = pending["img_path"]
    rag = st.session_state.rag_system

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            search_query = prompt_text
            vlm_description = None

            if img_path:
                vlm_description = rag.analyze_image_intent(img_path, prompt_text)
                search_query = vlm_description

            retrieval = rag.retrieve(search_query)

            gen_prompt = (
                f"User uploaded an image described as: '{vlm_description}'. User Question: {prompt_text}"
                if vlm_description
                else prompt_text
            )
            answer = rag.generate_answer(gen_prompt, retrieval)

        st.markdown(answer)

        if vlm_description:
            st.info(f"**VLM Analysis:** {vlm_description}")

        if retrieval.get("images"):
            st.write("**Supporting Visuals:**")
            imgs = retrieval["images"][:3]
            img_cols = st.columns(len(imgs), gap="medium")
            for i, img_meta in enumerate(imgs):
                with img_cols[i]:
                    try:
                        st.image(
                            Image.open(img_meta["filepath"]),
                            caption=f"Page {img_meta['page']}",
                            width=300,
                        )
                    except Exception:
                        pass

        with st.expander("🔎 Retrieved Manual Context (Top matches)"):
            if retrieval.get("text"):
                for i, chunk in enumerate(retrieval["text"], start=1):
                    page = chunk.get("page", "?")
                    doc = chunk.get("doc_id", "")
                    txt = chunk.get("text", "")
                    st.markdown(f"**{i}. Page {page}** — `{doc}`")
                    st.write(txt[:600] + ("..." if len(txt) > 600 else ""))
                    st.markdown("---")
            else:
                st.write("No text chunks retrieved.")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "images": retrieval["images"][:3] if retrieval.get("images") else [],
            "vlm_analysis": vlm_description,
        }
    )


def main():
    load_rag_system()
    load_whisper()

    # Status toast
    if st.session_state.rag_system.text_index:
        st.toast(f"Index: {st.session_state.rag_system.text_index.ntotal} chunks", icon="📚")
    else:
        st.toast("No index found. Upload a PDF manual to get started.", icon="📂")

    # Sidebar
    with st.sidebar:
        st.header("Document Ingestion")
        pdf_file = st.file_uploader("Upload PDF Manual", type=["pdf"])
        if pdf_file and st.button("Process PDF"):
            process_document(save_uploaded_file(pdf_file))

        st.markdown("---")
        st.subheader("Indexed Manuals")

        store = VectorStore()
        if store.load():
            chunks = store.text_metadata or []
            indexed_docs = sorted({c.get("doc_id", "Unknown") for c in chunks if isinstance(c, dict)})

            if indexed_docs:
                cols = st.columns(2) 
                for i, doc in enumerate(indexed_docs):
                    with cols[i % 2]:
                        st.markdown(
                            f"""
                            <div style="
                                padding: 16px;
                                border-radius: 12px;
                                background-color: #1e1e1e;
                                border: 1px solid #333;
                                margin-bottom: 12px;
                            ">
                                <div style="font-weight:600; font-size:16px;">
                                    📄 {doc}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            else:
                st.caption("Index loaded, but no doc metadata found.")
        else:
            st.caption("No manuals indexed yet.")



        # Clear Chat Button
        st.markdown("---")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.query_image = None


    # Init state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat
    display_chat_messages()

    # Process any queued query
    run_pending_query()

    prompt = st.chat_input(
        placeholder="Ask a question about the manual...",
        accept_file="multiple",                 
        file_type=["jpg", "jpeg", "png", "webp"],
        accept_audio=True,
        audio_sample_rate=16000,
        key="chat_input",
    )

    if prompt:
        # Streamlit returns dict-like object when accept_file/accept_audio is enabled
        text = (prompt.text or "").strip()

        # If audio exists and text is empty, transcribe audio -> text
        if (not text) and getattr(prompt, "audio", None):
            try:
                audio_bytes = prompt.audio.getvalue()
                with st.spinner("Transcribing audio..."):
                    transcript = stt.transcribe_audio(audio_bytes)
                text = (transcript or "").strip()
            except Exception:
                text = ""

        # Pick first image file (if any)
        img_bytes = None
        img_name = "upload.png"
        files = getattr(prompt, "files", []) or []
        for f in files:
            # best-effort image check by mimetype or extension
            ftype = (getattr(f, "type", "") or "").lower()
            fname = (getattr(f, "name", "") or "").lower()
            is_img = ftype.startswith("image/") or fname.endswith((".jpg", ".jpeg", ".png", ".webp"))
            if is_img:
                try:
                    img_bytes = f.getvalue()
                    img_name = f.name or img_name
                except Exception:
                    img_bytes = None
                break

        # If no inline files image, fall back to sidebar image (if any)
        fallback_sidebar_img = st.session_state.get("query_image")

        # Only queue if we have text (from typing or audio transcription)
        if text:
            if img_bytes is not None:
                queue_query(text, image_bytes=img_bytes, image_name=img_name)
            else:
                queue_query(text, image_file=fallback_sidebar_img)

            st.session_state.query_image = None
            st.rerun()
        else:
            st.toast("Please type a question or record audio.", icon="⚠️")
            st.session_state.query_image = None
            st.rerun()


if __name__ == "__main__":
    main()