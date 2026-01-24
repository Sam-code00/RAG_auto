# ui_app.py
import os
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image

from ingest import PDFProcessor, VectorStore
from rag import RAGSystem
from utils import MANUALS_DIR, setup_logger

logger = setup_logger(__name__)

st.set_page_config(page_title="SMART Assistant", layout="wide", page_icon="🚗")

# Paths
USER_UPLOADS_DIR = Path("data/user_uploads")
USER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# CSS
def load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

load_css()

def load_rag_system():
    if "rag_system" not in st.session_state:
        with st.spinner("Loading RAG System..."):
            try:
                st.session_state.rag_system = RAGSystem()
                st.session_state.rag_system.load_models()
                st.toast("System Loaded", icon="✅")
            except Exception as e:
                st.error(f"Failed to load system: {e}")
                st.stop()

def save_uploaded_pdf(uploaded_file) -> Path:
    path = MANUALS_DIR / uploaded_file.name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def save_uploaded_image(uploaded_img) -> str:
    safe_name = uploaded_img.name.replace("/", "_").replace("\\", "_")
    path = USER_UPLOADS_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    with open(path, "wb") as f:
        f.write(uploaded_img.getbuffer())
    return str(path)

def process_document(file_path: Path):

    with st.spinner(f"{file_path.name} is currently being processed..."):
        processor = PDFProcessor()
        text_chunks, images_meta = processor.process_pdf(str(file_path))

        st.info(f"Extracted {len(text_chunks)} text chunks and {len(images_meta)} images.")

        text_emb = processor.embed_text(text_chunks)
        img_emb, valid_imgs = processor.embed_images(images_meta)

        store = VectorStore()
        store.build_index(text_chunks, valid_imgs, text_emb, img_emb)
        store.save()

        # Refresh RAG session
        st.session_state.rag_system.refresh_index()

# Header 
header_center, header_right = st.columns([6, 1.5])


with header_center:
    st.markdown(
        """
        <div style="margin-top: -20px;">
            <h1 style="margin-bottom: 0;">SMART Assistant</h1>
            <p style="margin-top: 2px; color: #6b7280;">
                Ask questions about your car's manual.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    c1, c2 = st.columns(2)

    with c1:
        st.image("assets/nsf.png", width=100)

    with c2:
        st.image("assets/nsf1.png", width=100)

st.divider()


def main():
    load_rag_system()

    # Chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_img" not in st.session_state:
        st.session_state.pending_img = None  # filepath to attach to next user message

    # Sidebar
    with st.sidebar:
        st.header("Upload PDF Manual")
        uploaded_pdf = st.file_uploader("", type=["pdf"])

        if uploaded_pdf and st.button("Process PDF"):
            file_path = save_uploaded_pdf(uploaded_pdf)
            process_document(file_path)
            st.toast("Indexing Complete!", icon="✅")

        st.divider()

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.pending_img = None
            st.success("Chat history cleared.")

        # Bottom pinned image attach
        st.markdown('<div class="sidebar-bottom">', unsafe_allow_html=True)

        st.markdown("**Attach an image (optional):**")
        user_img = st.file_uploader(
            "Upload a photo (png/jpg)",
            type=["png", "jpg", "jpeg"],
            key="chat_image_upload",
            label_visibility="collapsed",
        )

        if user_img is not None:
            try:
                st.session_state.pending_img = save_uploaded_image(user_img)
                st.toast("Image Attached", icon="📎")
            except Exception as e:
                st.error(e)

        # Optional preview in sidebar
        if st.session_state.pending_img:
            try:
                st.image(st.session_state.pending_img, use_container_width=True)
            except Exception:
                pass

        st.markdown("</div>", unsafe_allow_html=True)

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content", ""))

            # If user attached an image
            if msg.get("user_img"):
                try:
                    st.image(msg["user_img"], caption="Attached image", width=280)
                except Exception:
                    pass

            # If assistant returned supporting visuals
            imgs = msg.get("images") or []
            if imgs:
                st.write("**Supporting Visuals:**")
                cols = st.columns(min(3, len(imgs)))
                for i, img_meta in enumerate(imgs[:3]):
                    with cols[i]:
                        try:
                            image = Image.open(img_meta["filepath"])
                            st.image(image, caption=f"Page {img_meta.get('page', '?')}", use_container_width=True)
                        except Exception:
                            st.write("Image not found")

    # Chat input
    prompt = st.chat_input("Ask a question about your manual...")

    if prompt:
        # consume pending image for this message
        img_path = st.session_state.pending_img
        st.session_state.pending_img = None

        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt, "user_img": img_path})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                rag = st.session_state.rag_system

                # NOTE: Your rag.py retrieve() currently takes only (query).
                results = rag.retrieve(prompt)
                answer = rag.generate_answer(prompt, results)

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "images": (results.get("images") or [])[:3],
                    }
                )

if __name__ == "__main__":
    main()