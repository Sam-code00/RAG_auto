import os
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image

from ingest import PDFProcessor, VectorStore
from rag import RAGSystem
from utils import MANUALS_DIR, IMAGES_DIR, setup_logger

logger = setup_logger(__name__)

st.set_page_config(page_title="SMART Assistant", layout="wide", page_icon="🚗")

USER_UPLOADS_DIR = Path("data/user_uploads")
USER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def init_session_state():
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_img" not in st.session_state:
        st.session_state.pending_img = None


def load_rag_system():
    if st.session_state.rag_system is None:
        with st.spinner("Loading RAG System..."):
            try:
                rag = RAGSystem()
                rag.load_models()
                st.session_state.rag_system = rag
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
    processor = PDFProcessor()
    
    progress_bar = st.progress(0, text="Starting processing...")
    status_text = st.empty()
    
    def update_progress(current, total, message):
        progress = current / total
        progress_bar.progress(progress, text=message)
        status_text.text(message)
    
    status_text.text("Step 1/4: Extracting text and images from PDF...")
    progress_bar.progress(0.05, text="Extracting content...")
    
    text_chunks, images_meta = processor.process_pdf(
        str(file_path),
        progress_callback=lambda c, t, m: progress_bar.progress(0.05 + (c/t) * 0.20, text=m)
    )
    
    status_text.text("Step 2/4: Generating text embeddings...")
    progress_bar.progress(0.25, text="Generating text embeddings...")
    
    text_emb = processor.embed_text(
        text_chunks,
        progress_callback=lambda c, t, m: progress_bar.progress(0.25 + (c/t) * 0.35, text=m)
    )
    
    status_text.text("Step 3/4: Generating image embeddings...")
    progress_bar.progress(0.60, text="Generating image embeddings...")
    
    img_emb, valid_imgs = processor.embed_images(
        images_meta,
        progress_callback=lambda c, t, m: progress_bar.progress(0.60 + (c/t) * 0.25, text=m)
    )
    
    status_text.text("Step 4/4: Building and saving index...")
    progress_bar.progress(0.85, text="Building index...")
    
    store = VectorStore()
    store.build_index(text_chunks, valid_imgs, text_emb, img_emb)
    store.save()
    
    progress_bar.progress(1.0, text="Complete!")
    status_text.text("Processing complete!")
    
    stats = store.get_stats()
    
    progress_bar.empty()
    status_text.empty()
    
    st.session_state.rag_system.refresh_index()
    st.toast(f"Indexing Complete! {stats['text_vectors']} text chunks, {stats['image_vectors']} images", icon="✅")


def display_image_safely(filepath: str, caption: str = "", width: int = None) -> bool:
    try:
        if not filepath or not os.path.exists(filepath):
            return False
        image = Image.open(filepath)
        if width:
            st.image(image, caption=caption, width=width)
        else:
            st.image(image, caption=caption, use_container_width=True)
        return True
    except Exception:
        return False


def display_retrieved_images(images: list, title: str = "📷 Supporting Visuals from Manual"):
    if not images:
        return
    
    valid_images = [img for img in images if img.get("filepath") and os.path.exists(img.get("filepath", ""))]
    
    if not valid_images:
        return
    
    st.write(f"**{title}:**")
    
    num_cols = min(3, len(valid_images))
    cols = st.columns(num_cols)
    
    for i, img_meta in enumerate(valid_images[:6]):
        col_idx = i % num_cols
        with cols[col_idx]:
            filepath = img_meta["filepath"]
            page = img_meta.get("page", "?")
            display_image_safely(filepath, caption=f"Page {page}")


def render_sidebar():
    with st.sidebar:
        st.header("📄 Upload PDF Manual")
        uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_uploader")

        if uploaded_pdf:
            if st.button("🔄 Process PDF", type="primary"):
                st.toast(f"PDF uploaded: {uploaded_pdf.name}", icon="📄")
                file_path = save_uploaded_pdf(uploaded_pdf)
                process_document(file_path)

        st.divider()

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.pending_img = None
            st.rerun()

        st.divider()
        
        st.markdown("**📎 Attach Image (optional):**")
        user_img = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"], 
                                     key="chat_image_upload", label_visibility="collapsed")

        if user_img is not None:
            img_path = save_uploaded_image(user_img)
            st.session_state.pending_img = img_path
            st.toast(f"Image attached: {user_img.name}", icon="🖼️")

        if st.session_state.pending_img:
            st.image(st.session_state.pending_img, width=200)
            if st.button("Remove Image"):
                st.session_state.pending_img = None
                st.rerun()


def render_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content", ""))
            if msg.get("user_img"):
                display_image_safely(msg["user_img"], "Attached image", width=280)
            if msg["role"] == "assistant" and msg.get("images"):
                display_retrieved_images(msg["images"])


def handle_user_query(prompt: str):
    img_path = st.session_state.pending_img
    st.session_state.pending_img = None

    st.session_state.messages.append({"role": "user", "content": prompt, "user_img": img_path})

    with st.chat_message("user"):
        st.markdown(prompt)
        if img_path:
            display_image_safely(img_path, "Attached image", width=280)

    with st.chat_message("assistant"):
        with st.spinner("Searching manual..."):
            rag = st.session_state.rag_system
            results = rag.retrieve(prompt, user_image_path=img_path)

        with st.spinner("Generating answer..."):
            answer = rag.generate_answer(prompt, results)

        st.markdown(answer)

        images = results.get("images", [])
        if images:
            display_retrieved_images(images)

        st.session_state.messages.append({"role": "assistant", "content": answer, "images": images})


def main():
    load_css()
    init_session_state()
    
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("SMART Assistant")
        st.caption("Ask questions about your vehicle's manual.")
    
    with col2:
        col3, col4 = st.columns([1, 1])
        with col3:
            logo_path = Path("assets/nsf.png")
            if logo_path.exists():
                st.image(str(logo_path), width=80)
        with col4:
            logo_path2 = Path("assets/nsf1.png")
            if logo_path2.exists():
                st.image(str(logo_path2), width=80)      
    
    st.divider()
    
    load_rag_system()
    render_sidebar()
    render_chat_history()
    
    if prompt := st.chat_input("Ask a question about your manual..."):
        handle_user_query(prompt)


if __name__ == "__main__":
    main()