import streamlit as st
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from PIL import Image
from ingest import PDFProcessor, VectorStore
from rag import RAGSystem
from utils import MANUALS_DIR, IMAGES_DIR, setup_logger
import speech_to_text as stt
import tempfile
from pathlib import Path

logger = setup_logger(__name__)

st.set_page_config(page_title="Local Multimodal RAG", layout="wide", page_icon="")


def load_css():
    if Path("assets/style.css").exists():
        with open("assets/style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Header
col1, col2 = st.columns([5, 1])
with col1:
    st.title("SMART Assistant")
    st.caption("Ask questions about your vehicle's manual. Upload a photo for visual guidance!")
with col2:
    c1, c2 = st.columns(2)
    with c1:
        if Path("assets/SMART.png").exists():
            st.image("assets/SMART.png", width=80)
    with c2:
        if Path("assets/nsf1.png").exists():
            st.image("assets/nsf1.png", width=80)


def load_rag_system():
    if "rag_system" not in st.session_state:
        try:
            st.session_state.rag_system = RAGSystem()
            st.session_state.rag_system.load_models()
        except Exception as e:
            st.error(f"Failed to load system: {e}")
            st.stop()


def load_vosk():
    if "vosk_loaded" not in st.session_state:
        try:
            stt.load_model()
            st.session_state.vosk_loaded = True
        except Exception as e:
            st.session_state.vosk_loaded = False
            logger.warning(f"Vosk not loaded: {e}")


def save_uploaded_file(uploaded_file):
    save_path = MANUALS_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def process_document(file_path):
    processor = PDFProcessor()
    text_chunks, images_meta = processor.process_pdf(str(file_path))
    st.toast(f"Extracted {len(text_chunks)} text chunks and {len(images_meta)} images.")
    
    text_emb = processor.embed_text(text_chunks)
    img_emb, valid_imgs = processor.embed_images(images_meta)
    
    store = VectorStore()
    store.build_index(text_chunks, valid_imgs, text_emb, img_emb)
    store.save()
    
    st.session_state.rag_system.refresh_index()
    st.toast("Indexing Complete!")


def display_chat_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("uploaded_image_path"):
                st.image(msg["uploaded_image_path"], caption="Uploaded Image", width=250)
            if msg.get("vlm_analysis"):
                st.info(f"**VLM Analysis:** {msg['vlm_analysis']}")
            if msg.get("images"):
                st.write("**Supporting Visuals:**")
                cols = st.columns(min(3, len(msg["images"])))
                for i, img_meta in enumerate(msg["images"][:3]):
                    with cols[i]:
                        try:
                            st.image(Image.open(img_meta["filepath"]), caption=f"Page {img_meta['page']}", use_container_width=True)
                        except:
                            pass


def process_query(prompt, image_file=None):
    """Process a query and generate response"""
    user_msg = {"role": "user", "content": prompt}
    img_path = None
    
    if image_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image_file.name.split('.')[-1]}") as tmp:
            tmp.write(image_file.getbuffer())
            img_path = tmp.name
        user_msg["uploaded_image_path"] = img_path

    st.session_state.messages.append(user_msg)

    rag = st.session_state.rag_system
    search_query = prompt
    vlm_description = None
    
    if img_path:
        vlm_description = rag.analyze_image_intent(img_path, prompt)
        search_query = vlm_description

    retrieval = rag.retrieve(search_query)
    
    gen_prompt = f"User uploaded an image described as: '{vlm_description}'. User Question: {prompt}" if vlm_description else prompt
    answer = rag.generate_answer(gen_prompt, retrieval)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "images": retrieval["images"][:3] if retrieval["images"] else [],
        "vlm_analysis": vlm_description
    })


def main():
    load_rag_system()
    load_vosk()

    # Sidebar
    with st.sidebar:
        st.header("Document Ingestion")
        pdf_file = st.file_uploader("Upload PDF Manual", type=["pdf"])
        if pdf_file and st.button("Process PDF"):
            process_document(save_uploaded_file(pdf_file))
        
        st.markdown("---")
        st.markdown("### System Status")
        if st.session_state.rag_system.text_index:
            st.write(f"Index: {st.session_state.rag_system.text_index.ntotal} chunks")
        else:
            st.warning("No index found.")
        
        st.markdown("---")
        
        # Voice Recording
        st.markdown("### Voice Input")
        
        is_recording = stt.is_recording()
        
        if is_recording:
            st.warning("Recording... Click again to stop and send")
            if st.button("Stop & Send", type="primary", use_container_width=True):
                transcript = stt.stop_recording()
                if transcript:
                    # Auto-send the transcript
                    query_image = st.session_state.get("query_image")
                    process_query(transcript, query_image)
                    st.session_state.query_image = None
                st.rerun()
        else:
            if st.button("Start Recording", use_container_width=True):
                if st.session_state.get("vosk_loaded"):
                    stt.start_recording()
                    st.rerun()
                else:
                    st.error("Voice not available. Run: python download_models.py")
        
        st.markdown("---")
        
        # Image Upload
        st.markdown("### Attach Image")
        query_image = st.file_uploader(
            "Image for question",
            type=["jpg", "jpeg", "png"],
            key="sidebar_img",
            label_visibility="collapsed"
        )
        if query_image:
            st.session_state.query_image = query_image
            st.image(query_image, width=150)
            st.caption("Will attach to next question")
        else:
            if "query_image" not in st.session_state:
                st.session_state.query_image = None

    # Init state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat display
    display_chat_messages()

    # Chat input (fixed at bottom by Streamlit)
    if prompt := st.chat_input("Ask a question about the manual..."):
        query_image = st.session_state.get("query_image")
        process_query(prompt, query_image)
        st.session_state.query_image = None
        st.rerun()


if __name__ == "__main__":
    main()