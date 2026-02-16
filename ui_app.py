import streamlit as st
import os

from PIL import Image
from ingest import PDFProcessor, VectorStore
from rag import RAGSystem
from utils import MANUALS_DIR, IMAGES_DIR, setup_logger
import speech_to_text as stt
import tempfile
from pathlib import Path

logger = setup_logger(__name__)

st.set_page_config(page_title="SMART Assistant", layout="wide", page_icon="🚗")


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
    if Path("assets/SMART.png").exists():
        st.image("assets/SMART.png", width=500)


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
                num_imgs = len(msg["images"][:3])
                col_widths = [1] * num_imgs + [3]
                img_cols = st.columns(col_widths, gap="medium")
                for i, img_meta in enumerate(msg["images"][:3]):
                    with img_cols[i]:
                        try:
                            st.image(Image.open(img_meta["filepath"]), caption=f"Page {img_meta['page']}", use_container_width=True)
                        except:
                            pass


def queue_query(prompt, image_file=None):
    user_msg = {"role": "user", "content": prompt}
    img_path = None

    if image_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image_file.name.split('.')[-1]}") as tmp:
            tmp.write(image_file.getbuffer())
            img_path = tmp.name
        user_msg["uploaded_image_path"] = img_path

    st.session_state.messages.append(user_msg)
    st.session_state.pending_query = {"prompt": prompt, "img_path": img_path}


def run_pending_query():
    pending = st.session_state.pop("pending_query", None)
    if not pending:
        return

    prompt = pending["prompt"]
    img_path = pending["img_path"]
    rag = st.session_state.rag_system

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            search_query = prompt
            vlm_description = None

            if img_path:
                vlm_description = rag.analyze_image_intent(img_path, prompt)
                search_query = vlm_description

            retrieval = rag.retrieve(search_query)

            gen_prompt = (
                f"User uploaded an image described as: '{vlm_description}'. User Question: {prompt}"
                if vlm_description else prompt
            )
            answer = rag.generate_answer(gen_prompt, retrieval)

        st.markdown(answer)
        if vlm_description:
            st.info(f"**VLM Analysis:** {vlm_description}")
        if retrieval.get("images"):
            st.write("**Supporting Visuals:**")
            imgs = retrieval["images"][:3]
            num_imgs = len(imgs)
            col_widths = [1] * num_imgs + [3]
            img_cols = st.columns(col_widths, gap="medium")
            for i, img_meta in enumerate(imgs):
                with img_cols[i]:
                    try:
                        st.image(Image.open(img_meta["filepath"]), caption=f"Page {img_meta['page']}", use_container_width=True)
                    except:
                        pass

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "images": retrieval["images"][:3] if retrieval.get("images") else [],
        "vlm_analysis": vlm_description
    })


def main():
    load_rag_system()
    load_whisper()

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
        
        # Voice Input with st.audio_input
        st.markdown("### 🎤 Voice Input")
        
        audio_data = st.audio_input("Record your question", key="audio_input")
        
        if audio_data is not None:
            # Check if this is new audio (not already processed)
            audio_bytes = audio_data.getvalue()
            audio_hash = hash(audio_bytes)
            
            if st.session_state.get("last_audio_hash") != audio_hash:
                # Mark as processed immediately to prevent reprocessing
                st.session_state.last_audio_hash = audio_hash
                
                with st.spinner("Transcribing..."):
                    transcript = stt.transcribe_audio(audio_bytes)
                
                if transcript:
                    # Auto-send to chat
                    query_image = st.session_state.get("query_image")
                    queue_query(transcript, query_image)
                    st.session_state.query_image = None
                    st.rerun()
                else:
                    st.toast("Could not transcribe audio. Please try again.", icon="⚠️")
        
        st.markdown("---")
        
        # Image Upload
        st.markdown("### 📷 Attach Image")
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

    # Process any pending query
    run_pending_query()

    # Chat input (fixed at bottom)
    if prompt := st.chat_input("Ask a question about the manual..."):
        query_image = st.session_state.get("query_image")
        queue_query(prompt, query_image)
        st.session_state.query_image = None
        st.rerun()


if __name__ == "__main__":
    main()