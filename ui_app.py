import streamlit as st
import os
# To allow offline usage
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from PIL import Image
from ingest import PDFProcessor, VectorStore

from rag import RAGSystem
from utils import MANUALS_DIR, IMAGES_DIR, setup_logger

logger = setup_logger(__name__)
st.set_page_config(page_title="Local Multimodal RAG", layout="wide", page_icon="🚗")

header_left, header_center, header_right = st.columns([2, 6, 1.5])

with header_left:
    st.image("assets/logo_main.png", width=500)

with header_center:
    st.markdown(
        """
        <div style="margin-top: -20px;">
            <h1 style="margin-bottom: 0;">Local Multimodal RAG</h1>
            <p style="margin-top: 2px; color: #6b7280;">
                Ask questions about your car's manual, including images or tables.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:
    st.image(str("assets/nsf.png"), width=60)
    st.image(str("assets/nsf1.png"), width=60)

st.divider()

def load_css():
    with open("assets/style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

def load_rag_system():
   
    if "rag_system" not in st.session_state:
        with st.spinner("Loading RAG System"):
            try:
                st.session_state.rag_system = RAGSystem()
                st.session_state.rag_system.load_models()

            except Exception as e:
                st.error(f"Failed to load system: {e}")
                st.stop()

def save_uploaded_file(uploaded_file):
    save_path = MANUALS_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def process_document(file_path):
    with st.spinner(f"{file_path.name} is currently being processed"):
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
        st.success("Indexing Complete!")

def main():

    load_rag_system()

    with st.sidebar:
        
        st.title("Upload PDF Manual")
        uploaded_file = st.file_uploader("", type=["pdf"])

        if uploaded_file:
            if st.button("Process PDF"):
                file_path = save_uploaded_file(uploaded_file)
                process_document(file_path)

        st.divider()

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.success("Chat history cleared.")

    # Main Chat 
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "images" in message:
                cols = st.columns(len(message["images"]))
                for idx, img_meta in enumerate(message["images"]):
                    with cols[idx]:
                        try:
                            image = Image.open(img_meta["filepath"])
                            st.image(image, caption=f"Page {img_meta['page']}", use_container_width=True)
                        except:
                            st.write("Image not found")

    if prompt := st.chat_input("Ask a question about your manual..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                rag = st.session_state.rag_system
                retrieval_results = rag.retrieve(prompt)
                
                
                answer = rag.generate_answer(prompt, retrieval_results)
                
                st.markdown(answer)
                
                # Display Images (Images are set to show after the answer)
                images_to_show = retrieval_results["images"]
                if images_to_show:
                    st.write("**Supporting Visuals:**")
                    cols = st.columns(min(3, len(images_to_show)))
                    for i, img_meta in enumerate(images_to_show[:3]):
                        # The amount of images to show is 3 for now
                         with cols[i]:
                            try:
                                image = Image.open(img_meta["filepath"])
                                st.image(image, caption=f"Page {img_meta['page']}")
                            except:
                                st.error(f"Missing file: {img_meta['filepath']}")
                
                # Save
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "images": images_to_show[:3]
                })

if __name__ == "__main__":
    main()
