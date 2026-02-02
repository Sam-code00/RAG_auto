import streamlit as st
import os
# To allow offline usage
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from PIL import Image
from ingest import PDFProcessor, VectorStore

from rag import RAGSystem
from utils import MANUALS_DIR, IMAGES_DIR, setup_logger
import io
import tempfile
from pathlib import Path

logger = setup_logger(__name__)

st.set_page_config(page_title="Local Multimodal RAG", layout="wide", page_icon="")

col1, col2 = st.columns([5, 1])
with col1:
    st.title("SMART Assistant")
    st.caption("Ask questions about your vehicle's manual. Upload a photo for visual guidance!")
    
with col2:
    col3, col4 = st.columns([1, 1])
    with col3:
        logo_path = Path("assets/SMART.png")
        if logo_path.exists():
            st.image(str(logo_path), width=80)
    with col4:
        logo_path2 = Path("assets/nsf1.png")
        if logo_path2.exists():
            st.image(str(logo_path2), width=80)      
    



def load_css():
    with open("assets/style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

def load_rag_system():
   
    if "rag_system" not in st.session_state:
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
    processor = PDFProcessor()
    text_chunks, images_meta = processor.process_pdf(str(file_path))
    
    st.toast(f"Extracted {len(text_chunks)} text chunks and {len(images_meta)} images.", icon="📄")
    
    text_emb = processor.embed_text(text_chunks)
    img_emb, valid_imgs = processor.embed_images(images_meta)
    
    store = VectorStore()
    store.build_index(text_chunks, valid_imgs, text_emb, img_emb)
    store.save()
    
    st.session_state.rag_system.refresh_index()
    st.toast("Indexing Complete!", icon="✅")


def main():
    load_rag_system()

    with st.sidebar:
        st.header("Document Ingestion")
        uploaded_file = st.file_uploader("Upload PDF Manual", type=["pdf"])
        if uploaded_file:
            if st.button("Process PDF"):
                file_path = save_uploaded_file(uploaded_file)
                process_document(file_path)
        
        st.markdown("---")
        st.markdown("### System Status")
        if st.session_state.rag_system.text_index:
            st.write(f" Index contains {st.session_state.rag_system.text_index.ntotal} text chunks.")
        else:
            st.warning("No index found.")
        
        st.markdown("---")

    # Main Chat 
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("uploaded_image_path"):
                st.image(message["uploaded_image_path"], caption="Uploaded Image", width=250)
            if "images" in message:
                cols = st.columns(len(message["images"]))
                for idx, img_meta in enumerate(message["images"]):
                    with cols[idx]:
                        try:
                            image = Image.open(img_meta["filepath"])
                            st.image(image, caption=f"Page {img_meta['page']}", use_column_width=True)
                        except:
                            st.write("Image not found")

    # Main Chat Input
    query_image = st.file_uploader(
        "Attach Image (Optional)",
        type=["jpg", "png", "jpeg"],
        key="query_img",
        label_visibility="collapsed"
    )

    if prompt := st.chat_input("Ask a question about uploaded manual"):
        user_msg = {"role": "user", "content": prompt}
        

        tmp_path = None
        if query_image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{query_image.name.split('.')[-1]}") as tmp:
                tmp.write(query_image.getbuffer())
                tmp_path = tmp.name
            user_msg["uploaded_image_path"] = tmp_path

        with st.chat_message("user"):
            st.markdown(prompt)
            if tmp_path:
                st.image(tmp_path, caption="Uploaded Image", width=250)

        st.session_state.messages.append(user_msg)

        with st.chat_message("assistant"):
            rag = st.session_state.rag_system
            
            search_query = prompt
            
            # VLM Analysis if image is present
            vlm_description = None
            if tmp_path:
                with st.spinner("Analyzing image..."):
                    inferred_query = rag.analyze_image_intent(tmp_path, prompt)
                    st.info(f"**VLM Analysis:** {inferred_query}")
                    search_query = inferred_query
                    vlm_description = inferred_query

            retrieval_results = rag.retrieve(search_query)
            
            # Construct generation prompt
            generation_prompt = prompt
            if vlm_description:
                generation_prompt = f"User uploaded an image described as: '{vlm_description}'. User Question: {prompt}"

            answer = rag.generate_answer(generation_prompt, retrieval_results)
            st.markdown(answer)

            # Display Images (Images set to show after the answer)
            images_to_show = retrieval_results["images"]
            if images_to_show:
                st.write("**Supporting Visuals from Manual:**")
                cols = st.columns(min(3, len(images_to_show)))
                for i, img_meta in enumerate(images_to_show[:3]):
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
