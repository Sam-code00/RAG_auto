import streamlit as st
import os
import uuid
from pathlib import Path
from PIL import Image

from ingest import PDFProcessor, VectorStore
from rag import RAGSystem
from utils import MANUALS_DIR, setup_logger

logger = setup_logger(__name__)

# Offline flags
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

st.set_page_config(page_title="Smart Assistant", layout="wide", page_icon="🚗")

st.markdown("""
<style>
section[aria-label="Sidebar"] > div:first-child { height: 100vh; }
section[aria-label="Sidebar"] div[data-testid="stSidebarContent"]{
  height: 100%;
  display: flex;
  flex-direction: column;
}
.sidebar-bottom { margin-top: auto; }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([6, 1], vertical_alignment="center")

with col1:
    st.title("Smart Assistant")
    st.caption("Ask questions about your car manual.")

with col2:
    st.markdown("<div style='text-align:right;'>", unsafe_allow_html=True)
    st.image("assets/nsf.png", width=50)
    st.image("assets/nsf1.png", width=50)
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()


DATA_DIR = Path("data")
USER_UPLOADS_DIR = DATA_DIR / "user_uploads"
USER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def load_rag_system():
    if "rag_system" not in st.session_state:
        with st.spinner("Loading RAG System"):
            st.session_state.rag_system = RAGSystem()
            st.session_state.rag_system.load_models()


def save_uploaded_pdf(uploaded_file):
    path = MANUALS_DIR / uploaded_file.name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def save_uploaded_image(uploaded_img):
    path = USER_UPLOADS_DIR / f"{uuid.uuid4()}_{uploaded_img.name}"
    with open(path, "wb") as f:
        f.write(uploaded_img.getbuffer())
    return str(path)



def main():
    load_rag_system()

    with st.sidebar:
        st.header("Upload PDF Manual")
        pdf = st.file_uploader("", type=["pdf"])

        if pdf and st.button("Process PDF"):
            processor = PDFProcessor()
            text_chunks, images_meta = processor.process_pdf(str(save_uploaded_pdf(pdf)))
            text_emb = processor.embed_text(text_chunks)
            img_emb, valid_imgs = processor.embed_images(images_meta)

            store = VectorStore()
            store.build_index(text_chunks, valid_imgs, text_emb, img_emb)
            store.save()
            st.session_state.rag_system.refresh_index()
            st.toast("Manual Indexed", icon="✅")

        st.divider()

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.pop("pending_img", None)

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

        st.markdown('</div>', unsafe_allow_html=True)

    # Chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_img" not in st.session_state:
        st.session_state.pending_img = None  # stores filepath to attach to next message

    # Render chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("user_img"):
                st.image(msg["user_img"], width=250)

            imgs = msg.get("images") or []
            if imgs:
                cols = st.columns(min(3, len(imgs)))
                for i, im in enumerate(imgs[:3]):
                    with cols[i]:
                        try:
                            st.image(im["filepath"], caption=f"Page {im['page']}")
                        except:
                            pass

    prompt = st.chat_input("Ask a question about your manual...")

    if prompt:
        img_path = st.session_state.pending_img
        st.session_state.pending_img = None

        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "user_img": img_path}
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                rag = st.session_state.rag_system

                image_hint = ""
                if img_path and hasattr(rag, "describe_user_image"):
                    try:
                        image_hint = rag.describe_user_image(img_path)
                    except Exception as e:
                        logger.warning(f"describe_user_image failed: {e}")

                try:
                    results = rag.retrieve(prompt, image_hint=image_hint)
                except TypeError:
                    results = rag.retrieve(prompt)

                answer = rag.generate_answer(prompt, results)
                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "images": results.get("images", [])[:3],
                    }
                )


if __name__ == "__main__":
    main()
