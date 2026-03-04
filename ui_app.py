import os
import tempfile
from pathlib import Path
import shutil
import streamlit as st
from PIL import Image

from ingest import PDFProcessor, VectorStore
from rag import RAGSystem
from utils import MANUALS_DIR, INDEX_DIR, IMAGES_DIR, setup_logger
import speech_to_text as stt

logger = setup_logger(__name__)

st.set_page_config(page_title="SMART Assistant", layout="wide", page_icon="🚗")
UPLOAD_DIR = IMAGES_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Load CSS
css_path = Path("assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{open(css_path).read()}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 340px; max-width: 340px; }
[data-testid="stSidebar"] > div:first-child { width: 340px; }
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] {
    background: none !important; border: none !important; box-shadow: none !important;
    padding: 0 !important; width: auto !important; color: #cc4444 !important;
    font-size: 16px !important; text-decoration: none !important;
    min-height: 0 !important; height: auto !important; margin-top: 10px !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"]:hover {
    color: #ff4444 !important; transform: none !important; box-shadow: none !important;
}
</style>""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([5, 1])
with col1:
    st.title("SMART Assistant")
    st.caption("Ask questions about your vehicle's manual. Upload a photo for visual guidance!")
with col2:
    if Path("assets/SMART.png").exists():
        st.image("assets/SMART.png", width=300)

NO_INFO_PHRASES = [
    "couldn't find this information", "could not find this information",
    "not in the manual", "not found in the", "no information available",
    "couldn't find information", "could not find information",
    "i couldn't find", "i could not find",
]


# Shared rendering

def render_images(images):
    seen = set()
    unique = [m for m in images if not (m.get("filepath", "") in seen or seen.add(m.get("filepath", "")))]
    imgs = unique[:3]
    if not imgs:
        return
    st.write("**Supporting Visuals:**")
    cols = st.columns(len(imgs), gap="medium")
    for i, meta in enumerate(imgs):
        with cols[i]:
            try:
                doc_label = meta.get("doc_id", "")
                caption = f"{doc_label} — Page {meta['page']}" if doc_label else f"Page {meta['page']}"
                st.image(Image.open(meta["filepath"]), caption=caption, width=300)
            except Exception:
                pass


def render_context(retrieval):
    with st.expander("🔎 Retrieved Manual Context (Top matches)"):
        if retrieval.get("text"):
            for i, chunk in enumerate(retrieval["text"], start=1):
                st.markdown(f"**{i}. Page {chunk.get('page', '?')}** — `{chunk.get('doc_id', '')}`")
                txt = chunk.get("text", "")
                st.write(txt[:600] + ("..." if len(txt) > 600 else ""))
                st.markdown("---")
        else:
            st.write("No text chunks retrieved.")


def render_answer(rag, prompt_text, vlm_description, retrieval):
    gen_prompt = (
        f"The assistant identified the relevant manual topic as: '{vlm_description}'. User Question: {prompt_text}"
        if vlm_description else prompt_text
    )
    answer = rag.generate_answer(gen_prompt, retrieval)

    if any(p in answer.lower() for p in NO_INFO_PHRASES):
        retrieval["images"] = []

    st.markdown(answer)

    source_docs = retrieval.get("source_docs", [])
    if len(source_docs) == 1:
        st.caption(f"📚 Source: `{source_docs[0]}`")

    if vlm_description:
        st.info(f"**VLM Analysis:** {vlm_description}")

    if retrieval.get("images"):
        render_images(retrieval["images"])

    render_context(retrieval)
    return answer


# Init

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
        except Exception as e:
            st.session_state.whisper_loaded = False
            logger.warning(f"Whisper not loaded: {e}")


# Document processing

def process_document(uploaded_file):
    save_path = MANUALS_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.status("Processing Manual...", expanded=True) as status:
        progress = st.progress(0)
        processor = PDFProcessor()

        st.write("Extracting text and diagrams...")
        text_chunks, images_meta = processor.process_pdf(str(save_path))
        progress.progress(25)
        st.write(f"Found {len(text_chunks)} text segments and {len(images_meta)} images.")

        st.write("Generating text embeddings...")
        text_emb = processor.embed_text(text_chunks)
        progress.progress(50)

        st.write("Running SigLIP2 on diagrams...")
        img_emb, valid_imgs = processor.embed_images(images_meta)
        progress.progress(75)

        store = st.session_state.rag_system.vector_store
        store.load()
        store.build_index(text_chunks, valid_imgs, text_emb, img_emb)
        store.save()
        st.session_state.rag_system.refresh_index()
        progress.progress(100)
        status.update(label="Manual Processing Complete!", state="complete", expanded=False)

    st.balloons()
    st.toast("Manual is ready for questions.")


# Chat display

def _on_pick_doc(doc):
    st.session_state.picked_doc = doc


def display_chat_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("uploaded_image_path"):
                try:
                    st.image(msg["uploaded_image_path"], caption="Attached Image", width=260)
                except Exception:
                    pass

            if msg.get("source_doc"):
                st.caption(f"📚 Source: `{msg['source_doc']}`")

            if msg.get("vlm_analysis"):
                st.info(f"**VLM Analysis:** {msg['vlm_analysis']}")

            if msg.get("disambiguation") and st.session_state.get("disambiguation"):
                source_docs = msg.get("source_docs", [])
                if source_docs:
                    cols = st.columns(len(source_docs))
                    for i, doc in enumerate(source_docs):
                        with cols[i]:
                            st.button(doc, key=f"hist_pick_{doc}", use_container_width=True,
                                      on_click=_on_pick_doc, args=(doc,))

            if msg.get("images"):
                render_images(msg["images"])


# Query handling

def queue_query(prompt_text, image_bytes=None, image_name="upload.png"):
    user_msg = {"role": "user", "content": prompt_text}
    img_path = None
    if image_bytes is not None:
        try:
            suffix = "." + image_name.split(".")[-1] if "." in image_name else ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(image_bytes)
                img_path = tmp.name
            user_msg["uploaded_image_path"] = img_path
        except Exception:
            pass
    st.session_state.messages.append(user_msg)
    st.session_state.pending_query = {"prompt": prompt_text, "img_path": img_path}


def run_pending_query():
    pending = st.session_state.pop("pending_query", None)
    if not pending:
        return

    rag = st.session_state.rag_system
    prompt_text = pending["prompt"]
    img_path = pending["img_path"]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            search_query = prompt_text
            vlm_description = None
            if img_path:
                st.toast("Processing image...")
                vlm_description = rag.analyze_image_intent(img_path, prompt_text)
                search_query = vlm_description

            retrieval = rag.retrieve(search_query, doc_id=st.session_state.get("active_doc_id"))
            source_docs = retrieval.get("source_docs", [])

        # Multiple manuals matched — ask user to pick
        if st.session_state.get("active_doc_id") is None and len(source_docs) > 1:
            docs_str = ", ".join(f"**{d}**" for d in source_docs)
            st.markdown(f"I found relevant information in multiple manuals: {docs_str}")
            st.markdown("Which manual are you asking about?")

            st.session_state.disambiguation = {
                "prompt": prompt_text, "search_query": search_query,
                "vlm_description": vlm_description, "img_path": img_path,
                "source_docs": source_docs,
            }
            cols = st.columns(len(source_docs))
            for i, doc in enumerate(source_docs):
                with cols[i]:
                    st.button(doc, key=f"pick_{doc}", use_container_width=True,
                              on_click=_on_pick_doc, args=(doc,))

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I found relevant information in multiple manuals: {docs_str}\nWhich manual are you asking about?",
                "disambiguation": True, "source_docs": source_docs,
            })
            return

        # Single manual — answer directly
        answer = render_answer(rag, prompt_text, vlm_description, retrieval)
        st.session_state.messages.append({
            "role": "assistant", "content": answer,
            "images": retrieval.get("images", [])[:3], "vlm_analysis": vlm_description,
        })


def run_disambiguation():
    picked = st.session_state.pop("picked_doc", None)
    disambig = st.session_state.get("disambiguation")
    if not picked or not disambig:
        return False

    rag = st.session_state.rag_system
    st.session_state.messages.append({"role": "user", "content": picked})

    # Disable buttons on old disambiguation message
    for msg in st.session_state.messages:
        if msg.get("disambiguation"):
            msg["disambiguation"] = False

    display_chat_messages()

    with st.chat_message("assistant"):
        with st.spinner("Searching manual..."):
            retrieval = rag.retrieve(disambig["search_query"], doc_id=picked)
            answer = render_answer(rag, disambig["prompt"], disambig["vlm_description"], retrieval)

        source_docs = retrieval.get("source_docs", [])

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "images": retrieval.get("images", [])[:3], "vlm_analysis": disambig["vlm_description"],
        "source_doc": source_docs[0] if len(source_docs) == 1 else None,
    })
    st.session_state.pop("disambiguation", None)
    return True


# Main

def main():
    load_rag_system()
    load_whisper()
    rag = st.session_state.rag_system

    if rag.text_index:
        st.toast(f"Index: {rag.text_index.ntotal} chunks", icon="📚")
    else:
        st.toast("No index found. Upload a PDF manual to get started.", icon="📂")

    with st.sidebar:
        st.header("Document Ingestion")
        pdf_file = st.file_uploader("Upload PDF Manual", type=["pdf"])
        if pdf_file and st.button("Process PDF"):
            process_document(pdf_file)

        st.markdown("---")
        st.subheader("Indexed Manuals")
        indexed_docs = rag.vector_store.get_indexed_doc_ids()

        if indexed_docs:
            for doc in indexed_docs:
                chunks = sum(1 for c in rag.vector_store.text_metadata if isinstance(c, dict) and c.get("doc_id") == doc)
                imgs = sum(1 for m in rag.vector_store.image_metadata if isinstance(m, dict) and m.get("doc_id") == doc)
                c1, c2 = st.columns([6, 1], vertical_alignment="center")
                with c1:
                    st.markdown(f"""
                    <div style="padding:12px 16px;border-radius:10px;background:#1e1e1e;border:1px solid #333;">
                        <div style="font-weight:600;font-size:15px;">📄 {doc}</div>
                        <div style="font-size:12px;color:#999;margin-top:4px;">{chunks} text chunks · {imgs} images</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    if st.button("✕", key=f"del_{doc}", type="tertiary"):
                        rag.vector_store.remove_doc(doc)
                        rag.refresh_index()
                        st.rerun()
                st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)
        else:
            st.caption("No manuals indexed yet.")

        if len(indexed_docs) == 1:
            st.session_state.active_doc_id = indexed_docs[0]
        elif len(indexed_docs) > 1:
            st.markdown("---")
            st.subheader("Select Manual")
            selected = st.selectbox("Answer from:", indexed_docs, index=0)
            search_all = st.toggle("Search all manuals", value=False)
            st.session_state.active_doc_id = None if search_all else selected
        else:
            st.session_state.active_doc_id = None

        st.markdown("---")
        if st.button("Clear Chat"):
            st.session_state.messages = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    did_disambiguate = run_disambiguation()
    if not did_disambiguate:
        display_chat_messages()

    run_pending_query()

    prompt = st.chat_input(
        placeholder="Ask a question about the manual...",
        accept_file="multiple",
        file_type=["jpg", "jpeg", "png", "webp"],
        accept_audio=True, audio_sample_rate=16000, key="chat_input",
    )

    if prompt:
        text = (prompt.text or "").strip()

        if not text and getattr(prompt, "audio", None):
            try:
                with st.spinner("Transcribing audio..."):
                    text = (stt.transcribe_audio(prompt.audio.getvalue()) or "").strip()
            except Exception as e:
                logger.error(f"Transcription failed: {e}")

        img_bytes, img_name = None, "upload.png"
        for f in getattr(prompt, "files", []):
            ftype = (getattr(f, "type", "") or "").lower()
            fname = (getattr(f, "name", "") or "").lower()
            if ftype.startswith("image/") or fname.endswith((".jpg", ".jpeg", ".png", ".webp")):
                try:
                    img_bytes = f.getvalue()
                    img_name = f.name or img_name
                except Exception:
                    pass
                break

        if text or img_bytes:
            if img_bytes:
                with open(UPLOAD_DIR / f"query_{img_name}", "wb") as f:
                    f.write(img_bytes)
            queue_query(text or "[Image Upload]", image_bytes=img_bytes, image_name=img_name)
            st.rerun()
        else:
            st.toast("Please provide a question, image, or audio.", icon="⚠️")


if __name__ == "__main__":
    main()