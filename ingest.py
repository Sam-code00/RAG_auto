import fitz  # PyMuPDF
import os
import re
import uuid
import pickle
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
from utils import (
    IMAGES_DIR, INDEX_DIR, TEXT_EMBED_MODEL, IMAGE_EMBED_MODEL,
    setup_logger
)

logger = setup_logger(__name__)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Config
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 60
MIN_IMAGE_DIM = 50


# Helpers
def _block_text(blk: dict) -> str:
    lines = []
    for line in blk.get("lines", []):
        spans = "".join(s.get("text", "") for s in line.get("spans", []))
        lines.append(spans)
    return "\n".join(lines)


def _table_to_markdown(table) -> str:
    try:
        rows = table.extract()
        if not rows:
            return ""
        clean = lambda c: str(c).replace("\n", " ").replace("|", "/").strip() if c else ""
        hdr = rows[0]
        lines = [
            "| " + " | ".join(clean(c) for c in hdr) + " |",
            "| " + " | ".join("---" for _ in hdr) + " |",
        ]
        for row in rows[1:]:
            padded = list(row) + [""] * (len(hdr) - len(row))
            lines.append("| " + " | ".join(clean(c) for c in padded[:len(hdr)]) + " |")
        return "\n".join(lines)
    except Exception:
        return ""


def _split_long_text(text: str, max_sz: int) -> list[str]:
    if len(text) <= max_sz:
        return [text]
    for sep, pattern in [("\n\n", r'\n\n+'), (" ", r'(?<=[.!?])\s+')]:
        parts = re.split(pattern, text)
        if len(parts) > 1:
            return _greedy_merge(parts, max_sz, sep)
    return [text[i:i+max_sz] for i in range(0, len(text), max_sz)]


def _greedy_merge(parts, max_sz, sep):
    chunks, cur, cur_len = [], [], 0
    for p in parts:
        add = len(p) + (len(sep) if cur else 0)
        if cur_len + add > max_sz and cur:
            chunks.append(sep.join(cur))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += add
    if cur:
        chunks.append(sep.join(cur))
    return chunks


def _find_caption(blocks, img_idx):
    cap_re = re.compile(r'(?i)^(?:figure|fig\.?|diagram|illustration|image|photo)\s*\d*')
    for off in [1, -1, 2, -2]:
        j = img_idx + off
        if 0 <= j < len(blocks) and blocks[j]["type"] == "text":
            t = blocks[j]["text"].strip()
            if cap_re.match(t):
                return t[:400]
            if len(t) < 200 and off in [1, -1]:
                return t
    return ""


# Page Processing
def _process_page(page, doc, page_num, doc_id):
    blocks = []
    page_no = page_num + 1

    # ── Tables ──
    table_rects = []
    if hasattr(page, "find_tables"):
        try:
            for tbl in page.find_tables().tables:
                bbox = tbl.bbox
                r = fitz.Rect(bbox) if isinstance(bbox, (list, tuple)) else fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
                table_rects.append(r)
                md = _table_to_markdown(tbl)
                if md and len(md.strip()) >= MIN_CHUNK_CHARS:
                    blocks.append({"type": "table", "text": md, "y": r.y0, "page": page_no})
        except Exception as e:
            logger.debug(f"Table extraction p{page_no}: {e}")

    # Text blocks (skip table regions)
    for blk in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
        bbox = blk.get("bbox")
        if not bbox or blk.get("type") != 0:
            continue
        r = fitz.Rect(bbox)
        if any(r.intersects(tr) for tr in table_rects):
            continue
        t = _block_text(blk)
        if t and len(t.strip()) > 3:
            blocks.append({"type": "text", "text": t.strip(), "y": r.y0, "page": page_no})

    # Images
    for img_i, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]
        try:
            bi = doc.extract_image(xref)
        except Exception:
            continue
        w, h = bi.get("width", 0), bi.get("height", 0)
        if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
            continue
        fname = f"{os.path.splitext(doc_id)[0]}_p{page_no}_{img_i}.{bi['ext']}"
        fpath = IMAGES_DIR / fname
        with open(fpath, "wb") as f:
            f.write(bi["image"])
        img_y = 0.0
        try:
            rects = page.get_image_rects(xref)
            if rects:
                img_y = rects[0].y0
        except Exception:
            pass
        blocks.append({
            "type": "image", "filepath": str(fpath),
            "y": img_y, "page": page_no, "dimensions": f"{w}x{h}",
        })

    blocks.sort(key=lambda b: b["y"])

    # Merge text blocks into chunks; tables stay standalone
    text_chunks, images_meta = [], []
    buf, buf_len = [], 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        merged = "\n\n".join(buf)
        if len(merged.strip()) >= MIN_CHUNK_CHARS:
            text_chunks.append({
                "id": str(uuid.uuid4()), "doc_id": doc_id,
                "page": page_no, "text": merged.strip(),
                "chunk_type": "text",
            })
        buf, buf_len = [], 0

    for i, blk in enumerate(blocks):
        if blk["type"] == "text":
            t = blk["text"]
            if buf_len + len(t) > MAX_CHUNK_CHARS and buf:
                flush()
            if len(t) > MAX_CHUNK_CHARS:
                flush()
                for sc in _split_long_text(t, MAX_CHUNK_CHARS):
                    if len(sc.strip()) >= MIN_CHUNK_CHARS:
                        text_chunks.append({
                            "id": str(uuid.uuid4()), "doc_id": doc_id,
                            "page": page_no, "text": sc.strip(),
                            "chunk_type": "text",
                        })
            else:
                buf.append(t)
                buf_len += len(t)
        elif blk["type"] == "table":
            flush()
            text_chunks.append({
                "id": str(uuid.uuid4()), "doc_id": doc_id,
                "page": page_no, "text": blk["text"],
                "chunk_type": "table",
            })
        elif blk["type"] == "image":
            caption = _find_caption(blocks, i)
            images_meta.append({
                "id": str(uuid.uuid4()), "doc_id": doc_id,
                "page": page_no, "filepath": blk["filepath"],
                "context": caption or f"Image on page {page_no} of {doc_id}",
                "caption": caption, "dimensions": blk.get("dimensions", ""),
            })
    flush()
    return text_chunks, images_meta


# PDFProcessor
class PDFProcessor:
    def __init__(self):
        try:
            self.clip_model = CLIPModel.from_pretrained(IMAGE_EMBED_MODEL, local_files_only=True)
            self.clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBED_MODEL, local_files_only=True)
        except Exception as e:
            logger.error(f"CLIP load failed: {e}. Run 'python download_models.py'.")
            raise

    def process_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        doc_id = os.path.basename(pdf_path)
        all_chunks, all_images = [], []
        logger.info(f"Processing {doc_id} ({len(doc)} pages)...")
        for pn in range(len(doc)):
            tc, im = _process_page(doc[pn], doc, pn, doc_id)
            all_chunks.extend(tc)
            all_images.extend(im)
        doc.close()
        nt = sum(1 for c in all_chunks if c["chunk_type"] == "text")
        ntb = sum(1 for c in all_chunks if c["chunk_type"] == "table")
        logger.info(f"Done: {len(all_chunks)} chunks ({nt} text, {ntb} table), {len(all_images)} images")
        return all_chunks, all_images

    def embed_text(self, chunks):
        import ollama
        embeddings = []
        for c in chunks:
            prefix = "[TABLE] " if c.get("chunk_type") == "table" else ""
            text = prefix + c["text"]
            if len(text) > 8000:
                text = text[:8000]
            resp = ollama.embeddings(model=TEXT_EMBED_MODEL, prompt=text)
            embeddings.append(resp["embedding"])
        return np.array(embeddings).astype('float32')

    def embed_images(self, images_metadata):
        images, valid_idx = [], []
        for i, m in enumerate(images_metadata):
            try:
            # Open + force decode + normalize mode (prevents PIL crashes in CLIP preprocessing)
                with Image.open(m["filepath"]) as im:
                    im = im.convert("RGB")

                    if im.size[0] < MIN_IMAGE_DIM or im.size[1] < MIN_IMAGE_DIM:
                        continue
                    images.append(im.copy())
                    valid_idx.append(i)

            except Exception as e:
                logger.error(f"Image load failed {m['filepath']}: {type(e).__name__}: {e}")
                continue
        if not images:
            return np.array([]).astype('float32'), []
        inputs = self.clip_processor(images=images, return_tensors="pt")
        feats = self.clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.detach().numpy().astype('float32'), [images_metadata[i] for i in valid_idx]


# VectorStore
class VectorStore:
    def __init__(self):
        self.text_index = None
        self.image_index = None
        self.text_metadata = []
        self.image_metadata = []

    def build_index(self, text_chunks, images_metadata, text_embeddings, image_embeddings):
        if len(text_chunks) > 0:
            d = text_embeddings.shape[1]
            self.text_index = faiss.IndexFlatL2(d)
            self.text_index.add(text_embeddings)
            self.text_metadata = text_chunks
        if len(images_metadata) > 0 and len(image_embeddings) > 0:
            d = image_embeddings.shape[1]
            self.image_index = faiss.IndexFlatL2(d)
            self.image_index.add(image_embeddings)
            self.image_metadata = images_metadata

    def save(self):
        if self.text_index:
            faiss.write_index(self.text_index, str(INDEX_DIR / "text.index"))
        if self.image_index:
            faiss.write_index(self.image_index, str(INDEX_DIR / "image.index"))
        with open(INDEX_DIR / "metadata.pkl", "wb") as f:
            pickle.dump({"text": self.text_metadata, "image": self.image_metadata}, f)
        logger.info("Index saved.")

    def load(self):
        try:
            self.text_index = faiss.read_index(str(INDEX_DIR / "text.index"))
            if (INDEX_DIR / "image.index").exists():
                self.image_index = faiss.read_index(str(INDEX_DIR / "image.index"))
            with open(INDEX_DIR / "metadata.pkl", "rb") as f:
                m = pickle.load(f)
                self.text_metadata = m["text"]
                self.image_metadata = m["image"]
            logger.info("Index loaded.")
            return True
        except Exception as e:
            logger.warning(f"Index load failed: {e}")
            return False