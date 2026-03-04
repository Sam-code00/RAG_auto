import fitz  # PyMuPDF
import io
import os
import re
import uuid
import pickle
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
import faiss
from utils import (
    IMAGES_DIR, INDEX_DIR, TEXT_EMBED_MODEL, IMAGE_EMBED_MODEL,
    setup_logger
)

logger = setup_logger(__name__)

# Config
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 60
MIN_IMAGE_DIM = 50  # min rasterized image dimension (pixels)

# Vector-diagram detection config (safe defaults)
MIN_VECTOR_AREA = 600      # min cluster area in pt^2
GRAPHICS_LIMIT = 5000      # cap drawing objects per page (performance)
DEFAULT_TOL = 5            # cluster tolerance
MIN_PATH_COUNT = 3         # cluster needs at least this many drawing paths


# -----------------
# Helpers: text/tables
# -----------------
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


def _find_caption(blocks, img_idx: int) -> str:
    # Captions often appear immediately above/below an image/diagram.
    cap_re = re.compile(r'(?i)^(?:figure|fig\.?|diagram|illustration|image|photo)\s*\d*')
    for off in [1, -1, 2, -2]:
        j = img_idx + off
        if 0 <= j < len(blocks) and blocks[j]["type"] == "text":
            t = blocks[j]["text"].strip()
            if cap_re.match(t):
                return t[:400]
            # fallback: short nearby text
            if len(t) < 200 and off in [1, -1]:
                return t
    return ""


# -----------------
# Helpers: geometry + vector diagram detection
# -----------------
def _overlap_frac(a: "fitz.Rect", b: "fitz.Rect") -> float:
    ix = a & b
    if ix.is_empty:
        return 0.0
    aa = a.width * a.height
    return (ix.width * ix.height / aa) if aa > 0 else 0.0


def _is_line(r: "fitz.Rect", ratio: float = 10.0) -> bool:
    if r.width <= 0 or r.height <= 0:
        return True
    return max(r.width / r.height, r.height / r.width) > ratio and min(r.width, r.height) < 5


def _text_coverage(page, rect: "fitz.Rect") -> float:
    area = rect.width * rect.height
    if area <= 0:
        return 1.0
    covered = 0.0
    try:
        td = page.get_text("dict", clip=rect, flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for blk in td.get("blocks", []):
            if blk.get("type") != 0:
                continue
            bb = blk.get("bbox")
            if not bb:
                continue
            ix = rect & fitz.Rect(bb)
            if not ix.is_empty:
                covered += ix.width * ix.height
    except Exception:
        pass
    return covered / area


def _filter_decorations(drawings: list, page_width: float) -> list:
    """Remove page-wide separators / hairlines that glue clusters together."""
    filtered = []
    for d in drawings:
        dr = d.get("rect")
        if dr is None:
            filtered.append(d)
            continue
        r = fitz.Rect(dr)
        if r.is_empty or r.is_infinite:
            continue

        # Skip header/footer bars / separators that span most of the page.
        if r.width > page_width * 0.7 and r.height < 30:
            continue

        # Skip very thin horizontal/vertical lines
        if r.width > 0 and r.height > 0:
            aspect = max(r.width / r.height, r.height / r.width)
            if aspect > 15 and min(r.width, r.height) < 4:
                continue

        filtered.append(d)
    return filtered


def _trim_to_paths(drawings: list, cluster_rect: "fitz.Rect") -> "fitz.Rect":
    """Tighten a cluster bbox to the union of the drawing path rects inside it."""
    tight = None
    for d in drawings:
        dr = d.get("rect")
        if dr is None:
            continue
        pr = fitz.Rect(dr)
        if pr.is_empty or pr.is_infinite:
            continue
        if not cluster_rect.intersects(pr):
            continue
        tight = fitz.Rect(pr) if tight is None else (tight | pr)
    return tight if tight else cluster_rect


def _shrink_past_text(page, rect: "fitz.Rect") -> "fitz.Rect":
    """Shave off edges that are mostly text, to avoid swallowing labels/paragraphs."""
    r = fitz.Rect(rect)
    strip = 30

    if r.width > strip * 3:
        left_strip = fitz.Rect(r.x0, r.y0, r.x0 + strip, r.y1)
        if _text_coverage(page, left_strip) > 0.5:
            r.x0 += strip

        right_strip = fitz.Rect(r.x1 - strip, r.y0, r.x1, r.y1)
        if _text_coverage(page, right_strip) > 0.5:
            r.x1 -= strip

    if r.height > strip * 3:
        top_strip = fitz.Rect(r.x0, r.y0, r.x1, r.y0 + strip)
        if _text_coverage(page, top_strip) > 0.5:
            r.y0 += strip

        bottom_strip = fitz.Rect(r.x0, r.y1 - strip, r.x1, r.y1)
        if _text_coverage(page, bottom_strip) > 0.5:
            r.y1 -= strip

    return r


def _count_paths_in_rect(drawings: list, rect: "fitz.Rect") -> int:
    count = 0
    for d in drawings:
        dr = d.get("rect")
        if dr is None:
            continue
        pr = fitz.Rect(dr)
        if pr.is_empty or pr.is_infinite:
            continue
        if rect.intersects(pr):
            count += 1
    return count


def _is_diagram(page, rect: "fitz.Rect", drawings: list, path_count: int) -> bool:
    """Heuristic: path density + quick 'distinct colors' check from a low-res render."""
    area = rect.width * rect.height
    if area <= 0:
        return False

    if path_count < MIN_PATH_COUNT:
        return False

    page_rect = page.rect
    if rect.width > page_rect.width * 0.85 and rect.height > page_rect.height * 0.85:
        return False

    try:
        mat = fitz.Matrix(0.3, 0.3)
        pix = page.get_pixmap(clip=rect, matrix=mat, alpha=False)
        w, h = pix.width, pix.height
        if w < 3 or h < 3:
            return False

        n = pix.n
        samples = pix.samples
        total_pixels = w * h

        colors = set()
        step = max(1, total_pixels // 200)  # sample ~200 pixels
        for i in range(0, total_pixels, step):
            off = i * n
            if off + n > len(samples):
                break
            # quantize RGB to reduce noise
            colors.add(tuple(b >> 5 for b in samples[off:off + n]))

        num_colors = len(colors)
        if num_colors <= 4:
            return False
        if num_colors >= 10:
            return True
        return path_count >= 5
    except Exception:
        return path_count >= 8


# -----------------
# Page processing
# -----------------
def _process_page(page, doc, page_num: int, doc_id: str, tolerance: int = DEFAULT_TOL):
    blocks = []
    page_no = page_num + 1
    page_rect = page.rect

    exclusions: list["fitz.Rect"] = []

    # ── 1) Tables ──
    table_rects: list["fitz.Rect"] = []
    if hasattr(page, "find_tables"):
        try:
            for tbl in page.find_tables().tables:
                bbox = tbl.bbox
                r = fitz.Rect(bbox) if isinstance(bbox, (list, tuple)) else fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
                if r.width < 20 or r.height < 20:
                    continue
                table_rects.append(r)
                exclusions.append(r)

                md = _table_to_markdown(tbl)
                if md and len(md.strip()) >= MIN_CHUNK_CHARS:
                    blocks.append({"type": "table", "text": md, "y": r.y0, "page": page_no})
        except Exception as e:
            logger.debug(f"Table extraction p{page_no}: {e}")

    # ── 2) Raster images ──
    raster_rects: list["fitz.Rect"] = []
    for img_i, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]
        try:
            bi = doc.extract_image(xref)
        except Exception:
            continue

        w, h = bi.get("width", 0), bi.get("height", 0)
        if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
            continue

        # A single image xref can appear multiple times; preserve each placement as its own block.
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []

        if not rects:
            # still save the image once, but we can't position it well
            rects = [fitz.Rect(0, 0, 0, 0)]

        # Save the image bytes once (same xref); re-use filepath for each rect
        # Convert non-standard formats (jpx, jp2, etc) to PNG for compatibility
        ext = bi["ext"]
        if ext.lower() in ("jpx", "jp2", "j2k", "jpc"):
            try:
                pil_img = Image.open(io.BytesIO(bi["image"]))
                pil_img.load()
                pil_img = pil_img.convert("RGB")
                ext = "png"
                fname = f"{os.path.splitext(doc_id)[0]}_p{page_no}_img{img_i}.{ext}"
                fpath = IMAGES_DIR / fname
                pil_img.save(str(fpath), "PNG")
            except Exception:
                continue
        else:
            fname = f"{os.path.splitext(doc_id)[0]}_p{page_no}_img{img_i}.{ext}"
            fpath = IMAGES_DIR / fname
            try:
                with open(fpath, "wb") as f:
                    f.write(bi["image"])
            except Exception:
                continue

        for r_i, ir in enumerate(rects):
            r = fitz.Rect(ir)
            if r.width < 10 or r.height < 10:
                continue
            if any(_overlap_frac(r, e) > 0.5 for e in exclusions):
                continue

            raster_rects.append(r)
            exclusions.append(r)

            blocks.append({
                "type": "image",
                "filepath": str(fpath),
                "y": r.y0,
                "page": page_no,
                "dimensions": f"{w}x{h}",
                "image_kind": "raster",
            })

    # ── 3) Vector diagrams (cluster drawings -> rasterize crop -> treat as image) ──
    vector_rects: list["fitz.Rect"] = []
    vector_text_chunks = []  # text extracted from inside diagram regions
    if hasattr(page, "get_drawings") and hasattr(page, "cluster_drawings"):
        try:
            all_drawings = page.get_drawings()
            drawings = _filter_decorations(all_drawings, page_rect.width)

            if 0 < len(drawings) <= GRAPHICS_LIMIT:
                try:
                    clusters = page.cluster_drawings(drawings=drawings, x_tolerance=tolerance, y_tolerance=tolerance)
                except TypeError:
                    clusters = page.cluster_drawings()

                for vec_i, bbox in enumerate(clusters):
                    cluster_r = fitz.Rect(bbox)

                    if cluster_r.width < 15 or cluster_r.height < 15:
                        continue
                    if (cluster_r.width * cluster_r.height) < MIN_VECTOR_AREA:
                        continue
                    if _is_line(cluster_r):
                        continue
                    if cluster_r.width > page_rect.width * 0.9 and cluster_r.height > page_rect.height * 0.9:
                        continue
                    if any(_overlap_frac(cluster_r, e) > 0.5 for e in exclusions):
                        continue

                    r = _trim_to_paths(drawings, cluster_r)
                    r = _shrink_past_text(page, r)

                    if r.width < 15 or r.height < 15:
                        continue
                    if (r.width * r.height) < MIN_VECTOR_AREA:
                        continue
                    if any(_overlap_frac(r, e) > 0.5 for e in exclusions):
                        continue

                    path_count = _count_paths_in_rect(drawings, r)
                    if not _is_diagram(page, r, drawings, path_count):
                        continue

                    # Extract text from inside the diagram region so it's not lost
                    diagram_text = ""
                    try:
                        diagram_text = page.get_text("text", clip=r).strip()
                    except Exception:
                        pass

                    # Render crop at higher res for downstream CLIP
                    try:
                        mat = fitz.Matrix(2.0, 2.0)
                        pix = page.get_pixmap(clip=r, matrix=mat, alpha=False)
                        if pix.width < MIN_IMAGE_DIM or pix.height < MIN_IMAGE_DIM:
                            continue

                        fname = f"{os.path.splitext(doc_id)[0]}_p{page_no}_vec{vec_i}.png"
                        fpath = IMAGES_DIR / fname
                        pix.save(str(fpath))
                    except Exception:
                        continue

                    vector_rects.append(r)
                    exclusions.append(r)

                    blocks.append({
                        "type": "image",
                        "filepath": str(fpath),
                        "y": r.y0,
                        "page": page_no,
                        "dimensions": f"{pix.width}x{pix.height}",
                        "image_kind": "vector",
                        "vector_paths": path_count,
                    })

                    # Keep the text from inside the diagram as a searchable chunk
                    if diagram_text and len(diagram_text) >= MIN_CHUNK_CHARS:
                        vector_text_chunks.append({
                            "text": diagram_text,
                            "y": r.y0,
                            "page": page_no,
                        })

        except Exception as e:
            logger.debug(f"Vector diagram extraction p{page_no}: {e}")

    # ── 4) Text blocks (skip exclusion zones) ──
    try:
        td = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for blk in td.get("blocks", []):
            bbox = blk.get("bbox")
            if not bbox or blk.get("type") != 0:
                continue

            r = fitz.Rect(bbox)
            if r.width < 8 or r.height < 8:
                continue

            # Skip text that is inside tables/diagrams/images
            if any(_overlap_frac(r, e) > 0.7 for e in exclusions):
                continue

            t = _block_text(blk)
            if t and len(t.strip()) > 3:
                blocks.append({"type": "text", "text": t.strip(), "y": r.y0, "page": page_no})
    except Exception as e:
        logger.debug(f"Text extraction p{page_no}: {e}")

    # Add text extracted from inside vector diagram regions
    for vtc in vector_text_chunks:
        blocks.append({"type": "text", "text": vtc["text"], "y": vtc["y"], "page": vtc["page"]})

    # Keep in reading order
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
                "id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "page": page_no,
                "text": merged.strip(),
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
                            "id": str(uuid.uuid4()),
                            "doc_id": doc_id,
                            "page": page_no,
                            "text": sc.strip(),
                            "chunk_type": "text",
                        })
            else:
                buf.append(t)
                buf_len += len(t)
        elif blk["type"] == "table":
            flush()
            text_chunks.append({
                "id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "page": page_no,
                "text": blk["text"],
                "chunk_type": "table",
            })
        elif blk["type"] == "image":
            caption = _find_caption(blocks, i)
            images_meta.append({
                "id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "page": page_no,
                "filepath": blk["filepath"],
                "context": caption or f"Image on page {page_no} of {doc_id}",
                "caption": caption,
                "dimensions": blk.get("dimensions", ""),
                # extra fields are safe; rest of system can ignore them
                "image_kind": blk.get("image_kind", "raster"),
                "vector_paths": blk.get("vector_paths", 0),
            })
    flush()

    return text_chunks, images_meta


# -----------------
# PDFProcessor
# -----------------
class PDFProcessor:
    def __init__(self):
        try:
            self.clip_model = AutoModel.from_pretrained(IMAGE_EMBED_MODEL).eval()
            self.clip_processor = AutoProcessor.from_pretrained(IMAGE_EMBED_MODEL)
        except Exception as e:
            logger.error(f"SigLIP2 load failed: {e}. Run 'python download_models.py'.")
            raise

    def process_pdf(self, pdf_path: str):
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
        nv = sum(1 for m in all_images if m.get("image_kind") == "vector")
        logger.info(f"Done: {len(all_chunks)} chunks ({nt} text, {ntb} table), {len(all_images)} images ({nv} vector)")
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
        return np.array(embeddings).astype("float32")

    def embed_images(self, images_metadata):
        images, valid_idx = [], []
        for i, m in enumerate(images_metadata):
            try:
                img = Image.open(m["filepath"])
                if img.size[0] < MIN_IMAGE_DIM or img.size[1] < MIN_IMAGE_DIM:
                    continue
                # Force load and convert to RGB to catch broken files early
                img.load()
                img = img.convert("RGB")
                images.append(img)
                valid_idx.append(i)
            except Exception as e:
                logger.warning(f"Skipping bad image {m['filepath']}: {e}")

        if not images:
            return np.array([]).astype("float32"), []

        inputs = self.clip_processor(images=images, return_tensors="pt")
        import torch
        with torch.no_grad():
            feats = self.clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.detach().numpy().astype("float32"), [images_metadata[i] for i in valid_idx]


# -----------------
# VectorStore
# -----------------
class VectorStore:
    def __init__(self):
        self.text_index = None
        self.image_index = None
        self.text_metadata = []
        self.image_metadata = []

    def build_index(self, text_chunks, images_metadata, text_embeddings, image_embeddings):
        if len(text_chunks) > 0:
            d = text_embeddings.shape[1]
            if self.text_index is None:
                self.text_index = faiss.IndexFlatL2(d)
                self.text_metadata = []
            self.text_index.add(text_embeddings)
            self.text_metadata.extend(text_chunks)

        if len(images_metadata) > 0 and len(image_embeddings) > 0:
            d = image_embeddings.shape[1]
            if self.image_index is None:
                self.image_index = faiss.IndexFlatL2(d)
                self.image_metadata = []
            self.image_index.add(image_embeddings)
            self.image_metadata.extend(images_metadata)

    def get_indexed_doc_ids(self) -> list[str]:
        docs = set()
        for c in self.text_metadata:
            if isinstance(c, dict) and c.get("doc_id"):
                docs.add(c["doc_id"])
        for m in self.image_metadata:
            if isinstance(m, dict) and m.get("doc_id"):
                docs.add(m["doc_id"])
        return sorted(docs)

    def remove_doc(self, doc_id: str):
        # Delete image files for this doc
        for m in self.image_metadata:
            if isinstance(m, dict) and m.get("doc_id") == doc_id:
                fp = m.get("filepath", "")
                if fp and os.path.exists(fp):
                    try:
                        os.remove(fp)
                    except Exception:
                        pass

        # Rebuild text index without the removed doc
        keep_text_idx = [i for i, c in enumerate(self.text_metadata) if c.get("doc_id") != doc_id]
        if keep_text_idx and self.text_index:
            d = self.text_index.d
            vecs = np.array([self.text_index.reconstruct(i) for i in keep_text_idx], dtype="float32")
            self.text_metadata = [self.text_metadata[i] for i in keep_text_idx]
            self.text_index = faiss.IndexFlatL2(d)
            self.text_index.add(vecs)
        else:
            self.text_metadata = [c for c in self.text_metadata if c.get("doc_id") != doc_id]
            if not self.text_metadata:
                self.text_index = None

        # Rebuild image index without the removed doc
        keep_img_idx = [i for i, m in enumerate(self.image_metadata) if m.get("doc_id") != doc_id]
        if keep_img_idx and self.image_index:
            d = self.image_index.d
            vecs = np.array([self.image_index.reconstruct(i) for i in keep_img_idx], dtype="float32")
            self.image_metadata = [self.image_metadata[i] for i in keep_img_idx]
            self.image_index = faiss.IndexFlatL2(d)
            self.image_index.add(vecs)
        else:
            self.image_metadata = [m for m in self.image_metadata if m.get("doc_id") != doc_id]
            if not self.image_metadata:
                self.image_index = None

        self.save()
        logger.info(f"Removed doc '{doc_id}' from index.")

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