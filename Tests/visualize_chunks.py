"""
Usage:
    python visualize_chunks.py input.pdf
"""

import fitz
import sys
import os
import re

MIN_IMAGE_DIM = 50
MIN_CHUNK_CHARS = 60
IMAGE_IN_TABLE_THRESHOLD = 0.40  # If image covers >40% of "table" area, it's not a real table

COLOR_TEXT  = (1, 0, 0)       # Red
COLOR_TABLE = (0, 0, 1)       # Blue
COLOR_IMAGE = (0, 0.7, 0)     # Green
COLOR_IMG_CAPTION = (0, 0.5, 0.3)  # Teal-green for image+caption

FILL_TEXT  = (1, 0.85, 0.85)
FILL_TABLE = (0.85, 0.85, 1)
FILL_IMAGE = (0.85, 1, 0.85)
FILL_IMG_CAPTION = (0.85, 1, 0.92)

LABEL_FONT_SIZE = 7
BOX_WIDTH = 1.5
BOX_WIDTH_TABLE = 2.0
BOX_WIDTH_IMAGE = 2.0


def _block_text(blk: dict) -> str:
    lines = []
    for line in blk.get("lines", []):
        spans = "".join(s.get("text", "") for s in line.get("spans", []))
        lines.append(spans)
    return "\n".join(lines)


def _rect_area(r):
    w = max(0, r.x1 - r.x0)
    h = max(0, r.y1 - r.y0)
    return w * h


def _get_image_rects(page, doc):
    image_rects = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            bi = doc.extract_image(xref)
        except Exception:
            continue
        w, h = bi.get("width", 0), bi.get("height", 0)
        if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
            continue
        try:
            rects = page.get_image_rects(xref)
            if rects:
                image_rects.append((rects[0], f"{w}x{h}", w, h))
        except Exception:
            pass
    return image_rects


def _get_table_rects(page):
    tables = []
    if not hasattr(page, "find_tables"):
        return tables
    try:
        for tbl in page.find_tables().tables:
            bbox = tbl.bbox
            r = fitz.Rect(bbox) if isinstance(bbox, (list, tuple)) else fitz.Rect(
                bbox.x0, bbox.y0, bbox.x1, bbox.y1
            )
            row_count = 0
            try:
                row_count = len(tbl.extract())
            except Exception:
                pass
            tables.append((r, row_count))
    except Exception:
        pass
    return tables


def _filter_image_tables(table_rects, image_rects):
    real_tables = []
    image_tables = []

    for table_rect, row_count in table_rects:
        table_area = _rect_area(table_rect)
        if table_area <= 0:
            real_tables.append((table_rect, row_count))
            continue

        # Check if any image significantly overlaps this table
        best_overlap_ratio = 0
        best_img = None
        best_dims = None

        for img_rect, dims, pw, ph in image_rects:
            # Compute intersection area
            intersection = fitz.Rect(img_rect) & fitz.Rect(table_rect)
            if intersection.is_empty:
                continue
            overlap_area = _rect_area(intersection)
            ratio = overlap_area / table_area

            if ratio > best_overlap_ratio:
                best_overlap_ratio = ratio
                best_img = img_rect
                best_dims = dims

        if best_overlap_ratio >= IMAGE_IN_TABLE_THRESHOLD:
            # This "table" is really an image+caption
            image_tables.append((table_rect, best_img, best_dims))
        else:
            real_tables.append((table_rect, row_count))

    return real_tables, image_tables


def _get_text_block_rects(page, table_rects):
    text_blocks = []
    raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    for blk in raw:
        bbox = blk.get("bbox")
        if not bbox or blk.get("type") != 0:
            continue
        r = fitz.Rect(bbox)
        if any(r.intersects(tr) for tr, _ in table_rects):
            continue
        t = _block_text(blk)
        if t and len(t.strip()) > 3:
            text_blocks.append((r, len(t.strip())))
    return text_blocks


def _find_caption_text(page, image_rect):
    cap_re = re.compile(r'(?i)^(?:figure|fig\.?|diagram|illustration|image|photo|view)\s*\d*')
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

    candidates = []
    for blk in blocks:
        if blk.get("type") != 0:
            continue
        bbox = blk.get("bbox")
        if not bbox:
            continue
        r = fitz.Rect(bbox)
        t = _block_text(blk).strip()
        if not t or len(t) < 3:
            continue

        # Check if this text block is near the image (below or above, within 30px)
        # and horizontally overlapping
        h_overlap = r.x0 < image_rect.x1 and r.x1 > image_rect.x0

        # Below the image
        if h_overlap and 0 <= (r.y0 - image_rect.y1) < 30:
            candidates.append(("below", abs(r.y0 - image_rect.y1), t))
        # Above the image
        elif h_overlap and 0 <= (image_rect.y0 - r.y1) < 30:
            candidates.append(("above", abs(image_rect.y0 - r.y1), t))
        # Inside the table rect (caption might be inside the bounding area)
        elif h_overlap and r.y0 >= image_rect.y1 - 5:
            candidates.append(("inside", abs(r.y0 - image_rect.y1), t))

    # Prefer below, then inside, then above, closest first
    priority = {"below": 0, "inside": 1, "above": 2}
    candidates.sort(key=lambda c: (priority.get(c[0], 3), c[1]))

    for _, _, text in candidates:
        if cap_re.match(text):
            return text[:200]
        if len(text) < 200:
            return text

    return ""


def _draw_legend(page):
    pw = page.rect.width
    legend_x = pw - 175
    legend_y = 10
    legend_w = 165
    legend_h = 80

    # Background
    legend_rect = fitz.Rect(legend_x, legend_y, legend_x + legend_w, legend_y + legend_h)
    page.draw_rect(legend_rect, color=(0.3, 0.3, 0.3), fill=(1, 1, 1), width=0.8)

    items = [
        (COLOR_TEXT,        "Text Chunks"),
        (COLOR_TABLE,       "Table Chunks"),
        (COLOR_IMAGE,       "Image Chunks"),
        (COLOR_IMG_CAPTION, "Image+Caption"),
    ]
    y = legend_y + 12
    for color, label in items:
        swatch = fitz.Rect(legend_x + 8, y - 5, legend_x + 22, y + 5)
        page.draw_rect(swatch, color=color, fill=color, width=0)
        page.insert_text(
            fitz.Point(legend_x + 28, y + 3),
            label,
            fontsize=LABEL_FONT_SIZE + 1,
            color=(0.1, 0.1, 0.1),
        )
        y += 16


def _draw_label(page, rect, label, color):
    text_w = len(label) * (LABEL_FONT_SIZE * 0.55)
    text_h = LABEL_FONT_SIZE + 4

    tag_rect = fitz.Rect(
        rect.x0, rect.y0 - text_h - 1,
        rect.x0 + text_w + 6, rect.y0 - 1
    )
    # Clamp to page bounds
    if tag_rect.y0 < 0:
        tag_rect = fitz.Rect(rect.x0, rect.y0, rect.x0 + text_w + 6, rect.y0 + text_h)

    page.draw_rect(tag_rect, color=color, fill=color, width=0)
    page.insert_text(
        fitz.Point(tag_rect.x0 + 3, tag_rect.y1 - 3),
        label,
        fontsize=LABEL_FONT_SIZE,
        color=(1, 1, 1),
    )


def visualize_chunks(pdf_in, pdf_out):
    doc = fitz.open(pdf_in)

    total_text = 0
    total_tables = 0
    total_images = 0
    total_img_caption = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_no = page_num + 1

        raw_table_rects = _get_table_rects(page)
        image_rects = _get_image_rects(page, doc)

        real_tables, image_tables = _filter_image_tables(raw_table_rects, image_rects)

        text_rects = _get_text_block_rects(page, real_tables)

        image_table_keys = set()
        for _, img_r, _ in image_tables:
            image_table_keys.add((round(img_r.x0, 1), round(img_r.y0, 1),
                                  round(img_r.x1, 1), round(img_r.y1, 1)))

        standalone_images = []
        for img_rect, dims, pw, ph in image_rects:
            key = (round(img_rect.x0, 1), round(img_rect.y0, 1),
                   round(img_rect.x1, 1), round(img_rect.y1, 1))
            if key not in image_table_keys:
                standalone_images.append((img_rect, dims))

        page_img_caption = 0

        for rect, char_count in text_rects:
            shape = page.new_shape()
            shape.draw_rect(rect)
            shape.finish(color=COLOR_TEXT, fill=FILL_TEXT, fill_opacity=0.15, width=BOX_WIDTH)
            shape.commit()
            _draw_label(page, rect, f"TEXT ({char_count} chars)", COLOR_TEXT)
            total_text += 1

        for rect, row_count in real_tables:
            shape = page.new_shape()
            shape.draw_rect(rect)
            shape.finish(color=COLOR_TABLE, fill=FILL_TABLE, fill_opacity=0.15, width=BOX_WIDTH_TABLE)
            shape.commit()
            label = f"TABLE ({row_count} rows)" if row_count else "TABLE"
            _draw_label(page, rect, label, COLOR_TABLE)
            total_tables += 1

        for table_rect, img_rect, dims in image_tables:
            shape = page.new_shape()
            shape.draw_rect(table_rect)
            shape.finish(color=COLOR_IMG_CAPTION, fill=FILL_IMG_CAPTION,
                         fill_opacity=0.15, width=BOX_WIDTH_IMAGE)
            shape.commit()

            caption = _find_caption_text(page, img_rect)
            if caption:
                cap_short = caption[:40] + "..." if len(caption) > 40 else caption
                label = f"IMG+CAP ({dims}) \"{cap_short}\""
            else:
                label = f"IMG+CAP ({dims})"
            _draw_label(page, table_rect, label, COLOR_IMG_CAPTION)
            total_img_caption += 1
            page_img_caption += 1

        for rect, dims in standalone_images:
            shape = page.new_shape()
            shape.draw_rect(rect)
            shape.finish(color=COLOR_IMAGE, fill=FILL_IMAGE, fill_opacity=0.15, width=BOX_WIDTH_IMAGE)
            shape.commit()
            _draw_label(page, rect, f"IMAGE ({dims})", COLOR_IMAGE)
            total_images += 1

        _draw_legend(page)

        reclassified = len(raw_table_rects) - len(real_tables)
        note = f" ({reclassified} table→img+cap)" if reclassified > 0 else ""
        print(
            f"  Page {page_no:3d}: "
            f"{len(text_rects):2d} text, "
            f"{len(real_tables):2d} table, "
            f"{len(standalone_images):2d} image, "
            f"{page_img_caption:2d} img+cap"
            f"{note}"
        )

    doc.save(pdf_out)
    doc.close()

    print(f"\n{'='*60}")
    print(f"  Text:         {total_text}")
    print(f"  Tables:       {total_tables}")
    print(f"  Images:       {total_images}")
    print(f"  Img+Caption:  {total_img_caption}")
    print(f"  Saved:        {pdf_out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_chunks.py input.pdf [output.pdf]")
        sys.exit(1)

    pdf_in = sys.argv[1]
    if len(sys.argv) >= 3:
        pdf_out = sys.argv[2]
    else:
        base, ext = os.path.splitext(pdf_in)
        pdf_out = f"{base}_chunked{ext}"

    print(f"Visualizing chunks: {pdf_in} → {pdf_out}\n")
    visualize_chunks(pdf_in, pdf_out)