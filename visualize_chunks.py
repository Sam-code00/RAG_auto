"""
visualize_chunks.py - Draw colored boxes on PDF to show chunking

Red = Text chunks
Blue = Tables  
Green = Images

Usage: python visualize_chunks.py input.pdf output.pdf
"""

import fitz
import sys

def visualize_chunks(pdf_in, pdf_out):
    doc = fitz.open(pdf_in)
    
    for page_num, page in enumerate(doc):
        # Get all blocks
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            bbox = block.get("bbox")
            if not bbox:
                continue
            
            rect = fitz.Rect(bbox)
            
            if block.get("type") == 0:  # Text
                page.draw_rect(rect, color=(1, 0, 0), width=1)  # Red
            elif block.get("type") == 1:  # Image
                page.draw_rect(rect, color=(0, 0.7, 0), width=2)  # Green
        
        # Tables (using find_tables if available)
        if hasattr(page, "find_tables"):
            tables = page.find_tables()
            for table in tables.tables:
                bbox = table.bbox
                if isinstance(bbox, (list, tuple)):
                    rect = fitz.Rect(bbox)
                else:
                    rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
                page.draw_rect(rect, color=(0, 0, 1), width=2)  # Blue
    
    doc.save(pdf_out)
    doc.close()
    print(f"Saved: {pdf_out}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_chunks.py input.pdf output.pdf")
        sys.exit(1)
    
    visualize_chunks(sys.argv[1], sys.argv[2])