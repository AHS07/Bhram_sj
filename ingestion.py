"""
Page-level parsing. Vision-first with PyMuPDF native fallback.

Vision (page image → Markdown) is the primary path because it preserves
table structure, column alignment, and reading order far better than raw
text extraction on complex financial documents.

Native fallback is used when:
  - The vision call fails or times out
  - The vision output is suspiciously thin (< 20 chars)

The previous "native-first" optimisation was reverted: PyMuPDF frequently
returns thousands of raw characters on table-heavy pages, but with broken
columnar alignment. Trusting text length as a quality proxy silently
degraded table extraction on financial reports.
"""
import base64
import pymupdf as fitz  # PyMuPDF 1.28+
from llm_client import client, VISION_MODEL

PARSE_SYSTEM_PROMPT = (
    "You are a document parsing engine. Convert this page image into clean "
    "Markdown, preserving reading order. Render any tables as Markdown tables "
    "with headers correctly aligned to columns. Do not summarize or omit content."
)


def render_page_to_image_b64(pdf_path: str, page_number: int) -> str:
    with fitz.open(pdf_path) as doc:
        page = doc[page_number]
        pix = page.get_pixmap(dpi=150)  # 150dpi keeps payload under ~300KB per page
        return base64.b64encode(pix.tobytes("png")).decode("utf-8")


def parse_page_vision(pdf_path: str, page_number: int) -> str:
    image_b64 = render_page_to_image_b64(pdf_path, page_number)
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]},
        ],
        max_tokens=4000,
        timeout=60,  # never hang indefinitely — fallback will catch the timeout
    )
    return response.choices[0].message.content or ""


def _blocks_overlap(block_bbox, table_bbox, overlap_ratio: float = 0.5) -> bool:
    """True if a text block sits mostly inside a table's bounding box —
    used to drop native text that find_tables() already re-extracts cleanly,
    so we don't feed the LLM the same table twice (once garbled, once clean)."""
    bx0, by0, bx1, by1 = block_bbox
    tx0, ty0, tx1, ty1 = table_bbox
    ix0, iy0 = max(bx0, tx0), max(by0, ty0)
    ix1, iy1 = min(bx1, tx1), min(by1, ty1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    intersect_area = (ix1 - ix0) * (iy1 - iy0)
    block_area = max((bx1 - bx0) * (by1 - by0), 1e-6)
    return (intersect_area / block_area) >= overlap_ratio


def parse_page_native_fallback(pdf_path: str, page_number: int) -> str:
    """PyMuPDF native text + table-finder fallback. Text blocks that fall
    inside a detected table's bounding box are excluded from the raw text
    and replaced once by the clean markdown table, instead of appearing
    twice (raw + markdown)."""
    with fitz.open(pdf_path) as doc:
        page = doc[page_number]

        table_bboxes = []
        table_markdowns = []
        try:
            tables = page.find_tables()
            for t in tables.tables:
                table_bboxes.append(t.bbox)
                table_markdowns.append(t.to_markdown())
        except Exception:
            pass

        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        text_parts = []
        for b in blocks:
            bbox, block_text = b[:4], b[4]
            if any(_blocks_overlap(bbox, tb) for tb in table_bboxes):
                continue
            text_parts.append(block_text)

        text = "\n".join(text_parts)
        for md in table_markdowns:
            text += "\n\n" + md
        return text


def parse_page(pdf_path: str, page_number: int) -> str:
    """Vision-first parse with native fallback.

    Vision is always attempted first to preserve table fidelity.
    Native fallback fires when the vision call fails or returns thin output.
    """
    try:
        result = parse_page_vision(pdf_path, page_number)
        if len(result.strip()) < 20:  # suspiciously thin — fall back
            raise ValueError("vision output too short")
        return result
    except Exception:
        return parse_page_native_fallback(pdf_path, page_number)
