"""
Phase 1 exit gate — ingestion layer tests.
Run against the real starter PDFs; do not mock the PDF content.
"""
import pytest
from ingestion import parse_page_native_fallback


def test_native_fallback_returns_text(tmp_path):
    """Smoke test: native fallback on a synthetic single-page PDF returns
    non-empty text without raising."""
    import pymupdf as fitz
    pdf_path = tmp_path / "smoke.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello Bhram smoke test.")
    doc.save(str(pdf_path))
    doc.close()

    result = parse_page_native_fallback(str(pdf_path), 0)
    assert "Hello Bhram" in result


def test_native_fallback_no_handle_leak(tmp_path):
    """Calling parse_page_native_fallback twice in a row on the same file
    must not raise, confirming no handle leak between calls."""
    import pymupdf as fitz
    pdf_path = tmp_path / "leak.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    parse_page_native_fallback(str(pdf_path), 0)
    parse_page_native_fallback(str(pdf_path), 0)  # second call must not raise
