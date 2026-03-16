"""
PDF type detector.

Determines whether a PDF is:
  - text_based  : has selectable text → use pdfplumber / pymupdf text extraction
  - image_based : scanned/rendered image → must use OCR

Heuristic:
  Open first 3 pages with pymupdf.
  Count total word tokens.
  If fewer than MIN_WORDS_TEXT_BASED words across sample pages → image_based.
"""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

MIN_WORDS_TEXT_BASED = 50   # fewer words than this → probably image-based


class PDFTypeInfo:
    def __init__(self, is_image_based: bool, word_count: int, page_count: int, method_used: str):
        self.is_image_based = is_image_based
        self.word_count     = word_count
        self.page_count     = page_count
        self.method_used    = method_used

    def __repr__(self) -> str:
        kind = "IMAGE-BASED" if self.is_image_based else "TEXT-BASED"
        return f"<PDFTypeInfo {kind} words={self.word_count} pages={self.page_count}>"


def detect_pdf_type(buf: io.BytesIO) -> PDFTypeInfo:
    """
    Detect whether the PDF is text-based or image/scan-based.
    Returns PDFTypeInfo.
    """
    buf.seek(0)
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=buf.read(), filetype="pdf")
        page_count = len(doc)
        total_words = 0
        sample_pages = min(3, page_count)

        for i in range(sample_pages):
            page = doc[i]
            text = page.get_text("text")
            words = [w for w in text.split() if len(w) > 1]
            total_words += len(words)

        is_image = total_words < MIN_WORDS_TEXT_BASED
        logger.info(
            f"PDF detection: {total_words} words in first {sample_pages} pages "
            f"→ {'IMAGE-BASED' if is_image else 'TEXT-BASED'}"
        )
        return PDFTypeInfo(
            is_image_based=is_image,
            word_count=total_words,
            page_count=page_count,
            method_used="pymupdf_word_count",
        )
    except Exception as exc:
        logger.warning(f"PDF detection failed: {exc} – assuming text-based")
        return PDFTypeInfo(
            is_image_based=False,
            word_count=-1,
            page_count=0,
            method_used=f"failed:{exc}",
        )
