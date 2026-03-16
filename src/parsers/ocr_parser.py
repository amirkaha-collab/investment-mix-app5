"""
OCR PDF Parser – for scanned / image-based PDFs.

Pipeline:
  1. Render each PDF page to a PIL image via pymupdf (no poppler needed)
  2. Run Tesseract OCR (Hebrew + English) to get word-level bounding boxes
  3. Group words into rows by Y-coordinate proximity
  4. Cluster row words into columns by X-coordinate
  5. Detect the header row (contains keywords like שם נייר / שווי / value)
  6. Build a structured DataFrame

Graceful degradation:
  - If pytesseract is not installed → raise ImportError (caught by PDFParser)
  - If tesseract binary is missing → raise with clear message
  - If Hebrew lang data missing → fall back to eng only
  - If page rendering fails → skip that page with a warning

This parser is always tried LAST, only when text-based strategies failed.
It is intentionally separated so it can be replaced by a Vision API later.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .base import BaseParser, ParseResult

logger = logging.getLogger(__name__)

# OCR config: Hebrew + English, PSM 6 = assume uniform block of text
_TESS_LANG_HEB_ENG = "heb+eng"
_TESS_LANG_ENG     = "eng"
_TESS_CONFIG       = "--psm 6"

# Row reconstruction tolerances (pixels at 200 DPI)
_ROW_Y_TOLERANCE   = 12   # words within 12px vertically → same row
_COL_X_TOLERANCE   = 40   # column cluster tolerance

# Keywords that signal a header row
_HEADER_KEYWORDS = [
    "שם", "נייר", "שווי", "מטבע", "כמות", "סוג",
    "isin", "ticker", "security", "value", "currency", "quantity", "type",
]

# Keywords that signal a data row (at least one numeric value)
_NUMERIC_PATTERN = r"[\d,\.₪\$]{3,}"


class OCRPDFParser(BaseParser):
    """
    OCR-based parser for image-based / scanned PDF statements.
    Requires: pytesseract, pymupdf, Pillow.
    System dependency: tesseract-ocr (+ heb language data recommended).
    """

    RENDER_DPI = 200    # higher = better quality but slower

    def parse(self, source: Union[Path, bytes, io.BytesIO]) -> ParseResult:
        warnings: list[str] = []
        errors:   list[str] = []

        # Check dependencies
        try:
            import pytesseract
            from PIL import Image
        except ImportError as e:
            return ParseResult(
                pd.DataFrame(),
                errors=[
                    f"pytesseract/Pillow not installed: {e}. "
                    "OCR is not available in this environment."
                ],
                parse_method="ocr_unavailable",
            )

        # Detect available OCR language
        tess_lang = self._get_tess_lang(pytesseract)
        logger.info(f"OCR parser using language: {tess_lang}")

        buf = self._to_bytesio(source)
        buf.seek(0)

        try:
            import fitz
            doc = fitz.open(stream=buf.read(), filetype="pdf")
        except Exception as e:
            return ParseResult(
                pd.DataFrame(),
                errors=[f"Cannot open PDF for OCR rendering: {e}"],
                parse_method="ocr_render_failed",
            )

        all_dfs: list[pd.DataFrame] = []
        mat = fitz.Matrix(self.RENDER_DPI / 72, self.RENDER_DPI / 72)

        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                pix  = page.get_pixmap(matrix=mat, alpha=False)
                img  = self._pixmap_to_pil(pix)

                df = self._ocr_page_to_dataframe(img, pytesseract, tess_lang)
                if df is not None and not df.empty:
                    all_dfs.append(df)
                    logger.info(f"OCR page {page_num+1}: {len(df)} rows extracted")
            except Exception as exc:
                warnings.append(f"OCR failed on page {page_num+1}: {exc}")

        if not all_dfs:
            return ParseResult(
                pd.DataFrame(),
                warnings=warnings,
                errors=["OCR extraction produced no usable tables. "
                        "The document may be in an unsupported format."],
                parse_method="ocr_no_output",
            )

        best = max(all_dfs, key=len)
        return ParseResult(
            primary_df=best,
            raw_tables=all_dfs,
            parse_method=f"ocr_tesseract_{tess_lang}",
            warnings=warnings + [
                f"OCR extraction used (Tesseract {tess_lang}). "
                "Please review and correct holdings in the confirmation step."
            ],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Core OCR → DataFrame logic
    # ──────────────────────────────────────────────────────────────────────────

    def _ocr_page_to_dataframe(
        self, img, pytesseract, lang: str
    ) -> pd.DataFrame | None:
        """
        Run OCR on one page image and reconstruct a holdings table.
        Returns a DataFrame or None if no table-like content detected.
        """
        # Get word-level bounding boxes
        try:
            data = pytesseract.image_to_data(
                img,
                lang=lang,
                config=_TESS_CONFIG,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            logger.warning(f"pytesseract.image_to_data failed: {exc}")
            return None

        # Filter to confident, non-empty words
        words = []
        for i, text in enumerate(data["text"]):
            text = str(text).strip()
            conf = int(data["conf"][i])
            if not text or conf < 20:
                continue
            words.append({
                "text": text,
                "x":    data["left"][i],
                "y":    data["top"][i],
                "w":    data["width"][i],
                "h":    data["height"][i],
                "conf": conf,
            })

        if not words:
            return None

        # Group words into rows by Y proximity
        rows = self._group_into_rows(words)
        if len(rows) < 2:
            return None

        # Find header row
        header_idx = self._find_header_row(rows)
        if header_idx is None:
            # No header found – use first row
            header_idx = 0

        # Build DataFrame from rows relative to header
        df = self._rows_to_dataframe(rows, header_idx)
        return df if df is not None and not df.empty else None

    def _group_into_rows(self, words: list[dict]) -> list[list[dict]]:
        """Group words into horizontal rows by Y-coordinate proximity."""
        if not words:
            return []

        sorted_words = sorted(words, key=lambda w: (w["y"], w["x"]))
        rows: list[list[dict]] = []
        current_row: list[dict] = [sorted_words[0]]
        current_y = sorted_words[0]["y"]

        for word in sorted_words[1:]:
            if abs(word["y"] - current_y) <= _ROW_Y_TOLERANCE:
                current_row.append(word)
            else:
                rows.append(sorted(current_row, key=lambda w: w["x"]))
                current_row = [word]
                current_y = word["y"]

        if current_row:
            rows.append(sorted(current_row, key=lambda w: w["x"]))

        return rows

    def _find_header_row(self, rows: list[list[dict]]) -> int | None:
        """Find the index of the row that looks like a table header."""
        for i, row in enumerate(rows[:20]):  # search first 20 rows
            row_text = " ".join(w["text"] for w in row).lower()
            hits = sum(1 for kw in _HEADER_KEYWORDS if kw in row_text)
            if hits >= 2:
                return i
        return None

    def _rows_to_dataframe(
        self, rows: list[list[dict]], header_idx: int
    ) -> pd.DataFrame | None:
        """
        Convert grouped rows (after header) into a DataFrame.
        Uses column X positions from the header row to assign values.
        """
        import re

        header_row = rows[header_idx]
        data_rows  = rows[header_idx + 1:]

        if not data_rows:
            return None

        # Header columns: (x_center, column_name)
        headers = [(w["x"] + w["w"] // 2, w["text"]) for w in header_row]
        col_names = [h[1] for h in headers]
        col_xs    = [h[0] for h in headers]

        result_rows: list[dict] = []
        for row in data_rows:
            row_text = " ".join(w["text"] for w in row)
            # Skip rows with no numeric content (likely separators / titles)
            if not re.search(_NUMERIC_PATTERN, row_text) and len(row) < 2:
                continue

            # Assign each word to nearest column
            assigned: dict[str, list[str]] = {c: [] for c in col_names}
            for word in row:
                word_x = word["x"] + word["w"] // 2
                nearest_col = min(
                    range(len(col_xs)),
                    key=lambda i: abs(col_xs[i] - word_x)
                )
                assigned[col_names[nearest_col]].append(word["text"])

            result_rows.append({col: " ".join(vals) for col, vals in assigned.items()})

        if not result_rows:
            return None

        df = pd.DataFrame(result_rows)
        df = df.dropna(how="all")
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _pixmap_to_pil(pix):
        """Convert a pymupdf Pixmap to a PIL Image."""
        from PIL import Image
        mode = "RGBA" if pix.alpha else "RGB"
        img  = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        return img.convert("RGB")

    @staticmethod
    def _get_tess_lang(pytesseract) -> str:
        """Return the best available Tesseract language string."""
        try:
            langs = pytesseract.get_languages()
            if "heb" in langs:
                return _TESS_LANG_HEB_ENG
        except Exception:
            pass
        return _TESS_LANG_ENG
