"""
PDF Parser – full extraction pipeline.

Decision flow:
  ┌─────────────────────────────────────────────────────────┐
  │  1. Detect: text-based or image-based?                  │
  │     (count selectable words via pymupdf)                │
  ├─────────────────────────────────────────────────────────┤
  │  TEXT-BASED                    IMAGE-BASED              │
  │  a) pdfplumber table extract   OCRParser (Tesseract)    │
  │  b) pymupdf structured recon   ↓                        │
  │  c) text-line regex fallback   text-line regex fallback │
  └─────────────────────────────────────────────────────────┘

User-facing messages:
  - Never show internal module names (camelot, pdfplumber, pymupdf)
  - Never show "No module named X"
  - Show one clear message about what happened and what to do
  - Debug panel in sidebar shows full technical detail (opt-in)
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Union

import pandas as pd

from .base import BaseParser, ParseResult
from .pdf_detector import detect_pdf_type
from .text_line_parser import parse_text_lines_to_dataframe

logger = logging.getLogger(__name__)

_HOLDINGS_KEYWORDS = [
    "שווי", "market value", "שם נייר", "security", "isin",
    "אחזקות", "holdings", "balance", "יתרה", "סכום",
    "שווי שוק", "שווי נוכחי", "value", "שם", "נייר",
    "quantity", "כמות", "מטבע", "currency",
]
_MIN_TABLE_ROWS = 2


class PDFParser(BaseParser):
    """
    Main PDF parser.  Routes automatically to the best available strategy.
    Yields zero silent failures — all paths produce either data or a clear error.
    """

    def parse(self, source: Union[Path, bytes, io.BytesIO]) -> ParseResult:
        buf = self._to_bytesio(source)

        # ── 1. Detect PDF type ────────────────────────────────────────────────
        type_info = detect_pdf_type(buf)
        buf.seek(0)
        logger.info(f"PDF detection result: {type_info}")

        # ── 2. Route to correct pipeline ──────────────────────────────────────
        if type_info.is_image_based:
            logger.info("Image-based PDF detected → OCR pipeline")
            result = self._run_ocr(buf)
            if result.success:
                result.warnings = [
                    "המסמך זוהה כ-PDF תמונתי/סרוק. "
                    "הופעל OCR לשחזור הטבלה. "
                    "אנא בדוק ותקן את הנתונים בשלב הבא."
                ]
                return result
            # OCR failed → try text-line on raw text
            buf.seek(0)
            result = self._run_text_line_fallback(buf, reason="image-based OCR failed")
            return result

        # Text-based path
        buf.seek(0)
        result = self._try_pdfplumber(buf)
        if result.success:
            return result

        buf.seek(0)
        result = self._try_pymupdf_structured(buf)
        if result.success:
            return result

        # Last resort: text-line regex on raw pymupdf text
        buf.seek(0)
        result = self._run_text_line_fallback(buf, reason="table extraction failed")
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Strategy A: pdfplumber
    # ──────────────────────────────────────────────────────────────────────────

    def _try_pdfplumber(self, buf: io.BytesIO) -> ParseResult:
        try:
            import pdfplumber
        except ImportError:
            logger.debug("pdfplumber not available")
            return ParseResult(pd.DataFrame(), parse_method="pdfplumber_unavailable")

        try:
            buf.seek(0)
            all_tables: list[pd.DataFrame] = []

            with pdfplumber.open(buf) as pdf:
                for page in pdf.pages:
                    for settings in [
                        {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
                        {"vertical_strategy": "text",         "horizontal_strategy": "text"},
                        {},
                    ]:
                        try:
                            tables = page.extract_tables(table_settings=settings) or []
                            for raw in tables:
                                if not raw or len(raw) < _MIN_TABLE_ROWS:
                                    continue
                                df = pd.DataFrame(raw[1:], columns=raw[0])
                                df = self._clean_header(df)
                                if self._looks_like_holdings(df):
                                    all_tables.append(df)
                        except Exception:
                            continue

            if not all_tables:
                logger.debug("pdfplumber: no holdings tables found")
                return ParseResult(pd.DataFrame(), parse_method="pdfplumber_no_tables")

            best = max(all_tables, key=len)
            logger.info(f"pdfplumber: extracted {len(best)} rows")
            return ParseResult(primary_df=best, raw_tables=all_tables, parse_method="pdfplumber")

        except Exception as exc:
            logger.debug(f"pdfplumber error: {exc}")
            return ParseResult(pd.DataFrame(), parse_method="pdfplumber_error")

    # ──────────────────────────────────────────────────────────────────────────
    # Strategy B: pymupdf structured reconstruction
    # ──────────────────────────────────────────────────────────────────────────

    def _try_pymupdf_structured(self, buf: io.BytesIO) -> ParseResult:
        try:
            import fitz
        except ImportError:
            return ParseResult(pd.DataFrame(), parse_method="pymupdf_unavailable")

        try:
            buf.seek(0)
            doc = fitz.open(stream=buf.read(), filetype="pdf")
            all_dfs: list[pd.DataFrame] = []

            for page in doc:
                df = self._reconstruct_page_table(page)
                if df is not None and not df.empty:
                    all_dfs.append(df)

            if not all_dfs:
                logger.debug("pymupdf structured: no tables reconstructed")
                return ParseResult(pd.DataFrame(), parse_method="pymupdf_structured_no_tables")

            best = max(all_dfs, key=len)
            logger.info(f"pymupdf structured: extracted {len(best)} rows")
            return ParseResult(
                primary_df=best,
                raw_tables=all_dfs,
                parse_method="pymupdf_structured",
                warnings=["חולצו נתונים ממבנה טקסט. אנא בדוק ותקן בשלב הבא."],
            )
        except Exception as exc:
            logger.debug(f"pymupdf structured error: {exc}")
            return ParseResult(pd.DataFrame(), parse_method="pymupdf_structured_error")

    def _reconstruct_page_table(self, page) -> pd.DataFrame | None:
        """
        Reconstruct a table from pymupdf word bounding boxes.
        Groups words by Y (rows) then assigns to header columns by X proximity.
        """
        Y_TOL = 6
        words_raw = page.get_text("words")
        if not words_raw:
            return None

        # Group into rows by Y coordinate
        rows_dict: dict[int, list] = {}
        for w in words_raw:
            y_key = int(w[1] / Y_TOL) * Y_TOL
            rows_dict.setdefault(y_key, []).append(w)

        rows = [sorted(v, key=lambda w: w[0]) for _, v in sorted(rows_dict.items())]
        row_texts = [" ".join(w[4] for w in row) for row in rows]

        # Find header row
        header_idx = None
        for i, txt in enumerate(row_texts[:25]):
            hits = sum(1 for kw in _HOLDINGS_KEYWORDS if kw.lower() in txt.lower())
            if hits >= 2:
                header_idx = i
                break

        if header_idx is None:
            return None

        header_words = rows[header_idx]
        col_names = [w[4] for w in header_words]
        col_xs    = [(w[0] + w[2]) / 2 for w in header_words]

        data_rows: list[dict] = []
        for row in rows[header_idx + 1:]:
            if not row:
                continue
            row_text = " ".join(w[4] for w in row)
            if not re.search(r"[\d,\.]{3,}", row_text):
                continue

            assigned = {c: [] for c in col_names}
            for word in row:
                wx = (word[0] + word[2]) / 2
                nearest = min(range(len(col_xs)), key=lambda i: abs(col_xs[i] - wx))
                assigned[col_names[nearest]].append(word[4])

            data_rows.append({col: " ".join(vals) for col, vals in assigned.items()})

        if not data_rows:
            return None

        df = pd.DataFrame(data_rows)
        return df if not df.empty else None

    # ──────────────────────────────────────────────────────────────────────────
    # Strategy C: OCR (image-based PDFs)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_ocr(self, buf: io.BytesIO) -> ParseResult:
        from .ocr_parser import OCRPDFParser
        buf.seek(0)
        return OCRPDFParser().parse(buf)

    # ──────────────────────────────────────────────────────────────────────────
    # Strategy D: text-line regex fallback (always available, last resort)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_text_line_fallback(self, buf: io.BytesIO, reason: str = "") -> ParseResult:
        """
        Extract all text lines from the PDF and run the regex-based
        line parser to recover holdings.  Works even when table structure
        is completely absent.
        """
        try:
            import fitz
            buf.seek(0)
            doc = fitz.open(stream=buf.read(), filetype="pdf")
            lines: list[str] = []
            for page in doc:
                text = page.get_text("text")
                lines.extend(text.splitlines())
        except Exception as exc:
            return ParseResult(
                pd.DataFrame(),
                errors=[
                    "לא ניתן לחלץ טקסט מהקובץ. "
                    "ייתכן שמדובר בקובץ פגום. "
                    "אנא נסה CSV או XLSX."
                ],
                parse_method="text_fallback_failed",
            )

        df = parse_text_lines_to_dataframe(lines)

        if df is None or df.empty:
            return ParseResult(
                pd.DataFrame(),
                errors=[
                    "לא זוהתה טבלת אחזקות בקובץ זה. "
                    f"(סיבה: {reason})\n\n"
                    "💡 **המלצה:** הורד את הדוח מהבנק בפורמט XLSX/CSV ישירות, "
                    "או ייצא את טבלת האחזקות ל-Excel והעלה אותה."
                ],
                parse_method="text_fallback_no_data",
            )

        logger.info(f"text_line_fallback: extracted {len(df)} rows")
        return ParseResult(
            primary_df=df,
            parse_method="text_line_regex_fallback",
            warnings=[
                "⚠️ זוהתה טבלה חלקית בלבד באמצעות חילוץ טקסט. "
                "הנתונים עשויים להיות לא מדויקים. "
                "**חובה לבדוק ולתקן בשלב אישור האחזקות.**"
            ],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _looks_like_holdings(self, df: pd.DataFrame) -> bool:
        if df.empty or df.shape[1] < 2 or len(df) < _MIN_TABLE_ROWS:
            return False
        col_text = " ".join(str(c) for c in df.columns).lower()
        return any(kw.lower() in col_text for kw in _HOLDINGS_KEYWORDS)
