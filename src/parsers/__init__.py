"""
Parser factory – selects the correct parser based on file extension.
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseParser, ParseResult
from .pdf_parser import PDFParser
from .excel_csv_parser import ExcelParser, CSVParser
from .normalizer import HoldingsNormalizer


def get_parser(filename: str) -> BaseParser:
    """Return the appropriate parser for a given filename."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return PDFParser()
    if ext in (".xlsx", ".xls"):
        return ExcelParser()
    if ext in (".csv", ".tsv"):
        return CSVParser()
    raise ValueError(f"Unsupported file type: {ext}")


__all__ = [
    "BaseParser",
    "ParseResult",
    "PDFParser",
    "ExcelParser",
    "CSVParser",
    "HoldingsNormalizer",
    "get_parser",
]
