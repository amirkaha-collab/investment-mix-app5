"""
Abstract base class for all document parsers.

Each parser accepts a file path (or bytes) and returns a list of raw rows
extracted from the document.  Normalisation happens in a separate step.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd


class ParseResult:
    """
    Container for parser output.

    raw_tables   : list of DataFrames extracted before any cleaning
    primary_df   : best-guess holdings DataFrame (after basic cleaning)
    parse_method : human-readable description of how the data was extracted
    warnings     : non-fatal issues encountered during parsing
    errors       : fatal errors (primary_df may be empty)
    """

    def __init__(
        self,
        primary_df: pd.DataFrame,
        raw_tables: list[pd.DataFrame] | None = None,
        parse_method: str = "",
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
    ) -> None:
        self.primary_df = primary_df
        self.raw_tables = raw_tables or []
        self.parse_method = parse_method
        self.warnings: list[str] = warnings or []
        self.errors: list[str] = errors or []

    @property
    def success(self) -> bool:
        return not self.errors and not self.primary_df.empty


class BaseParser(ABC):
    """
    All parsers implement this interface.

    Usage
    -----
    parser = PDFParser()
    result = parser.parse(path_or_bytes)
    """

    @abstractmethod
    def parse(self, source: Union[Path, bytes, io.BytesIO]) -> ParseResult:
        """
        Extract raw holdings data from the source file.

        Parameters
        ----------
        source : Path | bytes | BytesIO

        Returns
        -------
        ParseResult
        """
        ...

    @staticmethod
    def _to_bytesio(source: Union[Path, bytes, io.BytesIO]) -> io.BytesIO:
        if isinstance(source, Path):
            return io.BytesIO(source.read_bytes())
        if isinstance(source, bytes):
            return io.BytesIO(source)
        source.seek(0)
        return source

    @staticmethod
    def _clean_header(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase, strip, and de-duplicate column names."""
        df.columns = [
            str(c).strip().lower().replace(" ", "_").replace("\n", "_")
            for c in df.columns
        ]
        # Remove entirely blank rows
        df = df.dropna(how="all")
        return df
