"""
Excel / CSV parsers.  These are simpler than the PDF parser because the data
is already structured.  The main work is header detection and normalisation.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

import pandas as pd

from .base import BaseParser, ParseResult

logger = logging.getLogger(__name__)


class ExcelParser(BaseParser):
    """
    Parse XLSX / XLS files containing a holdings table.
    Supports multi-sheet workbooks; picks the sheet most likely to be holdings.
    """

    SKIP_SHEET_KEYWORDS = ["summary", "cover", "סיכום", "שער", "readme"]

    def parse(self, source: Union[Path, bytes, io.BytesIO]) -> ParseResult:
        buf = self._to_bytesio(source)
        warnings: list[str] = []

        try:
            xl = pd.ExcelFile(buf)
        except Exception as exc:
            return ParseResult(
                pd.DataFrame(),
                errors=[f"Cannot open Excel file: {exc}"],
                parse_method="excel",
            )

        candidate_sheets: list[tuple[str, pd.DataFrame]] = []

        for sheet_name in xl.sheet_names:
            if any(kw in sheet_name.lower() for kw in self.SKIP_SHEET_KEYWORDS):
                warnings.append(f"Skipping sheet '{sheet_name}' (summary/cover)")
                continue
            try:
                df = xl.parse(sheet_name, header=None)
                df = self._detect_and_set_header(df)
                if df is not None and not df.empty:
                    candidate_sheets.append((sheet_name, df))
            except Exception as exc:
                warnings.append(f"Sheet '{sheet_name}' parse error: {exc}")

        if not candidate_sheets:
            return ParseResult(
                pd.DataFrame(),
                warnings=warnings,
                errors=["No usable holdings sheet found in Excel file."],
                parse_method="excel",
            )

        # Pick the sheet with the most rows
        sheet_name, best_df = max(candidate_sheets, key=lambda x: len(x[1]))
        logger.info(f"ExcelParser: selected sheet '{sheet_name}' with {len(best_df)} rows")

        return ParseResult(
            primary_df=best_df,
            raw_tables=[df for _, df in candidate_sheets],
            parse_method=f"excel::{sheet_name}",
            warnings=warnings,
        )

    def _detect_and_set_header(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """
        Scan the first 20 rows for the header row (contains keywords like שווי / value).
        """
        HEADER_HINTS = [
            "שווי", "value", "שם", "name", "security", "isin",
            "כמות", "quantity", "מטבע", "currency", "סוג",
        ]
        for i, row in df.head(20).iterrows():
            row_text = " ".join(str(v) for v in row.values).lower()
            if any(h in row_text for h in HEADER_HINTS):
                # This row is the header
                df.columns = [str(v).strip() for v in row.values]
                df = df.iloc[i + 1 :].reset_index(drop=True)
                df = self._clean_header(df)
                df = df.dropna(how="all")
                return df

        # No header detected; treat first row as header
        df = self._clean_header(df)
        return df if not df.empty else None


class CSVParser(BaseParser):
    """
    Parse CSV / TSV holdings files.
    Auto-detects delimiter and encoding.
    """

    DELIMITERS = [",", ";", "\t", "|"]

    def parse(self, source: Union[Path, bytes, io.BytesIO]) -> ParseResult:
        buf = self._to_bytesio(source)
        raw_bytes = buf.read()

        # Try UTF-8, then cp1255 (Windows Hebrew), then latin-1
        for encoding in ("utf-8-sig", "utf-8", "cp1255", "latin-1"):
            try:
                text = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            return ParseResult(
                pd.DataFrame(),
                errors=["Could not decode CSV file.  Please save as UTF-8."],
                parse_method="csv",
            )

        # Try each delimiter
        best_df: pd.DataFrame | None = None
        best_cols = 0

        for delim in self.DELIMITERS:
            try:
                df = pd.read_csv(
                    io.StringIO(text),
                    sep=delim,
                    engine="python",
                    dtype=str,
                    na_filter=False,
                )
                if df.shape[1] > best_cols:
                    best_cols = df.shape[1]
                    best_df = df
            except Exception:
                continue

        if best_df is None or best_df.empty:
            return ParseResult(
                pd.DataFrame(),
                errors=["CSV parsing produced an empty table."],
                parse_method="csv",
            )

        best_df = self._clean_header(best_df)
        best_df = best_df.dropna(how="all")

        return ParseResult(
            primary_df=best_df,
            parse_method="csv",
        )
