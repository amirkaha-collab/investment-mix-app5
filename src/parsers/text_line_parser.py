"""
Text Line Parser – last-resort parser for raw text extracted from PDFs.

When pdfplumber and OCR both fail or aren't available, this parser
attempts to reconstruct holdings from raw text lines using:
  1. Hebrew/English pattern matching for financial data
  2. Line-by-line heuristic: a valid holdings row has a name + a number
  3. Header detection to assign column names

This is intentionally a "best effort" fallback.
All output is marked with warnings so the user knows to verify.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Pattern: a numeric value (possibly formatted with commas/decimals)
_NUM_PATTERN  = re.compile(r"[\d,\.]{3,}")
# Pattern: ISIN (IL/US + 10 chars)
_ISIN_PATTERN = re.compile(r"\b[A-Z]{2}[A-Z0-9]{10}\b")
# Pattern: currency codes
_CCY_PATTERN  = re.compile(r"\b(ILS|USD|EUR|GBP|CHF|JPY|₪|\$)\b", re.IGNORECASE)
# Asset class hints
_ASSET_HINTS  = re.compile(r"\b(מניות|מניה|אגח|אג\"ח|מזומן|bond|equity|cash|etf|fund|קרן)\b", re.IGNORECASE)

# Min tokens in a line to be considered a data row
_MIN_TOKENS = 2


def parse_text_lines_to_dataframe(lines: list[str]) -> Optional[pd.DataFrame]:
    """
    Try to parse a list of raw text lines into a holdings DataFrame.

    Strategy:
      1. Find the header line (contains שם/name/security + שווי/value)
      2. For each subsequent line, try to extract:
           raw_name | market_value | currency | asset_class_hint | isin
      3. Return DataFrame or None if fewer than 2 rows extracted
    """
    if not lines:
        return None

    # Step 1: find header line index
    header_idx = _find_header_idx(lines)

    # Step 2: parse data lines
    rows: list[dict] = []

    start = header_idx + 1 if header_idx is not None else 0
    for line in lines[start:]:
        row = _parse_line(line.strip())
        if row:
            rows.append(row)

    if len(rows) < 1:
        logger.debug(f"text_line_parser: only {len(rows)} rows extracted from {len(lines)} lines")
        return None

    df = pd.DataFrame(rows)
    logger.info(f"text_line_parser: extracted {len(df)} candidate rows from text")
    return df


def _find_header_idx(lines: list[str]) -> Optional[int]:
    """Find the index of the line that looks like a table header."""
    HEADER_KW = ["שם נייר", "שם", "security", "name", "נייר", "שווי", "value", "market"]
    for i, line in enumerate(lines[:30]):
        line_lower = line.lower()
        hits = sum(1 for kw in HEADER_KW if kw in line_lower)
        if hits >= 2:
            return i
    return None


def _parse_line(line: str) -> Optional[dict]:
    """
    Try to extract structured data from a single line of text.
    Returns a dict with raw_name, market_value, currency, etc., or None.
    """
    if not line or len(line.split()) < _MIN_TOKENS:
        return None

    # Extract all numbers from the line
    numbers = _NUM_PATTERN.findall(line)
    if not numbers:
        return None   # no number → not a data row

    # Extract currency
    ccy_match = _CCY_PATTERN.search(line)
    currency = ccy_match.group(1).upper() if ccy_match else "ILS"
    if currency == "₪":
        currency = "ILS"
    if currency == "$":
        currency = "USD"

    # Extract ISIN if present
    isin_match = _ISIN_PATTERN.search(line)
    isin = isin_match.group(0) if isin_match else ""

    # Extract asset class hint
    asset_match = _ASSET_HINTS.search(line)
    asset_hint = asset_match.group(0) if asset_match else ""

    # The "name" is everything before the first number, stripped
    first_num_pos = _NUM_PATTERN.search(line)
    if first_num_pos:
        name_part = line[:first_num_pos.start()].strip()
        # Remove currency symbols from name
        name_part = _CCY_PATTERN.sub("", name_part).strip()
    else:
        name_part = line.strip()

    # Remove ISIN from name
    if isin:
        name_part = name_part.replace(isin, "").strip()

    if not name_part or len(name_part) < 2:
        return None

    # Pick the largest number as the market value (most likely the value, not quantity)
    def _to_float(s: str) -> float:
        try:
            return float(s.replace(",", ""))
        except ValueError:
            return 0.0

    market_value = max((_to_float(n) for n in numbers), default=0.0)
    if market_value <= 0:
        return None

    return {
        "raw_name":         name_part,
        "market_value":     market_value,
        "currency":         currency,
        "isin":             isin,
        "asset_class_hint": asset_hint,
    }
