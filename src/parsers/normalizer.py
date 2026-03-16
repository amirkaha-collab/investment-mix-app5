"""
Normaliser: converts a raw parsed DataFrame into a list of HoldingNormalized.

Key design rule: column name matching uses a SINGLE normalisation function
(`_norm_key`) applied consistently to BOTH the DataFrame column headers
(already normalised by BaseParser._clean_header) AND the keys in
COLUMN_SYNONYMS.  This eliminates the Hebrew-with-spaces vs
lowercase-with-underscores mismatch that caused zero holdings.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from src.config.constants import (
    ASSET_CLASS_BOND,
    ASSET_CLASS_CASH,
    ASSET_CLASS_EQUITY,
    ASSET_CLASS_OTHER,
    BOND_LINKAGE_CPI,
    BOND_LINKAGE_NOMINAL_ILS,
    BOND_LINKAGE_UNKNOWN,
    BOND_LINKAGE_USD,
    CURRENCY_ILS,
    CURRENCY_USD,
    REGION_ISRAEL,
    REGION_UNKNOWN,
    REGION_US,
)
from src.domain.models import HoldingNormalized

logger = logging.getLogger(__name__)


# ─── Shared normalisation function ─────────────────────────────────────────────
# Applied to BOTH DataFrame column names AND COLUMN_SYNONYMS keys so they
# always match, regardless of how the original file was formatted.

def _norm_key(value: str) -> str:
    """Normalise a column name to a matchable key (lowercase, underscores)."""
    return str(value).strip().lower().replace(" ", "_").replace("\n", "_").replace('"', "").replace("'", "")


# ─── Column synonym map ─────────────────────────────────────────────────────────
# Keys are normalised via _norm_key so they match what _clean_header produces.
_RAW_SYNONYMS: dict[str, str] = {
    # Name
    "שם נייר": "raw_name",
    'שם ני"ע': "raw_name",
    "שם ניירערך": "raw_name",
    "security name": "raw_name",
    "name": "raw_name",
    "security": "raw_name",
    "נייר": "raw_name",
    "תיאור": "raw_name",
    "instrument": "raw_name",
    "שם": "raw_name",

    # ISIN
    "isin": "isin",
    'מסי"ן': "isin",

    # Ticker
    "ticker": "ticker",
    "סימול": "ticker",
    "symbol": "ticker",

    # Quantity
    "כמות": "quantity",
    "quantity": "quantity",
    "units": "quantity",
    "nominal": "quantity",
    "נומינל": "quantity",
    "יחידות": "quantity",

    # Market value
    "שווי שוק": "market_value",
    "שווי": "market_value",
    'שווי שוק בש"ח': "market_value_ils",
    "שווי נוכחי": "market_value",
    "market value": "market_value",
    "value": "market_value",
    "current value": "market_value",
    "סכום": "market_value",
    'שווי בש"ח': "market_value_ils",

    # Currency
    "מטבע": "currency",
    "currency": "currency",
    "ccy": "currency",

    # Asset class / type
    "סוג": "asset_class_hint",
    "סוג נייר": "asset_class_hint",
    "type": "asset_class_hint",
    "asset class": "asset_class_hint",
    "asset_class": "asset_class",
    "classification": "asset_class_hint",
    "סיווג": "asset_class_hint",

    # Weight
    "אחוז מהתיק": "weight_hint",
    "weight": "weight_hint",
    "% מהתיק": "weight_hint",
    "% of portfolio": "weight_hint",

    # Sector
    "ענף": "sector",
    "sector": "sector",

    # Country / region
    "מדינה": "country",
    "country": "country",
    "region": "region",
    "אזור": "region",
}

# Build the lookup dict with normalised keys
COLUMN_SYNONYMS: dict[str, str] = {
    _norm_key(k): v for k, v in _RAW_SYNONYMS.items()
}


# ─── Asset class heuristics ────────────────────────────────────────────────────

EQUITY_KEYWORDS = [
    "מניה", "מניות", "equity", "stock", "share", "etf מניות",
    "קרן מחקה מניות", "קרן סל", "מדד מניות",
    " etf", "etf ", "(etf)", "s&p", "nasdaq", "msci", "russell",
]
BOND_KEYWORDS = [
    "אגח", 'אג"ח', "אגרות חוב", "bond", "bonds", "obligat",
    "ממשלתי", "corporate bond", "treasury", "tlt",
]
CASH_KEYWORDS = [
    "מזומן", "cash", "פקדון", "deposit", "money market",
    "קרן כספית", "כספית",
]

BOND_CPI_KEYWORDS = ["צמוד", "מדד", "cpi", "linked", "גליל"]
BOND_USD_KEYWORDS = ["דולר", "usd", "$", "dollar"]


def _infer_asset_class(name: str, hint: str = "") -> str:
    text = (name + " " + hint).lower()
    if any(k in text for k in CASH_KEYWORDS):
        return ASSET_CLASS_CASH
    if any(k in text for k in BOND_KEYWORDS):
        return ASSET_CLASS_BOND
    if any(k in text for k in EQUITY_KEYWORDS):
        return ASSET_CLASS_EQUITY
    if "etf" in text.split() or text.strip().endswith("etf"):
        return ASSET_CLASS_EQUITY
    return ASSET_CLASS_OTHER


def _infer_bond_linkage(name: str, currency: str) -> str:
    text = name.lower()
    if any(k in text for k in BOND_CPI_KEYWORDS):
        return BOND_LINKAGE_CPI
    if currency.upper() == CURRENCY_USD or any(k in text for k in BOND_USD_KEYWORDS):
        return BOND_LINKAGE_USD
    if "שקל" in text or "nominal" in text:
        return BOND_LINKAGE_NOMINAL_ILS
    return BOND_LINKAGE_UNKNOWN


def _clean_number(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s in ("-", "N/A", "n/a", "", "—"):
        return None
    s = re.sub(r"[₪,$€£,\s]", "", s)
    s = s.replace("(", "-").replace(")", "")
    try:
        return float(s)
    except ValueError:
        return None


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename df columns to canonical names using COLUMN_SYNONYMS.

    Both the incoming column names AND the synonym keys have already been
    normalised via _norm_key (columns by BaseParser._clean_header,
    synonym keys at module load time), so the comparison is apples-to-apples.
    """
    rename_map: dict[str, str] = {}
    for col in df.columns:
        normed = _norm_key(str(col))
        if normed in COLUMN_SYNONYMS:
            canonical = COLUMN_SYNONYMS[normed]
            if canonical not in df.columns:      # don't rename if target exists
                rename_map[col] = canonical
    return df.rename(columns=rename_map)


class HoldingsNormalizer:
    """
    Converts a raw (post-parse) DataFrame to validated HoldingNormalized objects.
    """

    def __init__(self, usd_to_ils: float = 3.75, default_currency: str = "ILS") -> None:
        self.usd_to_ils = usd_to_ils
        self.default_currency = default_currency

    def normalize(
        self, df: pd.DataFrame
    ) -> tuple[list[HoldingNormalized], list[str]]:
        warnings: list[str] = []
        df = _map_columns(df)

        if "raw_name" not in df.columns:
            df = df.rename(columns={df.columns[0]: "raw_name"})
            warnings.append(
                "Could not identify a 'name' column – using the first column as "
                "security name.  Please verify in the review step."
            )

        holdings: list[HoldingNormalized] = []
        for _, row in df.iterrows():
            h = self._process_row(row, warnings)
            if h is not None:
                holdings.append(h)

        total = sum(h.market_value_ils for h in holdings)
        if total > 0:
            for h in holdings:
                h.weight_in_portfolio = round(h.market_value_ils / total, 6)

        return holdings, warnings

    def _process_row(
        self, row: pd.Series, warnings: list[str]
    ) -> HoldingNormalized | None:
        raw_name = str(row.get("raw_name", "")).strip()
        if not raw_name or raw_name.lower() in ("nan", "none", "", "total", 'סה"כ', "סהכ"):
            return None

        mv_raw = _clean_number(
            row.get("market_value") or row.get("market_value_ils")
        )
        if mv_raw is None:
            warnings.append(f"Row '{raw_name}': no market value found – skipped")
            return None
        if mv_raw <= 0:
            warnings.append(f"Row '{raw_name}': market value = {mv_raw} (≤0) – skipped")
            return None

        currency = str(row.get("currency", self.default_currency)).strip().upper()
        if not currency or currency == "NAN":
            currency = self.default_currency

        mv_ils = self._to_ils(mv_raw, currency)

        ac_hint = str(row.get("asset_class_hint", "")).strip()
        ac = row.get("asset_class") or _infer_asset_class(raw_name, ac_hint)
        ac = str(ac).strip().lower() or ASSET_CLASS_OTHER

        bond_linkage = ""
        if ac == ASSET_CLASS_BOND:
            bond_linkage = _infer_bond_linkage(raw_name, currency)

        isin   = str(row.get("isin",   "")).strip().upper()
        ticker = str(row.get("ticker", "")).strip().upper()
        qty    = _clean_number(row.get("quantity"))
        sector = str(row.get("sector",  "")).strip()
        country = str(row.get("country", "")).strip()
        region  = str(row.get("region",  "")).strip()

        if not region and country:
            region = self._infer_region(country)

        name_lower = raw_name.lower()
        is_etf  = "etf" in name_lower or "קרן סל" in name_lower
        is_fund = is_etf or "קרן" in name_lower or "fund" in name_lower

        estimated_fields: list[str] = []
        if ac == ASSET_CLASS_OTHER and not ac_hint:
            estimated_fields.append("asset_class: inferred from name, could not confirm")

        return HoldingNormalized(
            raw_name=raw_name,
            normalized_name=raw_name,
            asset_class=ac,
            quantity=qty,
            market_value=mv_raw,
            market_value_ils=mv_ils,
            currency=currency,
            isin=isin,
            ticker=ticker,
            bond_linkage_type=bond_linkage,
            is_etf=is_etf,
            is_fund=is_fund,
            sector=sector,
            country=country,
            region=region or REGION_UNKNOWN,
            estimated_fields=estimated_fields,
        )

    def _to_ils(self, value: float, currency: str) -> float:
        if currency in (CURRENCY_ILS, ""):
            return value
        if currency == CURRENCY_USD:
            return value * self.usd_to_ils
        return value * self.usd_to_ils  # other FX: use USD rate as proxy

    @staticmethod
    def _infer_region(country: str) -> str:
        COUNTRY_TO_REGION = {
            "ישראל": REGION_ISRAEL, "israel": REGION_ISRAEL, "il": REGION_ISRAEL,
            "us": REGION_US, "usa": REGION_US, "united states": REGION_US, 'ארה"ב': REGION_US,
        }
        return COUNTRY_TO_REGION.get(country.lower().strip(), REGION_UNKNOWN)
