"""
Formatting utilities – locale-aware number and table formatting.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def fmt_ils(value: float, decimals: int = 0) -> str:
    """Format a value as ILS currency string."""
    if value is None:
        return "—"
    return f"₪{value:,.{decimals}f}"


def fmt_pct(value: Optional[float], decimals: int = 1) -> str:
    """Format a value as percentage string."""
    if value is None:
        return "—"
    return f"{value * 100:.{decimals}f}%"


def fmt_pct_raw(value: Optional[float], decimals: int = 2) -> str:
    """Format a value that is already a percentage (e.g. fee_percent=0.15 → '0.15%')."""
    if value is None:
        return "—"
    return f"{value:.{decimals}f}%"


def fmt_duration(value: Optional[float]) -> str:
    """Format duration in years."""
    if value is None:
        return "—"
    return f"{value:.1f} שנים"


def mark_estimated(value: str, is_estimated: bool) -> str:
    """Append an [E] marker to estimated values."""
    return f"{value} [E]" if is_estimated else value


def holdings_to_dataframe(holdings: list) -> pd.DataFrame:
    """Convert a list of HoldingNormalized to a display DataFrame."""
    rows = []
    for h in holdings:
        rows.append({
            "מזהה": h.row_id,
            "שם נייר": h.normalized_name,
            "סוג נכס": h.asset_class,
            "מטבע": h.currency,
            "שווי (ILS)": h.market_value_ils,
            "משקל": h.weight_in_portfolio,
            "ISIN": h.isin,
            "Ticker": h.ticker,
            "אזור": h.region,
            "ענף": h.sector,
            "קרן/ETF": "✓" if h.is_fund else "",
            "דמי ניהול %": h.fee_percent,
            "מח\"מ": h.duration,
            "הערות": h.notes,
        })
    return pd.DataFrame(rows)


def analysis_table_asset_alloc(rows: list) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "סוג נכס": r.asset_class,
            "שווי (ILS)": fmt_ils(r.market_value_ils),
            "אחוז מהתיק": fmt_pct(r.weight),
        }
        for r in rows
    ])


def analysis_table_geo(rows: list) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "אזור": r.region,
            "שווי מניות (ILS)": fmt_ils(r.market_value_ils),
            "מתוך מניות": fmt_pct(r.weight_in_equities),
            "מתוך תיק": fmt_pct(r.weight_in_portfolio),
            "הערה": "[E]" if r.is_estimated else "",
        }
        for r in rows
    ])


def analysis_table_sectors(rows: list) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "ענף": r.sector,
            "שווי (ILS)": fmt_ils(r.market_value_ils),
            "מתוך מניות": fmt_pct(r.weight_in_equities),
            "מקור": r.source_note[:60] + "…" if len(r.source_note) > 60 else r.source_note,
            "הערה": "[E]" if r.is_estimated else "",
        }
        for r in rows
    ])


def analysis_table_bonds(rows: list) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "סוג אגח": r.linkage_type,
            "שווי (ILS)": fmt_ils(r.market_value_ils),
            "מתוך אגח": fmt_pct(r.weight_in_bonds),
            "מתוך תיק": fmt_pct(r.weight_in_portfolio),
        }
        for r in rows
    ])


def analysis_table_duration(rows: list) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "שם": r.name,
            "סוג": r.bond_linkage_type,
            "שווי (ILS)": fmt_ils(r.market_value_ils),
            "מח\"מ (שנים)": fmt_duration(r.duration),
            "מקור": r.duration_source,
            "הערה": "[E]" if r.is_estimated else "",
            "תרומה משוקללת": f"{r.weighted_contribution:.3f}" if r.weighted_contribution else "—",
        }
        for r in rows
    ])


def analysis_table_fund_costs(rows: list) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "שם קרן": r.name,
            "שווי (ILS)": fmt_ils(r.market_value_ils),
            "משקל בתיק": fmt_pct(r.weight_in_portfolio),
            "דמי ניהול %": fmt_pct_raw(r.fee_percent),
            "עלות שנתית (ILS)": fmt_ils(r.annual_cost_ils) if r.annual_cost_ils else "—",
            "מקור": r.fee_source,
            "הערה": "[E]" if r.is_estimated else "",
        }
        for r in rows
    ])


def analysis_table_fx(rows: list) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "מטבע": r.currency,
            "שווי (ILS)": fmt_ils(r.market_value_ils),
            "אחוז מהתיק": fmt_pct(r.weight),
            "גידור": r.hedging_note,
        }
        for r in rows
    ])


def analysis_table_top_holdings(rows: list) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "דרוג": r.rank,
            "שם נייר": r.name,
            "שווי (ILS)": fmt_ils(r.market_value_ils),
            "משקל בתיק": fmt_pct(r.weight),
        }
        for r in rows
    ])


def analysis_table_assumptions(rows: list) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "שדה": r.field,
            "נייר": r.holding_name,
            "ערך מוערך": r.assumed_value,
            "סיבה": r.reason,
            "רמת ביטחון": r.confidence.value,
            "מקור": r.source,
        }
        for r in rows
    ])
