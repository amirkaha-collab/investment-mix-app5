"""
Analysis module: FX Exposure + Concentration

Rule 9 – Currency/FX exposure table
Rule 10 – Concentration analysis (top-N holdings)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from src.config.constants import (
    CURRENCY_ILS,
    QA_CONCENTRATION_WARN_TOP10,
    QA_MAX_SINGLE_HOLDING_WEIGHT,
)
from src.domain.models import (
    ConcentrationRow,
    FXExposureRow,
    HoldingNormalized,
    UserAnalysisPreferences,
)


# ─── FX Exposure ────────────────────────────────────────────────────────────────

def compute_fx_exposure(
    holdings: list[HoldingNormalized],
    prefs: UserAnalysisPreferences,
) -> list[FXExposureRow]:
    """
    Aggregate holdings by currency, noting hedging status.

    A holding is ILS-exposed when:
      - currency == ILS, OR
      - currency != ILS but is_fx_hedged == True

    A holding is FX-exposed when:
      - currency != ILS AND is_fx_hedged != True

    is_fx_hedged == None → uncertain; reported in the 'unknown hedging' bucket.
    """
    total = sum(h.market_value_ils for h in holdings)
    if total <= 0:
        return []

    # key = (currency, is_hedged)
    buckets: dict[tuple[str, Optional[bool]], float] = defaultdict(float)

    for h in holdings:
        ccy = h.currency.upper() if h.currency else CURRENCY_ILS
        hedged = h.is_fx_hedged
        if ccy == CURRENCY_ILS:
            hedged = True  # ILS is effectively hedged
        buckets[(ccy, hedged)] += h.market_value_ils

    rows: list[FXExposureRow] = []
    for (ccy, hedged), value in sorted(buckets.items(), key=lambda x: -x[1]):
        if hedged is True:
            note = "ILS-hedged" if ccy != CURRENCY_ILS else "ILS"
        elif hedged is False:
            note = "Unhedged FX exposure"
        else:
            note = "Hedging status unknown"

        rows.append(
            FXExposureRow(
                currency=ccy,
                market_value_ils=round(value, 2),
                weight=round(value / total, 6),
                is_hedged=hedged,
                hedging_note=note,
            )
        )
    return rows


# ─── Concentration ──────────────────────────────────────────────────────────────

def compute_concentration(
    holdings: list[HoldingNormalized],
    top_n: int = 10,
) -> tuple[list[ConcentrationRow], Optional[float], Optional[float], Optional[float], list[str]]:
    """
    Compute concentration metrics.

    Returns
    -------
    (top_rows, top5_weight, top10_weight, max_single_weight, warnings)
    """
    sorted_holdings = sorted(holdings, key=lambda h: -h.market_value_ils)
    total = sum(h.market_value_ils for h in holdings)
    warnings: list[str] = []

    rows: list[ConcentrationRow] = []
    for rank, h in enumerate(sorted_holdings[:top_n], start=1):
        rows.append(
            ConcentrationRow(
                rank=rank,
                row_id=h.row_id,
                name=h.normalized_name,
                market_value_ils=round(h.market_value_ils, 2),
                weight=h.weight_in_portfolio,
            )
        )

    top5 = round(sum(h.weight_in_portfolio for h in sorted_holdings[:5]), 4)
    top10 = round(sum(h.weight_in_portfolio for h in sorted_holdings[:10]), 4)
    max_single = sorted_holdings[0].weight_in_portfolio if sorted_holdings else None

    if max_single and max_single > QA_MAX_SINGLE_HOLDING_WEIGHT:
        warnings.append(
            f"High single-holding concentration: "
            f"{sorted_holdings[0].normalized_name} = {max_single:.1%} of portfolio"
        )
    if top10 > QA_CONCENTRATION_WARN_TOP10:
        warnings.append(
            f"Top-10 holdings represent {top10:.1%} of portfolio – significant concentration."
        )

    return rows, top5, top10, max_single, warnings
