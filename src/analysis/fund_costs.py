"""
Analysis module: Fund Costs

Rule 7 – For each fund/ETF, capture fee_percent and compute weighted cost.
Rule 8 – Total portfolio cost = fund cost + PM fee (only if PM fee was provided).

Two denominators:
  weighted_fund_cost_on_funds     : denominator = funds/ETFs only
  effective_fund_cost_on_total    : denominator = entire portfolio
"""

from __future__ import annotations

from typing import Optional

from src.domain.models import (
    FundCostRow,
    HoldingNormalized,
    UserAnalysisPreferences,
)


def compute_fund_costs(
    holdings: list[HoldingNormalized],
    prefs: UserAnalysisPreferences,
) -> tuple[list[FundCostRow], Optional[float], Optional[float], Optional[float], bool]:
    """
    Compute fund-level cost table and portfolio cost metrics.

    Returns
    -------
    (rows, weighted_cost_on_funds, effective_cost_on_total,
     total_cost_percent, total_cost_is_assumption)
    """
    funds = [h for h in holdings if h.is_fund or h.is_etf]
    total_portfolio = sum(h.market_value_ils for h in holdings)
    total_funds = sum(h.market_value_ils for h in funds)

    rows: list[FundCostRow] = []
    weighted_fund_numerator = 0.0
    weighted_portfolio_numerator = 0.0

    for h in funds:
        is_estimated = any("fee_percent" in ef for ef in h.estimated_fields)
        annual_cost_ils = None
        if h.fee_percent is not None:
            annual_cost_ils = h.market_value_ils * h.fee_percent / 100.0
            weighted_fund_numerator += h.fee_percent * h.market_value_ils
            weighted_portfolio_numerator += h.fee_percent * h.market_value_ils

        rows.append(
            FundCostRow(
                row_id=h.row_id,
                name=h.normalized_name,
                asset_class=h.asset_class,
                market_value_ils=round(h.market_value_ils, 2),
                weight_in_portfolio=h.weight_in_portfolio,
                fee_percent=h.fee_percent,
                fee_source=h.fee_source,
                is_estimated=is_estimated,
                annual_cost_ils=round(annual_cost_ils, 2) if annual_cost_ils is not None else None,
            )
        )

    # Metrics
    wt_cost_on_funds: Optional[float] = None
    if total_funds > 0 and weighted_fund_numerator > 0:
        wt_cost_on_funds = round(weighted_fund_numerator / total_funds, 4)

    eff_cost_on_total: Optional[float] = None
    if total_portfolio > 0 and weighted_portfolio_numerator > 0:
        eff_cost_on_total = round(weighted_portfolio_numerator / total_portfolio, 4)

    # Total cost (Rule 8: only when PM fee is known)
    total_cost: Optional[float] = None
    is_assumption = False

    if eff_cost_on_total is not None and prefs.portfolio_manager_fee_percent is not None:
        total_cost = round(eff_cost_on_total + prefs.portfolio_manager_fee_percent, 4)
        is_assumption = prefs.manager_fee_is_assumption
    elif eff_cost_on_total is not None:
        # PM fee not provided – do not invent a total cost
        total_cost = None

    return rows, wt_cost_on_funds, eff_cost_on_total, total_cost, is_assumption
