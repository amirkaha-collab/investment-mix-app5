"""
Shared pytest fixtures for the portfolio-analyzer test suite.
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.domain.models import HoldingNormalized, UserAnalysisPreferences
from src.config.constants import (
    ASSET_CLASS_EQUITY,
    ASSET_CLASS_BOND,
    ASSET_CLASS_CASH,
    BOND_LINKAGE_CPI,
    BOND_LINKAGE_USD,
    REGION_ISRAEL,
    REGION_US,
    REGION_EUROPE,
)


@pytest.fixture
def sample_preferences() -> UserAnalysisPreferences:
    return UserAnalysisPreferences(
        include_cash_in_allocation=True,
        portfolio_manager_fee_percent=0.75,
        manager_fee_is_assumption=False,
        classify_global_usd_bond_as_us_exposure=False,
        compute_extended_duration_with_estimates=True,
    )


@pytest.fixture
def sample_holdings() -> list[HoldingNormalized]:
    """A minimal but realistic set of holdings for testing."""
    return [
        HoldingNormalized(
            row_id="h001",
            raw_name="קרן מחקה S&P 500",
            normalized_name="S&P 500 Index Fund",
            asset_class=ASSET_CLASS_EQUITY,
            market_value=185_000,
            market_value_ils=185_000,
            currency="ILS",
            region=REGION_US,
            is_fund=True,
            is_etf=True,
            fee_percent=0.0945,
            fee_source="Mock",
            geography_breakdown={"USA": 1.0},
            sector_breakdown={
                "Information Technology": 0.31,
                "Financials": 0.13,
                "Health Care": 0.12,
                "Consumer Discretionary": 0.10,
                "Other": 0.34,
            },
            weight_in_portfolio=0.0,  # filled by normalizer
        ),
        HoldingNormalized(
            row_id="h002",
            raw_name="מניות ת\"א-125",
            normalized_name="TA-125 ETF",
            asset_class=ASSET_CLASS_EQUITY,
            market_value=98_000,
            market_value_ils=98_000,
            currency="ILS",
            region=REGION_ISRAEL,
            is_fund=True,
            is_etf=True,
            fee_percent=0.10,
            fee_source="Mock",
            geography_breakdown={"Israel": 1.0},
            weight_in_portfolio=0.0,
        ),
        HoldingNormalized(
            row_id="h003",
            raw_name="אגח ממשלתי גליל",
            normalized_name="Israel Govt CPI Bond",
            asset_class=ASSET_CLASS_BOND,
            market_value=103_500,
            market_value_ils=103_500,
            currency="ILS",
            region=REGION_ISRAEL,
            is_bond=True,
            bond_linkage_type=BOND_LINKAGE_CPI,
            duration=5.2,
            duration_source="TASE",
            weight_in_portfolio=0.0,
        ),
        HoldingNormalized(
            row_id="h004",
            raw_name="AGG ETF",
            normalized_name="iShares US Aggregate Bond ETF",
            asset_class=ASSET_CLASS_BOND,
            market_value=31_500,
            market_value_ils=31_500 * 3.75,
            currency="USD",
            region=REGION_US,
            is_fund=True,
            is_etf=True,
            is_bond=True,
            bond_linkage_type=BOND_LINKAGE_USD,
            duration=6.2,
            duration_source="BlackRock factsheet",
            fee_percent=0.03,
            fee_source="BlackRock factsheet",
            weight_in_portfolio=0.0,
        ),
        HoldingNormalized(
            row_id="h005",
            raw_name="פיקדון",
            normalized_name="Bank Deposit",
            asset_class=ASSET_CLASS_CASH,
            market_value=50_000,
            market_value_ils=50_000,
            currency="ILS",
            weight_in_portfolio=0.0,
        ),
    ]


@pytest.fixture
def sample_csv_content() -> bytes:
    """Minimal valid CSV for parser tests."""
    return (
        "שם נייר,שווי שוק,מטבע,סוג\n"
        "SPY ETF,185000,ILS,מניות\n"
        "AGG ETF,118125,ILS,אגח\n"
        "פיקדון,50000,ILS,מזומן\n"
    ).encode("utf-8-sig")
