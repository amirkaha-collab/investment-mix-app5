"""
Tests for the holdings normalizer.
"""

from __future__ import annotations

import io
import pytest
import pandas as pd

from src.parsers.normalizer import HoldingsNormalizer, _clean_number, _infer_asset_class
from src.config.constants import ASSET_CLASS_BOND, ASSET_CLASS_CASH, ASSET_CLASS_EQUITY


class TestCleanNumber:
    def test_plain_float(self):
        assert _clean_number("123456.78") == pytest.approx(123456.78)

    def test_comma_thousands(self):
        assert _clean_number("1,234,567") == pytest.approx(1_234_567.0)

    def test_shekel_symbol(self):
        assert _clean_number("₪500,000") == pytest.approx(500_000.0)

    def test_negative_parenthesis(self):
        assert _clean_number("(1000)") == pytest.approx(-1000.0)

    def test_empty_returns_none(self):
        assert _clean_number("") is None
        assert _clean_number("N/A") is None
        assert _clean_number("-") is None


class TestInferAssetClass:
    def test_equity_keywords(self):
        assert _infer_asset_class("מניות ת\"א 125") == ASSET_CLASS_EQUITY
        assert _infer_asset_class("S&P 500 ETF") == ASSET_CLASS_EQUITY

    def test_bond_keywords(self):
        assert _infer_asset_class("אגרות חוב ממשלתיות") == ASSET_CLASS_BOND
        assert _infer_asset_class("US Treasury Bond ETF") == ASSET_CLASS_BOND

    def test_cash_keywords(self):
        assert _infer_asset_class("קרן כספית") == ASSET_CLASS_CASH
        assert _infer_asset_class("money market") == ASSET_CLASS_CASH


class TestHoldingsNormalizer:
    def test_basic_normalization(self):
        df = pd.DataFrame({
            "שם נייר": ["SPY ETF", "AGG Bond", "Cash Deposit"],
            "שווי שוק": [185_000, 50_000, 30_000],
            "מטבע": ["ILS", "ILS", "ILS"],
        })
        normalizer = HoldingsNormalizer(usd_to_ils=3.75)
        holdings, warnings = normalizer.normalize(df)

        assert len(holdings) == 3
        total = sum(h.market_value_ils for h in holdings)
        assert total == pytest.approx(265_000)

    def test_weights_sum_to_one(self):
        df = pd.DataFrame({
            "שם נייר": ["A", "B", "C"],
            "שווי שוק": [100, 200, 700],
            "מטבע": ["ILS", "ILS", "ILS"],
        })
        normalizer = HoldingsNormalizer()
        holdings, _ = normalizer.normalize(df)
        total_weight = sum(h.weight_in_portfolio for h in holdings)
        assert total_weight == pytest.approx(1.0, abs=1e-5)

    def test_skip_zero_value_rows(self):
        df = pd.DataFrame({
            "שם נייר": ["Good", "Zero", "Also Good"],
            "שווי שוק": [10_000, 0, 20_000],
            "מטבע": ["ILS", "ILS", "ILS"],
        })
        normalizer = HoldingsNormalizer()
        holdings, warnings = normalizer.normalize(df)
        assert len(holdings) == 2
        assert any("Zero" in w or "0" in w for w in warnings)

    def test_usd_conversion(self):
        df = pd.DataFrame({
            "שם נייר": ["US ETF"],
            "שווי שוק": [1_000],
            "מטבע": ["USD"],
        })
        normalizer = HoldingsNormalizer(usd_to_ils=4.0)
        holdings, _ = normalizer.normalize(df)
        assert holdings[0].market_value_ils == pytest.approx(4_000.0)

    def test_skip_total_row(self):
        df = pd.DataFrame({
            "שם נייר": ["Stock A", "סה\"כ"],
            "שווי שוק": [50_000, 50_000],
            "מטבע": ["ILS", "ILS"],
        })
        normalizer = HoldingsNormalizer()
        holdings, _ = normalizer.normalize(df)
        assert len(holdings) == 1
        assert holdings[0].raw_name == "Stock A"
