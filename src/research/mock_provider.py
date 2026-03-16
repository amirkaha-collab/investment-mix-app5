"""
Mock research provider.

Used during development, testing, and when RESEARCH_PROVIDER=mock.
Returns realistic but synthetic data for a fixed set of known tickers/ISINs.
For unknowns, returns rule-based estimates flagged as estimated.

No network calls are made.
"""

from __future__ import annotations

import logging

from src.config.constants import (
    ASSET_CLASS_BOND,
    ASSET_CLASS_EQUITY,
    CONFIDENCE_ESTIMATED,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    DURATION_FALLBACK_GLOBAL_BOND_ETF,
    DURATION_FALLBACK_LONG_BOND,
    DURATION_FALLBACK_MEDIUM_BOND,
    DURATION_FALLBACK_SHORT_BOND,
    KNOWN_TICKERS,
    MSCI_WORLD_SECTOR_WEIGHTS,
    SP500_SECTOR_WEIGHTS,
)
from .base import ResearchProvider, ResearchResult

logger = logging.getLogger(__name__)

# ─── Extended mock database ────────────────────────────────────────────────────
MOCK_DB: dict[str, dict] = {
    # ── US ETFs ──────────────────────────────────────────────────────────────
    "SPY": {
        "security_name": "SPDR S&P 500 ETF Trust",
        "fee_percent": 0.0945,
        "fee_source": "State Street Global Advisors factsheet",
        "benchmark": "S&P 500",
        "sector_breakdown": SP500_SECTOR_WEIGHTS,
        "geography_breakdown": {"USA": 1.0},
        "is_leveraged": False,
        "is_fx_hedged": False,
        "confidence_score": CONFIDENCE_HIGH,
        "source_urls": ["https://www.ssga.com/us/en/individual/etfs/funds/spdr-sp-500-etf-trust-spy"],
    },
    "QQQ": {
        "security_name": "Invesco QQQ Trust",
        "fee_percent": 0.20,
        "fee_source": "Invesco factsheet",
        "benchmark": "NASDAQ-100",
        "geography_breakdown": {"USA": 1.0},
        "sector_breakdown": {
            "Information Technology": 0.49, "Communication Services": 0.17,
            "Consumer Discretionary": 0.14, "Health Care": 0.06,
            "Industrials": 0.05, "Consumer Staples": 0.05, "Financials": 0.02,
            "Other": 0.02,
        },
        "is_leveraged": False,
        "confidence_score": CONFIDENCE_HIGH,
        "source_urls": ["https://www.invesco.com/us/financial-products/etfs/product-detail?audienceType=Investor&ticker=QQQ"],
    },
    "VTI": {
        "security_name": "Vanguard Total Stock Market ETF",
        "fee_percent": 0.03,
        "fee_source": "Vanguard factsheet",
        "benchmark": "CRSP US Total Market",
        "geography_breakdown": {"USA": 1.0},
        "sector_breakdown": SP500_SECTOR_WEIGHTS,
        "confidence_score": CONFIDENCE_HIGH,
        "source_urls": ["https://investor.vanguard.com/investment-products/etfs/profile/vti"],
    },
    "IEFA": {
        "security_name": "iShares Core MSCI EAFE ETF",
        "fee_percent": 0.07,
        "fee_source": "BlackRock factsheet",
        "benchmark": "MSCI EAFE",
        "geography_breakdown": {
            "Japan": 0.22, "United Kingdom": 0.14, "France": 0.11,
            "Switzerland": 0.10, "Germany": 0.09, "Australia": 0.07,
            "Netherlands": 0.05, "Sweden": 0.04, "Other": 0.18,
        },
        "sector_breakdown": MSCI_WORLD_SECTOR_WEIGHTS,
        "confidence_score": CONFIDENCE_HIGH,
        "source_urls": ["https://www.ishares.com/us/products/239600/"],
    },
    "AGG": {
        "security_name": "iShares Core U.S. Aggregate Bond ETF",
        "fee_percent": 0.03,
        "fee_source": "BlackRock factsheet",
        "benchmark": "Bloomberg US Aggregate",
        "duration": 6.2,
        "duration_source": "BlackRock factsheet (updated 2024)",
        "geography_breakdown": {"USA": 1.0},
        "confidence_score": CONFIDENCE_HIGH,
        "source_urls": ["https://www.ishares.com/us/products/239458/"],
    },
    "TLT": {
        "security_name": "iShares 20+ Year Treasury Bond ETF",
        "fee_percent": 0.15,
        "fee_source": "BlackRock factsheet",
        "benchmark": "ICE US Treasury 20+ Year",
        "duration": 16.5,
        "duration_source": "BlackRock factsheet (updated 2024)",
        "geography_breakdown": {"USA": 1.0},
        "confidence_score": CONFIDENCE_HIGH,
        "source_urls": ["https://www.ishares.com/us/products/239454/"],
    },
    # ── Israeli funds (illustrative) ─────────────────────────────────────────
    "1159247": {  # Example Israeli ETF by ISIN fragment
        "security_name": "קרן מחקה ת\"א 125",
        "fee_percent": 0.10,
        "fee_source": "פרוספקטוס הקרן",
        "benchmark": "ת\"א 125",
        "geography_breakdown": {"Israel": 1.0},
        "sector_breakdown": {
            "Financials": 0.28, "Real Estate": 0.20, "Industrials": 0.15,
            "Health Care": 0.12, "Communication Services": 0.10, "Other": 0.15,
        },
        "confidence_score": CONFIDENCE_MEDIUM,
        "source_urls": [],
    },
}

# Name-fragment → lookup key (lower-case)
NAME_FRAGMENT_MAP: dict[str, str] = {
    "s&p 500": "SPY",
    "sp500": "SPY",
    "spdr s&p": "SPY",
    "nasdaq": "QQQ",
    "qqq": "QQQ",
    "msci eafe": "IEFA",
    "eafe": "IEFA",
    "us aggregate": "AGG",
    "aggregate bond": "AGG",
    "treasury 20": "TLT",
    "20+ year": "TLT",
    "total stock market": "VTI",
    "vanguard total": "VTI",
}


class MockResearchProvider(ResearchProvider):
    """
    Returns research data from the in-memory MOCK_DB.
    Provides rule-based estimates for unknowns, always marked as estimated.
    """

    def lookup(
        self,
        *,
        ticker: str = "",
        isin: str = "",
        name: str = "",
        asset_class: str = "",
    ) -> ResearchResult:
        # 1. Exact ticker match
        key = ticker.upper().strip()
        if key and key in MOCK_DB:
            return self._from_db(MOCK_DB[key])

        # 2. ISIN fragment match
        for db_key, data in MOCK_DB.items():
            if isin and isin in db_key:
                return self._from_db(data)

        # 3. Name fragment match
        name_lower = name.lower()
        for fragment, db_key in NAME_FRAGMENT_MAP.items():
            if fragment in name_lower and db_key in MOCK_DB:
                logger.debug(f"MockResearch: matched '{name}' via fragment '{fragment}'")
                return self._from_db(MOCK_DB[db_key])

        # 4. Rule-based estimate (no source)
        return self._estimate(asset_class=asset_class, name=name)

    @staticmethod
    def _from_db(data: dict) -> ResearchResult:
        return ResearchResult(
            security_name=data.get("security_name"),
            fee_percent=data.get("fee_percent"),
            fee_source=data.get("fee_source", ""),
            duration=data.get("duration"),
            duration_source=data.get("duration_source", ""),
            benchmark=data.get("benchmark"),
            sector_breakdown=data.get("sector_breakdown", {}),
            geography_breakdown=data.get("geography_breakdown", {}),
            is_fx_hedged=data.get("is_fx_hedged"),
            is_leveraged=data.get("is_leveraged", False),
            source_urls=data.get("source_urls", []),
            confidence_score=data.get("confidence_score", CONFIDENCE_MEDIUM),
            estimated_fields=[],
        )

    @staticmethod
    def _estimate(asset_class: str, name: str = "") -> ResearchResult:
        """Rule-based estimate for unknowns. Everything marked as estimated."""
        estimated: list[str] = []
        result = ResearchResult(confidence_score=CONFIDENCE_ESTIMATED)

        if asset_class == ASSET_CLASS_BOND:
            name_lower = name.lower()
            if any(k in name_lower for k in ["קצר", "short", "1-3"]):
                result.duration = DURATION_FALLBACK_SHORT_BOND
            elif any(k in name_lower for k in ["ארוך", "long", "10+", "20+"]):
                result.duration = DURATION_FALLBACK_LONG_BOND
            elif any(k in name_lower for k in ["גלובל", "global", "world"]):
                result.duration = DURATION_FALLBACK_GLOBAL_BOND_ETF
            else:
                result.duration = DURATION_FALLBACK_MEDIUM_BOND

            result.duration_source = "Estimated from name/category (rule-based fallback)"
            estimated.append("duration: estimated from asset class category")

        result.estimated_fields = estimated
        result.notes = "No matching data found – values are rule-based estimates only"
        return result
