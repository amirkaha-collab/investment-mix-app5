"""
All hard-coded constants and reference tables live here.
No magic numbers should appear anywhere else in the codebase.
"""

from __future__ import annotations

# ─── Asset classes ──────────────────────────────────────────────────────────────
ASSET_CLASS_EQUITY = "equity"
ASSET_CLASS_BOND = "bond"
ASSET_CLASS_CASH = "cash"
ASSET_CLASS_OTHER = "other"
ASSET_CLASS_ALTERNATIVES = "alternatives"
ASSET_CLASS_REAL_ESTATE = "real_estate"

ASSET_CLASSES = [
    ASSET_CLASS_EQUITY,
    ASSET_CLASS_BOND,
    ASSET_CLASS_CASH,
    ASSET_CLASS_OTHER,
    ASSET_CLASS_ALTERNATIVES,
    ASSET_CLASS_REAL_ESTATE,
]

# ─── Equity regions ─────────────────────────────────────────────────────────────
REGION_ISRAEL = "Israel"
REGION_US = "USA"
REGION_EUROPE = "Europe"
REGION_ASIA_PACIFIC = "Asia-Pacific"
REGION_EMERGING = "Emerging Markets"
REGION_GLOBAL = "Global"
REGION_OTHER = "Other"
REGION_UNKNOWN = "Unknown"

EQUITY_REGIONS = [
    REGION_ISRAEL,
    REGION_US,
    REGION_EUROPE,
    REGION_ASIA_PACIFIC,
    REGION_EMERGING,
    REGION_GLOBAL,
    REGION_OTHER,
]

# ─── Bond linkage types ─────────────────────────────────────────────────────────
BOND_LINKAGE_CPI = "cpi_linked"         # צמוד מדד
BOND_LINKAGE_NOMINAL_ILS = "nominal_ils" # שקלי לא צמוד
BOND_LINKAGE_USD = "usd"
BOND_LINKAGE_FX_OTHER = "fx_other"
BOND_LINKAGE_GLOBAL = "global"
BOND_LINKAGE_UNKNOWN = "unknown"

BOND_LINKAGE_TYPES = [
    BOND_LINKAGE_CPI,
    BOND_LINKAGE_NOMINAL_ILS,
    BOND_LINKAGE_USD,
    BOND_LINKAGE_FX_OTHER,
    BOND_LINKAGE_GLOBAL,
    BOND_LINKAGE_UNKNOWN,
]

# ─── GICS sectors ───────────────────────────────────────────────────────────────
SECTOR_COMMUNICATION = "Communication Services"
SECTOR_CONSUMER_DISC = "Consumer Discretionary"
SECTOR_CONSUMER_STAPLES = "Consumer Staples"
SECTOR_ENERGY = "Energy"
SECTOR_FINANCIALS = "Financials"
SECTOR_HEALTHCARE = "Health Care"
SECTOR_INDUSTRIALS = "Industrials"
SECTOR_IT = "Information Technology"
SECTOR_MATERIALS = "Materials"
SECTOR_REAL_ESTATE = "Real Estate"
SECTOR_UTILITIES = "Utilities"
SECTOR_UNKNOWN = "Unknown"

GICS_SECTORS = [
    SECTOR_COMMUNICATION,
    SECTOR_CONSUMER_DISC,
    SECTOR_CONSUMER_STAPLES,
    SECTOR_ENERGY,
    SECTOR_FINANCIALS,
    SECTOR_HEALTHCARE,
    SECTOR_INDUSTRIALS,
    SECTOR_IT,
    SECTOR_MATERIALS,
    SECTOR_REAL_ESTATE,
    SECTOR_UTILITIES,
]

# ─── Currency codes ─────────────────────────────────────────────────────────────
CURRENCY_ILS = "ILS"
CURRENCY_USD = "USD"
CURRENCY_EUR = "EUR"
CURRENCY_GBP = "GBP"

# ─── Data source confidence ─────────────────────────────────────────────────────
CONFIDENCE_HIGH = 1.0      # Official issuer / factsheet
CONFIDENCE_MEDIUM = 0.75   # Reliable aggregator
CONFIDENCE_LOW = 0.50      # Estimated / inferred
CONFIDENCE_ESTIMATED = 0.25  # Rule-based estimate, user must confirm

# ─── Duration estimation fallbacks (years) ──────────────────────────────────────
DURATION_FALLBACK_SHORT_BOND = 2.0
DURATION_FALLBACK_MEDIUM_BOND = 4.5
DURATION_FALLBACK_LONG_BOND = 8.0
DURATION_FALLBACK_GLOBAL_BOND_ETF = 6.5
DURATION_FALLBACK_CASH = 0.0

# ─── Well-known index sector weights ────────────────────────────────────────────
# Approximate S&P 500 sector weights (as of 2024 – update periodically)
SP500_SECTOR_WEIGHTS: dict[str, float] = {
    SECTOR_IT: 0.31,
    SECTOR_FINANCIALS: 0.13,
    SECTOR_HEALTHCARE: 0.12,
    SECTOR_CONSUMER_DISC: 0.10,
    SECTOR_COMMUNICATION: 0.09,
    SECTOR_INDUSTRIALS: 0.08,
    SECTOR_CONSUMER_STAPLES: 0.06,
    SECTOR_ENERGY: 0.04,
    SECTOR_REAL_ESTATE: 0.02,
    SECTOR_MATERIALS: 0.02,
    SECTOR_UTILITIES: 0.02,
}

# Approximate MSCI World sector weights (2024)
MSCI_WORLD_SECTOR_WEIGHTS: dict[str, float] = {
    SECTOR_IT: 0.24,
    SECTOR_FINANCIALS: 0.16,
    SECTOR_HEALTHCARE: 0.11,
    SECTOR_INDUSTRIALS: 0.11,
    SECTOR_CONSUMER_DISC: 0.10,
    SECTOR_COMMUNICATION: 0.08,
    SECTOR_CONSUMER_STAPLES: 0.07,
    SECTOR_ENERGY: 0.05,
    SECTOR_MATERIALS: 0.04,
    SECTOR_REAL_ESTATE: 0.02,
    SECTOR_UTILITIES: 0.02,
}

# ─── Well-known tickers → metadata (minimal fallback) ───────────────────────────
KNOWN_TICKERS: dict[str, dict] = {
    "SPY": {
        "name": "SPDR S&P 500 ETF Trust",
        "asset_class": ASSET_CLASS_EQUITY,
        "region": REGION_US,
        "fee_percent": 0.0945,
        "is_etf": True,
        "sector_weights": SP500_SECTOR_WEIGHTS,
        "benchmark": "S&P 500",
    },
    "QQQ": {
        "name": "Invesco QQQ Trust",
        "asset_class": ASSET_CLASS_EQUITY,
        "region": REGION_US,
        "fee_percent": 0.20,
        "is_etf": True,
        "benchmark": "NASDAQ-100",
    },
    "VTI": {
        "name": "Vanguard Total Stock Market ETF",
        "asset_class": ASSET_CLASS_EQUITY,
        "region": REGION_US,
        "fee_percent": 0.03,
        "is_etf": True,
        "benchmark": "CRSP US Total Market",
    },
    "IEFA": {
        "name": "iShares Core MSCI EAFE ETF",
        "asset_class": ASSET_CLASS_EQUITY,
        "region": REGION_EUROPE,
        "fee_percent": 0.07,
        "is_etf": True,
        "benchmark": "MSCI EAFE",
    },
    "AGG": {
        "name": "iShares Core U.S. Aggregate Bond ETF",
        "asset_class": ASSET_CLASS_BOND,
        "region": REGION_US,
        "fee_percent": 0.03,
        "is_etf": True,
        "bond_linkage_type": BOND_LINKAGE_USD,
        "duration": 6.2,
        "benchmark": "Bloomberg US Aggregate",
    },
    "TLT": {
        "name": "iShares 20+ Year Treasury Bond ETF",
        "asset_class": ASSET_CLASS_BOND,
        "region": REGION_US,
        "fee_percent": 0.15,
        "is_etf": True,
        "bond_linkage_type": BOND_LINKAGE_USD,
        "duration": 16.5,
        "benchmark": "ICE US Treasury 20+ Year",
    },
}

# ─── QA thresholds ──────────────────────────────────────────────────────────────
QA_WEIGHT_SUM_TOLERANCE = 0.005   # 0.5% rounding tolerance for weight sums
QA_MAX_SINGLE_HOLDING_WEIGHT = 0.40  # Flag if single holding > 40%
QA_CONCENTRATION_WARN_TOP10 = 0.80   # Warn if top-10 holdings > 80%

# ─── PPTX layout constants ──────────────────────────────────────────────────────
PPTX_SLIDE_WIDTH_EMU = 9_144_000   # 10 inches
PPTX_SLIDE_HEIGHT_EMU = 5_143_500  # 7.5 inches
PPTX_CHART_IMAGE_DPI = 150

# ─── Disclaimer ─────────────────────────────────────────────────────────────────
LEGAL_DISCLAIMER = (
    "המערכת מספקת ניתוח מידע בלבד ואינה מהווה ייעוץ השקעות, "
    "שיווק השקעות או המלצה. כל ההחלטות הפיננסיות הן באחריות המשתמש בלבד."
)
LEGAL_DISCLAIMER_EN = (
    "This system provides informational analysis only and does not constitute "
    "investment advice, investment marketing, or a recommendation. "
    "All financial decisions are the sole responsibility of the user."
)
