"""
Core domain models for the Portfolio Analyzer.

All data flowing through the system is validated by these Pydantic v2 models.
Estimated fields are always explicitly labelled; nothing is silently guessed.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class Language(str, Enum):
    HEBREW = "he"
    ENGLISH = "en"


class PresentationStyle(str, Enum):
    FAMILY_OFFICE = "family_office"
    INSTITUTIONAL = "institutional"
    RETAIL = "retail"


class ConfidenceLevel(str, Enum):
    HIGH = "high"          # Official source
    MEDIUM = "medium"      # Reliable aggregator
    LOW = "low"            # Estimated / inferred
    ESTIMATED = "estimated"  # Rule-based assumption


# ─────────────────────────────────────────────────────────────────────────────
# Holding model
# ─────────────────────────────────────────────────────────────────────────────

class HoldingNormalized(BaseModel):
    """
    A single normalized holding in the portfolio.

    Fields suffixed with `_source` describe where the value came from.
    Fields in `estimated_fields` are assumptions, not facts.
    """

    # Identity
    row_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    raw_name: str = Field(..., description="Name as it appeared in the source document")
    normalized_name: str = Field(..., description="Cleaned, de-duplicated name")
    security_name: str = Field("", description="Full official security name")
    issuer_name: str = Field("", description="Issuer or fund manager")
    ticker: str = Field("", description="Bloomberg/exchange ticker if known")
    isin: str = Field("", description="ISIN code if known")

    # Classification
    asset_class: str = Field(..., description="equity | bond | cash | other | alternatives | real_estate")
    asset_subclass: str = Field("", description="e.g. large_cap_equity, government_bond")

    # Quantities & value
    quantity: Optional[float] = Field(None, description="Units / nominal")
    market_value: float = Field(..., description="Market value in original currency", ge=0)
    market_value_ils: float = Field(..., description="Market value in ILS (converted)", ge=0)
    currency: str = Field("ILS")
    weight_in_portfolio: float = Field(0.0, ge=0.0, le=1.0, description="Fraction of total portfolio")

    # Geography
    country: str = Field("", description="Primary country")
    region: str = Field("Unknown", description="Broad region (Israel / USA / Europe / …)")

    # Sector
    sector: str = Field("", description="GICS sector for equities")

    # Benchmark
    benchmark: str = Field("", description="Benchmark index if applicable")

    # Type flags
    is_fund: bool = False
    is_etf: bool = False
    is_bond: bool = False
    is_equity: bool = False
    is_cash: bool = False

    # FX
    is_fx_hedged: Optional[bool] = Field(None, description="True = hedged to ILS; None = unknown")
    is_leveraged: bool = False

    # Bond specifics
    bond_linkage_type: str = Field("", description="cpi_linked | nominal_ils | usd | fx_other | global | unknown")
    duration: Optional[float] = Field(None, description="Macaulay / modified duration (years)", ge=0)
    duration_source: str = Field("", description="Where duration came from")

    # Fees
    fee_percent: Optional[float] = Field(None, description="Annual management fee %", ge=0, le=5)
    fee_source: str = Field("", description="Source of fee data")

    # Fund-level breakdown (for ETFs/funds with known composition)
    geography_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="region -> fraction (must sum to 1.0 if provided)"
    )
    sector_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="sector -> fraction (must sum to 1.0 if provided)"
    )

    # Provenance
    source_urls: list[str] = Field(default_factory=list)
    confidence_score: float = Field(1.0, ge=0.0, le=1.0)
    notes: str = Field("")

    # Traceability
    estimated_fields: list[str] = Field(
        default_factory=list,
        description="Names of fields that were estimated, not retrieved from official sources"
    )

    @field_validator("market_value", "market_value_ils", mode="before")
    @classmethod
    def coerce_numeric(cls, v: Any) -> float:
        if isinstance(v, str):
            v = v.replace(",", "").replace("₪", "").replace("$", "").strip()
        return float(v)

    @model_validator(mode="after")
    def set_type_flags(self) -> "HoldingNormalized":
        """Ensure type flags are consistent with asset_class."""
        ac = self.asset_class.lower()
        self.is_equity = ac == "equity"
        self.is_bond = ac == "bond"
        self.is_cash = ac == "cash"
        return self

    def mark_estimated(self, field_name: str, reason: str = "") -> None:
        """Mark a field as estimated and append to estimated_fields list."""
        tag = f"{field_name}: {reason}" if reason else field_name
        if tag not in self.estimated_fields:
            self.estimated_fields.append(tag)
        if field_name not in self.notes:
            self.notes += f" [ESTIMATED: {field_name}]"


# ─────────────────────────────────────────────────────────────────────────────
# User preferences gathered via clarification questions
# ─────────────────────────────────────────────────────────────────────────────

class UserAnalysisPreferences(BaseModel):
    """
    Answers to the clarification questions asked before analysis.
    All monetary values are in percent unless noted.
    """
    include_cash_in_allocation: bool = Field(
        True, description="Include cash holdings in asset allocation table"
    )
    portfolio_manager_fee_percent: Optional[float] = Field(
        None, description="Annual fee charged by the portfolio manager (%)", ge=0, le=5
    )
    manager_fee_is_assumption: bool = Field(
        False, description="Is the above fee an assumption rather than a confirmed value?"
    )
    classify_global_usd_bond_as_us_exposure: bool = Field(
        False,
        description=(
            "When True, global USD-denominated bond funds are counted as US exposure "
            "in the broad_us_exposure metric."
        ),
    )
    compute_extended_duration_with_estimates: bool = Field(
        False,
        description="When True, estimated durations are included in weighted duration calculation."
    )
    presentation_style: PresentationStyle = PresentationStyle.FAMILY_OFFICE
    language: Language = Language.HEBREW
    rtl_mode: bool = True
    report_title: str = "ניתוח תיק השקעות"
    client_name: str = ""
    report_date: str = ""  # ISO date string, filled automatically if empty


# ─────────────────────────────────────────────────────────────────────────────
# Per-row analysis outputs (intermediate)
# ─────────────────────────────────────────────────────────────────────────────

class AssetAllocationRow(BaseModel):
    asset_class: str
    market_value_ils: float
    weight: float
    is_estimated: bool = False


class EquityGeographyRow(BaseModel):
    region: str
    market_value_ils: float
    weight_in_equities: float
    weight_in_portfolio: float
    is_estimated: bool = False


class SectorAllocationRow(BaseModel):
    sector: str
    market_value_ils: float
    weight_in_equities: float
    source_note: str = ""
    is_estimated: bool = False


class BondBreakdownRow(BaseModel):
    linkage_type: str
    market_value_ils: float
    weight_in_bonds: float
    weight_in_portfolio: float
    is_estimated: bool = False


class DurationRow(BaseModel):
    row_id: str
    name: str
    bond_linkage_type: str
    market_value_ils: float
    duration: Optional[float]
    duration_source: str
    is_estimated: bool
    weighted_contribution: Optional[float]  # duration * weight


class FundCostRow(BaseModel):
    row_id: str
    name: str
    asset_class: str
    market_value_ils: float
    weight_in_portfolio: float
    fee_percent: Optional[float]
    fee_source: str
    is_estimated: bool
    annual_cost_ils: Optional[float]


class FXExposureRow(BaseModel):
    currency: str
    market_value_ils: float
    weight: float
    is_hedged: Optional[bool]
    hedging_note: str = ""


class ConcentrationRow(BaseModel):
    rank: int
    row_id: str
    name: str
    market_value_ils: float
    weight: float


class AssumptionRow(BaseModel):
    field: str
    holding_id: str
    holding_name: str
    assumed_value: str
    reason: str
    confidence: ConfidenceLevel
    source: str = ""


class DataQualityNote(BaseModel):
    category: str   # missing_data | weak_source | classification_ambiguity
    holding_id: str
    holding_name: str
    description: str
    recommendation: str = ""


class USExposureSummary(BaseModel):
    conservative_us_value_ils: float
    conservative_us_weight: float
    broad_us_value_ils: float
    broad_us_weight: float
    methodology_note: str
    is_estimated: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Final aggregated analysis output
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisOutputs(BaseModel):
    """
    Complete analysis result.  Every number here is traceable to source holdings.
    """
    # Portfolio totals
    total_portfolio_value_ils: float
    holdings_count: int
    analysis_currency: str = "ILS"

    # Tables
    asset_allocation: list[AssetAllocationRow] = Field(default_factory=list)
    equity_geography: list[EquityGeographyRow] = Field(default_factory=list)
    us_exposure: Optional[USExposureSummary] = None
    sector_allocation: list[SectorAllocationRow] = Field(default_factory=list)
    bond_breakdown: list[BondBreakdownRow] = Field(default_factory=list)
    duration_table: list[DurationRow] = Field(default_factory=list)
    fund_cost_table: list[FundCostRow] = Field(default_factory=list)
    fx_exposure: list[FXExposureRow] = Field(default_factory=list)
    top_holdings: list[ConcentrationRow] = Field(default_factory=list)

    # Summary scalars
    conservative_weighted_duration: Optional[float] = None
    extended_weighted_duration: Optional[float] = None
    weighted_fund_cost_on_funds: Optional[float] = None     # % — denominator = funds only
    effective_fund_cost_on_total_portfolio: Optional[float] = None  # % — denominator = total
    total_cost_percent: Optional[float] = None               # fund cost + PM fee (if provided)
    total_cost_is_assumption: bool = False

    # Concentration
    top5_concentration: Optional[float] = None
    top10_concentration: Optional[float] = None
    max_single_holding_weight: Optional[float] = None
    concentration_warnings: list[str] = Field(default_factory=list)

    # Metadata / transparency
    assumptions: list[AssumptionRow] = Field(default_factory=list)
    data_quality_notes: list[DataQualityNote] = Field(default_factory=list)
    methodology_notes: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    qa_warnings: list[str] = Field(default_factory=list)
    qa_errors: list[str] = Field(default_factory=list)

    # Preferences echo (for reproducibility)
    preferences: Optional[UserAnalysisPreferences] = None
