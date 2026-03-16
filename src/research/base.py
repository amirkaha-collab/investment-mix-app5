"""
Abstract research provider interface.

All research providers must implement ResearchProvider.
Consumers depend on this interface; the concrete implementation
is injected via the factory in src/research/__init__.py.

Fields that a provider CAN fill in for each holding:
  - security_name        (full official name)
  - fee_percent          (annual management fee)
  - fee_source
  - duration             (bond modified/Macaulay duration)
  - duration_source
  - benchmark
  - sector_breakdown     (for funds/ETFs)
  - geography_breakdown  (for funds/ETFs)
  - sector               (for single equities)
  - region / country     (if unknown)
  - is_fx_hedged
  - is_leveraged
  - source_urls
  - confidence_score
  - notes
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ResearchResult:
    """
    Partial update for a HoldingNormalized.

    Only non-None fields should be written back to the holding.
    All estimated fields must be listed in `estimated_fields`.
    """
    security_name: Optional[str] = None
    issuer_name: Optional[str] = None
    fee_percent: Optional[float] = None
    fee_source: str = ""
    duration: Optional[float] = None
    duration_source: str = ""
    benchmark: Optional[str] = None
    sector: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    is_fx_hedged: Optional[bool] = None
    is_leveraged: Optional[bool] = None
    sector_breakdown: dict[str, float] = field(default_factory=dict)
    geography_breakdown: dict[str, float] = field(default_factory=dict)
    source_urls: list[str] = field(default_factory=list)
    confidence_score: float = 0.5
    notes: str = ""
    estimated_fields: list[str] = field(default_factory=list)


class ResearchProvider(ABC):
    """
    Pluggable research data provider.

    To add a new provider:
      1. Subclass ResearchProvider
      2. Implement `lookup`
      3. Register in src/research/__init__.py
    """

    @abstractmethod
    def lookup(
        self,
        *,
        ticker: str = "",
        isin: str = "",
        name: str = "",
        asset_class: str = "",
    ) -> ResearchResult:
        """
        Look up metadata for a single security.

        Callers provide as much identity information as available;
        implementations decide which identifier to prioritise.

        Returns
        -------
        ResearchResult – fields left as None if not found.
        """
        ...

    def batch_lookup(
        self, holdings: list[dict]
    ) -> list[ResearchResult]:
        """
        Default batch implementation – calls lookup() for each holding.
        Override for efficiency (bulk API calls, caching, etc.).
        """
        return [
            self.lookup(
                ticker=h.get("ticker", ""),
                isin=h.get("isin", ""),
                name=h.get("normalized_name", ""),
                asset_class=h.get("asset_class", ""),
            )
            for h in holdings
        ]
