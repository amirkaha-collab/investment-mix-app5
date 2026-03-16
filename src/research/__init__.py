"""
Research module.

Factory selects the provider based on settings.
EnrichmentService applies research results back to holdings.
"""

from __future__ import annotations

import logging

from src.config.settings import get_settings, ResearchProvider as ProviderEnum
from src.domain.models import HoldingNormalized
from .base import ResearchProvider, ResearchResult
from .mock_provider import MockResearchProvider

logger = logging.getLogger(__name__)


def get_research_provider() -> ResearchProvider:
    """Return the configured research provider."""
    settings = get_settings()
    if settings.research_provider == ProviderEnum.WEB:
        from .web_provider import WebResearchProvider
        return WebResearchProvider()
    return MockResearchProvider()


class EnrichmentService:
    """
    Applies ResearchResult data to HoldingNormalized objects in-place.

    Only overwrites a field if:
      - the holding's current value is empty / None
      - OR the research confidence is higher

    Estimated fields are always tracked.
    """

    def __init__(self, provider: ResearchProvider | None = None) -> None:
        self._provider = provider or get_research_provider()

    def enrich(self, holdings: list[HoldingNormalized]) -> list[str]:
        """
        Enrich all holdings.  Returns a list of warning strings.
        """
        warnings: list[str] = []
        total = len(holdings)
        for i, h in enumerate(holdings, start=1):
            logger.info(f"Researching holding {i}/{total}: {h.normalized_name}")
            try:
                result = self._provider.lookup(
                    ticker=h.ticker,
                    isin=h.isin,
                    name=h.normalized_name,
                    asset_class=h.asset_class,
                )
                self._apply(h, result)
            except Exception as exc:
                warnings.append(f"{h.normalized_name}: research error – {exc}")
        return warnings

    @staticmethod
    def _apply(h: HoldingNormalized, r: ResearchResult) -> None:
        """Merge ResearchResult into holding (non-destructive)."""
        if r.security_name and not h.security_name:
            h.security_name = r.security_name

        if r.issuer_name and not h.issuer_name:
            h.issuer_name = r.issuer_name

        if r.fee_percent is not None and h.fee_percent is None:
            h.fee_percent = r.fee_percent
            h.fee_source = r.fee_source
            if "estimated" in r.estimated_fields or r.confidence_score < 0.5:
                h.mark_estimated("fee_percent", r.fee_source or "rule-based estimate")

        if r.duration is not None and h.duration is None:
            h.duration = r.duration
            h.duration_source = r.duration_source
            if "duration" in " ".join(r.estimated_fields):
                h.mark_estimated("duration", r.duration_source or "rule-based estimate")

        if r.benchmark and not h.benchmark:
            h.benchmark = r.benchmark

        if r.sector and not h.sector:
            h.sector = r.sector

        if r.country and not h.country:
            h.country = r.country

        if r.region and h.region in ("Unknown", ""):
            h.region = r.region

        if r.is_fx_hedged is not None and h.is_fx_hedged is None:
            h.is_fx_hedged = r.is_fx_hedged

        if r.is_leveraged is not None:
            h.is_leveraged = r.is_leveraged

        if r.sector_breakdown:
            h.sector_breakdown = r.sector_breakdown

        if r.geography_breakdown:
            h.geography_breakdown = r.geography_breakdown

        if r.source_urls:
            h.source_urls = list(set(h.source_urls + r.source_urls))

        # Update confidence (take minimum – weakest link)
        h.confidence_score = min(h.confidence_score, r.confidence_score)

        if r.notes:
            h.notes = (h.notes + " " + r.notes).strip()

        for ef in r.estimated_fields:
            if ef not in h.estimated_fields:
                h.estimated_fields.append(ef)


__all__ = [
    "ResearchProvider",
    "ResearchResult",
    "MockResearchProvider",
    "EnrichmentService",
    "get_research_provider",
]
