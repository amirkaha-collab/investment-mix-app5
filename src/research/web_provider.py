"""
Web-based research provider.

Uses publicly accessible web pages (no aggressive scraping).
Implements caching, timeout, and retry via tenacity.

Architecture notes:
- This provider is intentionally conservative: it only reads pages that are
  clearly designed for public consumption (ETF product pages, Yahoo Finance
  summary pages, etc.).
- The layer is abstracted so it can be replaced wholesale with a licensed
  market-data API without touching any calling code.
- All field values that come from heuristic scraping (vs. a structured API)
  are marked as confidence=MEDIUM.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tenacity import (
    RetryError,
    retry,
    stop_after_attempt,
    wait_fixed,
)

from src.config.constants import CONFIDENCE_LOW, CONFIDENCE_MEDIUM
from src.config.settings import get_settings
from .base import ResearchProvider, ResearchResult
from .mock_provider import MockResearchProvider  # fallback

logger = logging.getLogger(__name__)

settings = get_settings()


class WebResearchProvider(ResearchProvider):
    """
    Fetches security metadata from public web sources.

    Priority:
      1. Yahoo Finance summary page (fee, name, category)
      2. ETF.com detail page (for ETFs – expense ratio, holdings)
      3. Morningstar (fees)
      4. Fallback to MockResearchProvider

    All results are cached to disk for `settings.research_cache_ttl` seconds.
    """

    YAHOO_BASE = "https://finance.yahoo.com/quote/{ticker}"
    ETFCOM_BASE = "https://www.etf.com/{ticker}"

    def __init__(self) -> None:
        self._cache_dir = settings.cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._fallback = MockResearchProvider()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; PortfolioAnalyzer/1.0; "
                    "+https://github.com/yourorg/portfolio-analyzer)"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def lookup(
        self,
        *,
        ticker: str = "",
        isin: str = "",
        name: str = "",
        asset_class: str = "",
    ) -> ResearchResult:
        # 1. Try mock/static database first (fastest, most reliable)
        mock_result = self._fallback.lookup(
            ticker=ticker, isin=isin, name=name, asset_class=asset_class
        )
        if mock_result.confidence_score >= CONFIDENCE_MEDIUM:
            return mock_result

        # 2. Try Yahoo Finance for basic metadata
        if ticker:
            cached = self._cache_get(f"yahoo:{ticker}")
            if cached:
                return cached

            try:
                result = self._fetch_yahoo(ticker)
                if result:
                    self._cache_set(f"yahoo:{ticker}", result)
                    return result
            except Exception as exc:
                logger.warning(f"Yahoo fetch failed for {ticker}: {exc}")

        # 3. Fall back to mock (which returns rule-based estimates)
        return mock_result

    # ──────────────────────────────────────────────────────────────────────────
    # Fetchers
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_yahoo(self, ticker: str) -> Optional[ResearchResult]:
        """
        Fetch summary data from Yahoo Finance.
        Returns None on failure so caller can try other sources.
        """
        url = self.YAHOO_BASE.format(ticker=ticker.upper())
        try:
            html = self._get_with_retry(url)
        except RetryError:
            return None

        soup = BeautifulSoup(html, "lxml")
        result = ResearchResult(confidence_score=CONFIDENCE_LOW)
        result.source_urls.append(url)

        # Security name from page title
        title_tag = soup.find("h1")
        if title_tag:
            result.security_name = title_tag.get_text(strip=True)

        # Expense ratio (often in the summary table for ETFs/funds)
        summary_rows = soup.select("td[data-test]")
        for td in summary_rows:
            label = td.get("data-test", "")
            if "EXPENSE_RATIO" in label or "expense" in label.lower():
                try:
                    val = float(td.get_text(strip=True).replace("%", ""))
                    result.fee_percent = val
                    result.fee_source = "Yahoo Finance"
                except ValueError:
                    pass

        result.confidence_score = CONFIDENCE_MEDIUM if result.fee_percent else CONFIDENCE_LOW
        return result if result.security_name else None

    # ──────────────────────────────────────────────────────────────────────────
    # HTTP helper
    # ──────────────────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(settings.research_retry_delay),
        reraise=True,
    )
    def _get_with_retry(self, url: str) -> str:
        resp = self._session.get(url, timeout=settings.research_request_timeout)
        resp.raise_for_status()
        return resp.text

    # ──────────────────────────────────────────────────────────────────────────
    # Disk cache
    # ──────────────────────────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{h}.json"

    def _cache_get(self, key: str) -> Optional[ResearchResult]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > settings.research_cache_ttl:
            path.unlink(missing_ok=True)
            return None
        try:
            data = json.loads(path.read_text())
            return ResearchResult(**data)
        except Exception:
            return None

    def _cache_set(self, key: str, result: ResearchResult) -> None:
        path = self._cache_path(key)
        try:
            import dataclasses
            path.write_text(json.dumps(dataclasses.asdict(result)))
        except Exception as exc:
            logger.warning(f"Cache write failed for {key}: {exc}")
