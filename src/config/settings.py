"""
Application-wide settings loaded from environment variables / .env file.

On Streamlit Community Cloud there is no .env file.
Secrets are injected via st.secrets (set in the Cloud dashboard).
This module bridges both: it copies st.secrets into os.environ
before Pydantic reads them, so the rest of the codebase is unaware
of which runtime it is running in.

All configuration is centralised here.  Nothing in the codebase should
read os.environ directly – import from this module instead.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_streamlit_secrets() -> None:
    """
    If running inside Streamlit, copy st.secrets into os.environ so that
    pydantic-settings can read them the same way it reads .env variables.
    This is a no-op when running outside Streamlit (e.g. pytest, CLI).
    """
    try:
        import streamlit as st
        # st.secrets raises an error if called outside a Streamlit session
        for key, value in st.secrets.items():
            env_key = key.upper()
            if env_key not in os.environ:
                os.environ[env_key] = str(value)
    except Exception:
        # Not in a Streamlit context, or secrets not configured – safe to ignore
        pass


_load_streamlit_secrets()


class AppEnvironment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ResearchProvider(str, Enum):
    MOCK = "mock"
    WEB = "web"


class StorageBackend(str, Enum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    SUPABASE = "supabase"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ─────────────────────────────────────────────────────────────
    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT
    app_log_level: str = "INFO"
    app_secret_key: str = "change-me"

    # ── Research ─────────────────────────────────────────────────────────────────
    research_provider: ResearchProvider = ResearchProvider.MOCK
    research_request_timeout: int = 15
    research_max_retries: int = 3
    research_retry_delay: float = 2.0
    research_cache_ttl: int = 86_400

    # ── Cache ────────────────────────────────────────────────────────────────────
    cache_dir: Path = PROJECT_ROOT / ".cache" / "research"

    # ── Storage ──────────────────────────────────────────────────────────────────
    storage_backend: StorageBackend = StorageBackend.LOCAL

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_s3_bucket: str = ""
    aws_region: str = "eu-west-1"

    gcs_bucket: str = ""
    google_application_credentials: str = ""

    supabase_url: str = ""
    supabase_key: str = ""

    # ── Locale / FX ──────────────────────────────────────────────────────────────
    default_language: str = "he"
    default_currency: str = "ILS"
    usd_to_ils_rate: float = Field(default=3.75, ge=1.0, le=10.0)

    # ── Templates ────────────────────────────────────────────────────────────────
    @property
    def pptx_template_path(self) -> Path:
        return PROJECT_ROOT / "templates" / "Strategic_Portfolio_Template_AI_Fillable.pptx"

    @field_validator("cache_dir", mode="before")
    @classmethod
    def resolve_cache_dir(cls, v: str | Path) -> Path:
        p = Path(v)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
