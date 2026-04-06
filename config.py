"""
Application configuration.
Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # reads .env from project root


def _get_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with common truthy values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Application settings."""

    # API
    app_name: str = "Smart City Narrative Visualization Platform"
    debug: bool = False

    # LLM
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_intent_model: str = "qwen3:4b"
    ollama_generation_model: str = "qwen3:4b"  # 4b with JSON mode — fast enough for 16GB
    ollama_intent_think: bool = False      # thinking off for fast intent parsing
    ollama_generation_think: bool = True   # thinking on for richer narratives
    ollama_timeout: float = 180.0          # seconds per request
    openai_api_key: str = ""
    openai_intent_model: str = "gpt-4o-mini"
    openai_generation_model: str = "gpt-4o"

    # CKAN portal
    ckan_portal_url: str = "https://gagnagatt.reykjavik.is/en/api/3"
    ckan_portal_language: str = "is"  # ISO 639-1 code for portal metadata language

    # Storage
    metadata_db_path: str = "metadata.sqlite"
    cache_dir: str = "data/cache/snapshots"
    cache_ttl_hours: int = 24

    # API security
    narr_api_key: str = ""           # empty = auth disabled (local dev)
    rate_limit_rpm: int = 60         # requests per minute per client (0 = disabled)
    trust_proxy_headers: bool = False  # trust X-Forwarded-For only behind trusted proxy

    # Streamlit
    api_base_url: str = "http://localhost:8000"


def load_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        debug=_get_bool("DEBUG", False),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_intent_model=os.getenv("OLLAMA_INTENT_MODEL", "qwen3:4b"),
        ollama_generation_model=os.getenv("OLLAMA_GENERATION_MODEL", "qwen3:4b"),
        ollama_intent_think=_get_bool("OLLAMA_INTENT_THINK", False),
        ollama_generation_think=_get_bool("OLLAMA_GENERATION_THINK", True),
        ollama_timeout=float(os.getenv("OLLAMA_TIMEOUT", "180")),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_intent_model=os.getenv("OPENAI_INTENT_MODEL", "gpt-4o-mini"),
        openai_generation_model=os.getenv("OPENAI_GENERATION_MODEL", "gpt-4o"),
        ckan_portal_url=os.getenv("CKAN_PORTAL_URL", "https://gagnagatt.reykjavik.is/en/api/3"),
        ckan_portal_language=os.getenv("CKAN_PORTAL_LANGUAGE", "is"),
        metadata_db_path=os.getenv("METADATA_DB_PATH", "metadata.sqlite"),
        cache_dir=os.getenv("CACHE_DIR", "data/cache/snapshots"),
        cache_ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "24")),
        narr_api_key=os.getenv("NARR_API_KEY", ""),
        rate_limit_rpm=int(os.getenv("RATE_LIMIT_RPM", "60")),
        trust_proxy_headers=_get_bool("TRUST_PROXY_HEADERS", False),
        api_base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
    )


settings = load_settings()
