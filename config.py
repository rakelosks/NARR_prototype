"""
Application configuration.
Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # reads .env from project root


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

    # Streamlit
    api_base_url: str = "http://localhost:8000"


def load_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        debug=os.getenv("DEBUG", "false").lower() == "true",
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_intent_model=os.getenv("OLLAMA_INTENT_MODEL", "qwen3:4b"),
        ollama_generation_model=os.getenv("OLLAMA_GENERATION_MODEL", "qwen3:4b"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_intent_model=os.getenv("OPENAI_INTENT_MODEL", "gpt-4o-mini"),
        openai_generation_model=os.getenv("OPENAI_GENERATION_MODEL", "gpt-4o"),
        ckan_portal_url=os.getenv("CKAN_PORTAL_URL", "https://gagnagatt.reykjavik.is/en/api/3"),
        ckan_portal_language=os.getenv("CKAN_PORTAL_LANGUAGE", "is"),
        metadata_db_path=os.getenv("METADATA_DB_PATH", "metadata.sqlite"),
        cache_dir=os.getenv("CACHE_DIR", "data/cache/snapshots"),
        cache_ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "24")),
        api_base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
    )


settings = load_settings()
