"""
Application configuration.
Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass


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
    ollama_generation_model: str = "qwen3:8b"
    ollama_intent_think: bool = False      # thinking off for fast intent parsing
    ollama_generation_think: bool = True   # thinking on for richer narratives
    ollama_timeout: float = 300.0          # seconds per request
    openai_api_key: str = ""

    # Storage
    metadata_db_path: str = "metadata.sqlite"
    cache_dir: str = "data/cache/snapshots"

    # Streamlit
    api_base_url: str = "http://localhost:8000"


def load_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        debug=os.getenv("DEBUG", "false").lower() == "true",
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_intent_model=os.getenv("OLLAMA_INTENT_MODEL", "qwen3:4b"),
        ollama_generation_model=os.getenv("OLLAMA_GENERATION_MODEL", "qwen3:8b"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        metadata_db_path=os.getenv("METADATA_DB_PATH", "metadata.sqlite"),
        cache_dir=os.getenv("CACHE_DIR", "data/cache/snapshots"),
        api_base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
    )


settings = load_settings()
