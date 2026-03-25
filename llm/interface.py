"""
Provider-agnostic LLM interface.
Defines the abstract interface for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a text response from the LLM."""
        ...

    @abstractmethod
    async def generate_structured(self, prompt: str, schema: dict, system_prompt: Optional[str] = None) -> dict:
        """Generate a structured (JSON) response validated against a schema."""
        ...


def get_provider(provider_name: str = "ollama", **kwargs) -> LLMProvider:
    """Factory function to get an LLM provider instance."""
    if provider_name == "ollama":
        from llm.providers.ollama import OllamaProvider
        return OllamaProvider(**kwargs)
    elif provider_name == "openai":
        from llm.providers.openai import OpenAIProvider
        return OpenAIProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def get_providers() -> tuple[LLMProvider, LLMProvider]:
    """
    Get separate LLM providers for intent parsing and narrative generation.

    Returns:
        (intent_provider, generation_provider)
    """
    from config import settings

    provider = settings.llm_provider

    if provider == "openai":
        intent = get_provider(
            provider,
            api_key=settings.openai_api_key,
            model=settings.openai_intent_model,
        )
        generation = get_provider(
            provider,
            api_key=settings.openai_api_key,
            model=settings.openai_generation_model,
        )
    else:
        intent = get_provider(
            provider,
            base_url=settings.ollama_base_url,
            model=settings.ollama_intent_model,
            think=settings.ollama_intent_think,
            timeout=settings.ollama_timeout,
        )
        generation = get_provider(
            provider,
            base_url=settings.ollama_base_url,
            model=settings.ollama_generation_model,
            think=settings.ollama_generation_think,
            timeout=settings.ollama_timeout,
        )
    return intent, generation
