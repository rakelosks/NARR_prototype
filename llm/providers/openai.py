"""
OpenAI LLM provider.
Optional integration for development — not required for deployment.
"""

from typing import Optional
from llm.interface import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI-based LLM provider (optional, for development)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.model = model
        # TODO: Initialize OpenAI client
        raise NotImplementedError("OpenAI provider not yet implemented")

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError

    async def generate_structured(self, prompt: str, schema: dict, system_prompt: Optional[str] = None) -> dict:
        raise NotImplementedError
