"""
Ollama LLM provider.
Default deployment option using open-weight models.
"""

import httpx
from typing import Optional
from llm.interface import LLMProvider


class OllamaProvider(LLMProvider):
    """Ollama-based LLM provider for open-weight models."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url
        self.model = model

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a text response via Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()["response"]

    async def generate_structured(self, prompt: str, schema: dict, system_prompt: Optional[str] = None) -> dict:
        """Generate structured output via Ollama with JSON mode."""
        json_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema}"
        payload = {
            "model": self.model,
            "prompt": json_prompt,
            "stream": False,
            "format": "json",
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            import json
            return json.loads(response.json()["response"])
