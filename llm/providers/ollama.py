"""
Ollama LLM provider.
Default deployment option using open-weight models.

Handles qwen3 thinking-model quirks:
- When format="json" is used, qwen3 may place structured output
  in the 'thinking' field instead of 'response'. The provider
  checks both fields automatically.
"""

import json as json_module
import logging
import re
import httpx
from typing import Optional
from llm.interface import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama-based LLM provider for open-weight models."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        think: bool = True,
        timeout: float = 300.0,
    ):
        self.base_url = base_url
        self.model = model
        self.think = think
        self.timeout = timeout

    def _extract_response(self, data: dict) -> str:
        """
        Extract the text response from Ollama's JSON output.

        qwen3 thinking models sometimes place output in the 'thinking'
        field when 'response' is empty (especially with format=json).
        """
        response = data.get("response", "")
        if response.strip():
            return response

        # Fallback: qwen3 may put structured output in 'thinking'
        thinking = data.get("thinking", "")
        if thinking.strip():
            logger.debug("Response empty, using 'thinking' field (qwen3 quirk)")
            return thinking

        return response

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a text response via Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if self.think is not None:
            payload["think"] = self.think

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return self._extract_response(response.json())

    async def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response with Ollama's JSON format mode enabled.

        Faster than generate() for models with thinking (e.g. qwen3)
        because the JSON constraint bypasses extended chain-of-thought.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return self._extract_response(response.json())

    async def generate_structured(self, prompt: str, schema: dict, system_prompt: Optional[str] = None) -> dict:
        """Generate structured output via Ollama with JSON mode."""
        json_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema}"
        raw = await self.generate_json(json_prompt, system_prompt=system_prompt)
        return json_module.loads(raw)
