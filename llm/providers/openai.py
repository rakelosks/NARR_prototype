"""
OpenAI LLM provider.
Optional integration for development — not required for deployment.
Includes rate-limit-aware retry with exponential backoff.
"""

import asyncio
import json as json_module
import logging
from typing import Optional

from llm.interface import LLMProvider

logger = logging.getLogger(__name__)

# Module-level throttle shared across all OpenAI provider instances
_openai_throttle = None


def _get_throttle():
    global _openai_throttle
    if _openai_throttle is None:
        from data.ingestion.throttle import Throttle
        _openai_throttle = Throttle(max_concurrent=5, min_interval=0.1)
    return _openai_throttle


class OpenAIProvider(LLMProvider):
    """OpenAI-based LLM provider (optional, for development)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        **kwargs,
    ):
        from openai import AsyncOpenAI

        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var.")
        self.model = model
        self.max_retries = max_retries
        self.client = AsyncOpenAI(api_key=api_key)

    async def _call(self, messages: list[dict], json_mode: bool = False) -> str:
        """Shared call logic with throttling, retry, and error handling."""
        import openai

        kwargs = {"model": self.model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with _get_throttle():
                    response = await self.client.chat.completions.create(**kwargs)
                    return response.choices[0].message.content or ""
            except openai.RateLimitError as e:
                last_exc = e
                if attempt == self.max_retries:
                    break
                delay = 2 ** attempt
                logger.warning(
                    f"OpenAI rate limited, retry {attempt}/{self.max_retries} "
                    f"in {delay}s"
                )
                await asyncio.sleep(delay)
            except openai.APIConnectionError as e:
                last_exc = e
                if attempt == self.max_retries:
                    break
                delay = 2 ** attempt
                logger.warning(
                    f"OpenAI connection error, retry {attempt}/{self.max_retries} "
                    f"in {delay}s: {e}"
                )
                await asyncio.sleep(delay)
            except openai.AuthenticationError:
                raise PermissionError(
                    "OpenAI API key is invalid or expired. Check OPENAI_API_KEY."
                )
            except openai.APITimeoutError:
                raise TimeoutError(
                    f"OpenAI request timed out (model: {self.model})"
                )
            except openai.APIStatusError as e:
                raise RuntimeError(
                    f"OpenAI API error {e.status_code}: {e.message}"
                )

        if isinstance(last_exc, openai.RateLimitError):
            raise RuntimeError(
                f"OpenAI rate limit exceeded after {self.max_retries} retries."
            )
        raise ConnectionError(
            f"Cannot connect to OpenAI API after {self.max_retries} retries."
        )

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a text response via OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self._call(messages)

    async def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response with JSON format mode enabled.
        Used by IntentParser for fast structured output.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self._call(messages, json_mode=True)

    async def generate_structured(self, prompt: str, schema: dict, system_prompt: Optional[str] = None) -> dict:
        """Generate structured output via OpenAI with JSON mode."""
        json_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema}"
        raw = await self.generate_json(json_prompt, system_prompt=system_prompt)
        try:
            return json_module.loads(raw)
        except json_module.JSONDecodeError as e:
            raise ValueError(f"OpenAI returned invalid JSON: {e} — raw: {raw[:200]}")
