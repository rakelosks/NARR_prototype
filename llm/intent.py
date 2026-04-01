"""
LLM intent parsing.
Parses a user's natural language prompt into a structured intent
that drives dataset selection, analysis, and narrative generation.
"""

import json
import logging
import re
from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent models
# ---------------------------------------------------------------------------

class AnalysisType(str, Enum):
    """Types of analysis the user might request."""
    TREND = "trend"              # "How has X changed over time?"
    COMPARISON = "comparison"    # "Compare X across categories"
    DISTRIBUTION = "distribution"  # "What's the distribution of X?"
    SUMMARY = "summary"          # "Summarize this dataset"
    SPATIAL = "spatial"          # "Where are the X located?"
    CORRELATION = "correlation"  # "Is there a relationship between X and Y?"


class AudienceLevel(str, Enum):
    """Target audience for the narrative."""
    CITIZEN = "citizen"          # General public, non-technical
    POLICY = "policy"            # Policymakers, officials
    TECHNICAL = "technical"      # Data analysts, developers


class UserIntent(BaseModel):
    """
    Structured representation of a user's data exploration intent.
    Parsed from natural language by the LLM.
    """
    dataset_query: str = Field(
        description="Keywords or description of the dataset the user wants to explore.",
    )
    analysis_type: AnalysisType = Field(
        default=AnalysisType.SUMMARY,
        description="The type of analysis requested.",
    )
    focus_columns: list[str] = Field(
        default_factory=list,
        description="Specific columns or fields the user mentioned.",
    )
    time_range: Optional[str] = Field(
        default=None,
        description="Time period if mentioned (e.g. '2023', 'last 5 years').",
    )
    geographic_focus: Optional[str] = Field(
        default=None,
        description="Geographic area if mentioned (e.g. 'downtown', 'Vesturbær').",
    )
    audience: AudienceLevel = Field(
        default=AudienceLevel.CITIZEN,
        description="Target audience level.",
    )
    language: str = Field(
        default="en",
        description="Language for the narrative output (ISO 639-1 code).",
    )
    dataset_query_local: Optional[str] = Field(
        default=None,
        description="The dataset_query translated to the portal's language for catalog search.",
    )
    custom_question: Optional[str] = Field(
        default=None,
        description="The user's original question, preserved verbatim.",
    )

    @field_validator("dataset_query")
    @classmethod
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("dataset_query cannot be empty")
        return v.strip()


class IntentParseResult(BaseModel):
    """Result of parsing a user prompt."""
    success: bool
    intent: Optional[UserIntent] = None
    raw_llm_output: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Intent parsing prompt
# ---------------------------------------------------------------------------

INTENT_SYSTEM_PROMPT = """You are an intent parser for a city open data platform.
Your job is to parse a user's natural language question into a structured JSON intent.

You must respond with ONLY valid JSON matching this exact schema, no other text:

{{
    "dataset_query": "keywords describing the dataset in the user's language",
    "dataset_query_local": "the same keywords translated to {portal_language}",
    "analysis_type": "trend|comparison|distribution|summary|spatial|correlation",
    "focus_columns": ["column1", "column2"],
    "time_range": null,
    "geographic_focus": null,
    "audience": "citizen|policy|technical",
    "language": "en",
    "custom_question": "the user's original question"
}}

Notes on nullable fields:
- time_range: set to a string like "2020-2023" or "last 5 years" if the user mentions a time period, otherwise null
- geographic_focus: set to a string like "downtown" or "Vesturbær" if the user mentions an area, otherwise null

Rules:
- dataset_query: Extract the core data topic (e.g. "budget", "traffic", "air quality")
- dataset_query_local: Translate the dataset_query keywords into {portal_language}.
  This is used to search the city's open data catalog which is in {portal_language}.
  Use natural terms a data publisher would use, not literal translations.
  Example: if the user asks about "waste collection" and portal language is Icelandic,
  use "úrgangur sorphirða" (not a word-for-word translation).
  If the user's language already matches {portal_language}, copy dataset_query as-is.
- analysis_type: Infer from the question:
  - "trend" for questions about change over time
  - "comparison" for questions comparing categories
  - "distribution" for questions about spread or breakdown
  - "summary" for general "tell me about" questions
  - "spatial" for questions about locations or areas
  - "correlation" for questions about relationships between variables
- focus_columns: Extract any specific field names or metrics mentioned
- audience: Default to "citizen" unless the question uses technical language
- language: Detect from the question language, default "en"
- Respond ONLY with the JSON object, no markdown, no explanation"""


# Map ISO 639-1 codes to language names for the prompt
_LANGUAGE_NAMES: dict[str, str] = {
    "is": "Icelandic", "en": "English", "es": "Spanish", "fr": "French",
    "de": "German", "pt": "Portuguese", "it": "Italian", "nl": "Dutch",
    "sv": "Swedish", "no": "Norwegian", "da": "Danish", "fi": "Finnish",
    "pl": "Polish", "cs": "Czech", "ro": "Romanian", "hu": "Hungarian",
    "el": "Greek", "tr": "Turkish", "ar": "Arabic", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "hi": "Hindi", "th": "Thai",
}


def build_intent_prompt(user_message: str, portal_language: str = "en") -> str:
    """Build the prompt for intent parsing."""
    return f"Parse this user question into a structured intent:\n\n\"{user_message}\""


def get_intent_system_prompt(portal_language: str = "en") -> str:
    """Build the intent system prompt with the portal language filled in."""
    lang_name = _LANGUAGE_NAMES.get(portal_language, portal_language)
    return INTENT_SYSTEM_PROMPT.format(portal_language=lang_name)


# ---------------------------------------------------------------------------
# Intent parser
# ---------------------------------------------------------------------------

class IntentParser:
    """
    Parses user prompts into structured intents using an LLM.
    """

    def __init__(self, llm_provider):
        """
        Args:
            llm_provider: An LLMProvider instance (Ollama or OpenAI).
        """
        self.llm = llm_provider

    async def parse(self, user_message: str, portal_language: str = "en") -> IntentParseResult:
        """
        Parse a user's natural language message into a structured intent.

        Args:
            user_message: The user's question or request.
            portal_language: ISO 639-1 code for the portal's catalog language.
                The parser will translate dataset_query into this language.

        Returns:
            IntentParseResult with the parsed intent or error.
        """
        logger.info(f"Parsing intent from: '{user_message[:80]}...' (portal_lang={portal_language})")

        prompt = build_intent_prompt(user_message, portal_language)
        system_prompt = get_intent_system_prompt(portal_language)

        try:
            # Use JSON format mode if available (much faster with thinking models)
            generate_fn = getattr(self.llm, "generate_json", self.llm.generate)
            raw_output = await generate_fn(
                prompt=prompt,
                system_prompt=system_prompt,
            )

            intent = self._parse_json_response(raw_output, user_message)

            return IntentParseResult(
                success=True,
                intent=intent,
                raw_llm_output=raw_output,
            )

        except Exception as e:
            logger.error(f"Intent parsing failed: {e}")
            # Fallback: create a basic intent from the raw message
            fallback = self._fallback_intent(user_message)
            return IntentParseResult(
                success=False,
                intent=fallback,
                raw_llm_output="",
                error=str(e),
            )

    def _parse_json_response(self, raw: str, original_message: str) -> UserIntent:
        """Parse the LLM's JSON response into a UserIntent."""
        # Strip qwen3 <think>...</think> tags if present
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Remove "json" language tag
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

        data = json.loads(cleaned)

        # Ensure custom_question is preserved
        if not data.get("custom_question"):
            data["custom_question"] = original_message

        return UserIntent(**data)

    def _fallback_intent(self, user_message: str) -> UserIntent:
        """Create a basic fallback intent when LLM parsing fails."""
        # Simple keyword-based analysis type detection
        msg_lower = user_message.lower()

        analysis_type = AnalysisType.SUMMARY
        if any(kw in msg_lower for kw in ["trend", "change", "over time", "growth", "decline"]):
            analysis_type = AnalysisType.TREND
        elif any(kw in msg_lower for kw in ["compare", "versus", "vs", "difference", "ranking"]):
            analysis_type = AnalysisType.COMPARISON
        elif any(kw in msg_lower for kw in ["where", "location", "map", "area", "district"]):
            analysis_type = AnalysisType.SPATIAL
        elif any(kw in msg_lower for kw in ["distribution", "breakdown", "spread", "proportion"]):
            analysis_type = AnalysisType.DISTRIBUTION
        elif any(kw in msg_lower for kw in ["relationship", "correlation", "related", "affect"]):
            analysis_type = AnalysisType.CORRELATION

        return UserIntent(
            dataset_query=user_message,
            analysis_type=analysis_type,
            custom_question=user_message,
        )
