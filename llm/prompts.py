"""
Prompt assembly for narrative generation.
Builds structured prompts from evidence bundles and user intents,
with guardrails to ensure safe and accurate output.
"""

import logging
from typing import Optional

from pydantic import BaseModel, Field

from llm.intent import UserIntent, AnalysisType, AudienceLevel
from data.analytics.evidence_bundle import EvidenceBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AssembledPrompt(BaseModel):
    """A fully assembled prompt ready for LLM generation."""
    system_prompt: str
    user_prompt: str
    expected_output_schema: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

GUARDRAILS = """
IMPORTANT RULES — you must follow these strictly:

1. ACCURACY: Only reference data values, statistics, and findings that are
   explicitly provided in the evidence context below. Never invent numbers,
   trends, or facts. If data is insufficient, say so.

2. NO HALLUCINATION: Do not reference datasets, columns, time periods, or
   geographic areas that are not mentioned in the evidence context.

3. HEDGING: Use hedging language when describing trends or patterns
   ("the data suggests", "appears to show") rather than absolute claims.

4. SCOPE: Only discuss what the data shows. Do not speculate about causes
   unless the data directly supports it.

5. LIMITATIONS: Briefly note any data limitations you observe (missing values,
   short time ranges, small sample sizes).

6. NEUTRALITY: Present findings neutrally without political commentary or
   value judgments about city governance.

7. STRUCTURED OUTPUT: Respond ONLY with valid JSON matching the requested
   schema. No markdown, no extra text.
"""


# ---------------------------------------------------------------------------
# System prompts by audience
# ---------------------------------------------------------------------------

AUDIENCE_PROMPTS = {
    AudienceLevel.CITIZEN: """You are a city data storyteller. Your job is to explain
open data findings to everyday citizens in clear, accessible language.
Avoid jargon. Use concrete examples. Write as if explaining to someone
who reads the local newspaper but has no data background.
Keep paragraphs short. Use simple sentence structures.""",

    AudienceLevel.POLICY: """You are a policy analysis assistant. Your job is to
present data findings to city officials and policymakers. Be precise with
numbers. Highlight actionable insights. Structure the narrative around
implications for city services, resource allocation, and planning.
Use professional but accessible language.""",

    AudienceLevel.TECHNICAL: """You are a data analysis assistant. Your job is to
present data findings to analysts and technical staff. You can use
statistical terminology. Include specific metrics and methodology notes.
Reference column names and data types when relevant. Be concise and precise.""",
}


# ---------------------------------------------------------------------------
# Analysis-specific prompt fragments
# ---------------------------------------------------------------------------

ANALYSIS_PROMPTS = {
    AnalysisType.TREND: """Focus on:
- The overall direction of change (increasing, decreasing, stable)
- The magnitude of change (percentage, absolute values)
- Notable inflection points or anomalies
- Seasonal patterns if visible
- Comparison of the most recent value to the historical range""",

    AnalysisType.COMPARISON: """Focus on:
- Rankings: which categories lead and trail
- The gap between highest and lowest values
- Notable outliers or surprises
- Clusters of similar categories
- What the average looks like and who's above/below it""",

    AnalysisType.DISTRIBUTION: """Focus on:
- How values are spread across the range
- Whether the distribution is even or skewed
- Outliers and extreme values
- Concentration: are most values in a narrow band?
- What the typical (median/mode) value looks like""",

    AnalysisType.SUMMARY: """Provide a general overview:
- What the dataset contains and what it measures
- Key statistics (totals, averages, ranges)
- The most notable finding or pattern
- Any data quality observations
- A suggestion for deeper analysis""",

    AnalysisType.SPATIAL: """Focus on:
- Geographic patterns and clusters
- Areas with notably high or low values
- Coverage gaps or data density variations
- Spatial relationships between features
- What the geographic spread suggests""",

    AnalysisType.CORRELATION: """Focus on:
- Whether variables appear to move together
- The strength and direction of relationships
- Notable outliers from the general pattern
- Caveats: correlation does not imply causation
- What further analysis might reveal""",
}


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

NARRATIVE_OUTPUT_SCHEMA = {
    "title": "A concise title for the narrative (max 10 words)",
    "summary": "A 1-2 sentence executive summary of the key finding",
    "sections": [
        {
            "heading": "Section heading",
            "body": "Section narrative text (2-4 sentences)",
            "key_metric": {
                "label": "Metric name",
                "value": "Metric value as string",
                "context": "Brief context for the metric",
            },
        }
    ],
    "data_limitations": "Brief note on any data limitations observed",
    "suggested_followup": "One suggested follow-up question for the user",
}


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

class PromptAssembler:
    """
    Assembles prompts for narrative generation from evidence bundles
    and user intents.
    """

    def assemble(
        self,
        bundle: EvidenceBundle,
        intent: Optional[UserIntent] = None,
    ) -> AssembledPrompt:
        """
        Assemble a complete prompt from an evidence bundle and user intent.

        Args:
            bundle: The evidence bundle with metrics, specs, and context.
            intent: The parsed user intent. If None, defaults to summary.

        Returns:
            AssembledPrompt ready for the LLM.
        """
        if intent is None:
            intent = UserIntent(
                dataset_query=bundle.dataset_id,
                analysis_type=AnalysisType.SUMMARY,
            )

        system_prompt = self._build_system_prompt(intent)
        user_prompt = self._build_user_prompt(bundle, intent)

        logger.info(
            f"Assembled prompt: audience={intent.audience.value}, "
            f"analysis={intent.analysis_type.value}, "
            f"system={len(system_prompt)} chars, user={len(user_prompt)} chars"
        )

        return AssembledPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expected_output_schema=NARRATIVE_OUTPUT_SCHEMA,
        )

    def _build_system_prompt(self, intent: UserIntent) -> str:
        """Build the system prompt from audience and guardrails."""
        audience_prompt = AUDIENCE_PROMPTS.get(
            intent.audience,
            AUDIENCE_PROMPTS[AudienceLevel.CITIZEN],
        )

        return f"""{audience_prompt}

{GUARDRAILS}

You must respond with ONLY a valid JSON object matching this schema:
{_format_schema(NARRATIVE_OUTPUT_SCHEMA)}
"""

    def _build_user_prompt(self, bundle: EvidenceBundle, intent: UserIntent) -> str:
        """Build the user prompt from evidence and intent."""
        # Evidence context
        evidence_context = bundle.to_llm_context()

        # Analysis guidance
        analysis_guidance = ANALYSIS_PROMPTS.get(
            intent.analysis_type,
            ANALYSIS_PROMPTS[AnalysisType.SUMMARY],
        )

        # Build the prompt
        parts = [
            "=== EVIDENCE CONTEXT ===",
            evidence_context,
            "",
            "=== ANALYSIS GUIDANCE ===",
            analysis_guidance,
        ]

        # Add user's custom question if present
        if intent.custom_question:
            parts.extend([
                "",
                "=== USER QUESTION ===",
                intent.custom_question,
            ])

        # Add language instruction
        if intent.language != "en":
            parts.extend([
                "",
                f"=== LANGUAGE ===",
                f"Write the narrative in language code: {intent.language}",
            ])

        # Add geographic focus if present
        if intent.geographic_focus:
            parts.extend([
                "",
                f"=== GEOGRAPHIC FOCUS ===",
                f"Focus the analysis on: {intent.geographic_focus}",
            ])

        # Add time range if present
        if intent.time_range:
            parts.extend([
                "",
                f"=== TIME FOCUS ===",
                f"Focus on the time period: {intent.time_range}",
            ])

        parts.extend([
            "",
            "=== INSTRUCTION ===",
            "Generate a data narrative based on the evidence context above.",
            "Respond with ONLY the JSON object, no other text.",
        ])

        return "\n".join(parts)


def _format_schema(schema: dict, indent: int = 2) -> str:
    """Format the output schema as a readable string."""
    import json
    return json.dumps(schema, indent=indent)
