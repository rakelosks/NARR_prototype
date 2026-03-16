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
    AudienceLevel.CITIZEN: """You are an editorial data storyteller for a city open data portal.
Write as if composing a short article for the city's website — accessible to
anyone who lives in the city, regardless of their background.
Use clear, conversational language. Avoid jargon and statistical terms.
Explain WHY the data matters to everyday residents.
Each paragraph should introduce and explain its accompanying chart before
the reader sees it — describe the pattern they should notice.""",

    AudienceLevel.POLICY: """You are a policy-focused data storyteller for a city open data portal.
Write as if composing a briefing for city officials and policymakers.
Be precise with numbers. Highlight actionable insights and implications
for city services, resource allocation, and planning.
Each paragraph should introduce and explain its accompanying chart,
pointing out what is most relevant for policy decisions.""",

    AudienceLevel.TECHNICAL: """You are a technical data storyteller for a city open data portal.
Write for analysts and technical staff. You may use statistical terminology
and reference column names. Include specific metrics and methodology notes.
Each paragraph should introduce and explain its accompanying chart,
noting data distributions, outliers, and analytical implications.""",
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
    "headline": "A concise headline for the data story (max 10 words)",
    "lede": "An opening paragraph (2-4 sentences) that gives context — WHY this data matters to a citizen, not just what it contains.",
    "story_blocks": [
        {
            "type": "narrative",
            "heading": "Section heading describing the insight",
            "body": "2-4 sentences introducing and explaining the chart. Describe the pattern the reader should notice.",
            "viz_index": 0,
        },
        {
            "type": "timeline",
            "heading": "Timeline heading (optional block — only include if data suggests clear milestones)",
            "milestones": [
                {"label": "2009", "description": "What happened this year"},
                {"label": "2015", "description": "What happened this year"},
            ],
        },
        {
            "type": "callout",
            "body": "A sentence explaining the highlighted stat",
            "highlight_value": "35%",
            "highlight_label": "Short label for the stat",
        },
    ],
    "data_note": "Brief note on any data limitations (or empty string if none)",
    "followup_question": "One suggested follow-up question for the reader",
}


STORY_STRUCTURE_GUIDE = """
STORY STRUCTURE — follow this carefully:

1. HEADLINE: A short, engaging title (max 10 words). Think newspaper headline.

2. LEDE: An opening paragraph that frames WHY this data matters to a city
   resident. Don't start with numbers — start with context, then weave in
   the key finding. This is what hooks the reader.

3. STORY BLOCKS: 2-4 blocks that form the body of the story. Types:

   a) "narrative" blocks (required, at least 2):
      - heading: a descriptive section title
      - body: 2-4 sentences that INTRODUCE and EXPLAIN the chart. Describe
        what the chart shows and what pattern to notice BEFORE the reader
        sees it. This is the most important part.
      - viz_index: which chart to show (0 = primary, 1 = secondary, or omit
        if no chart accompanies this block). Each chart should only be
        referenced once.

   b) "timeline" block (optional, max 1):
      - Only include if the data suggests clear milestones, policy changes,
        or turning points. Do NOT invent milestones — only use if the data
        or context supports it.
      - milestones: at least 2 entries with label (year/date) and description.

   c) "callout" block (optional, max 1):
      - For the single most striking number or finding.
      - highlight_value: the number/percentage (e.g. "35%", "150 kg")
      - highlight_label: short label (e.g. "decrease since 2005")
      - body: one sentence of context.

4. DATA NOTE: Brief mention of limitations, or empty string if none obvious.

5. FOLLOWUP QUESTION: One question the reader might want to explore next.
"""


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
        """Build the system prompt from audience, guardrails, and story structure."""
        audience_prompt = AUDIENCE_PROMPTS.get(
            intent.audience,
            AUDIENCE_PROMPTS[AudienceLevel.CITIZEN],
        )

        return f"""{audience_prompt}

{GUARDRAILS}

{STORY_STRUCTURE_GUIDE}

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

        # Add chart inventory so the LLM knows what viz_index values are valid
        parts.append("")
        parts.append("=== AVAILABLE CHARTS ===")
        if bundle.visualizations:
            for i, viz in enumerate(bundle.visualizations):
                primary_tag = " (primary)" if viz.is_primary else " (secondary)"
                parts.append(
                    f"Chart {i}{primary_tag}: {viz.chart_type} — \"{viz.title}\""
                )
            parts.append(
                f"Use viz_index 0-{len(bundle.visualizations)-1} in your story blocks "
                f"to place these charts. Each chart should appear at most once."
            )
        else:
            parts.append("No charts available. Omit viz_index from all blocks.")

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
            "Generate an editorial data story based on the evidence context above.",
            "Write it as a cohesive article, not a mechanical report.",
            "Respond with ONLY the JSON object, no other text.",
        ])

        return "\n".join(parts)


def _format_schema(schema: dict, indent: int = 2) -> str:
    """Format the output schema as a readable string."""
    import json
    return json.dumps(schema, indent=indent)
