"""
Prompt assembly for narrative generation.
Builds structured prompts from evidence bundles and user intents,
with guardrails to ensure safe and accurate output.
"""

import logging
from typing import Optional

from pydantic import BaseModel, Field

from llm.intent import UserIntent, AnalysisType, AudienceLevel, _LANGUAGE_NAMES
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
   CRITICAL: If the user asked to compare or break down data BY a specific
   dimension (e.g. "by neighborhood", "by age group", "by district") but the
   dataset does NOT contain that dimension in its columns or sample data,
   you MUST state this clearly in the lede: "The available data does not
   include a breakdown by [dimension]." Then describe what the data DOES
   contain instead. NEVER fabricate categories, area names, or groupings
   that are not present in the actual data rows.

3. DATA INTERPRETATION: Use the dataset title, description, column values,
   and sample data rows to understand what the data ACTUALLY measures.
   Column names may be in a non-English language — read the categorical
   values and sample rows carefully to understand what each column represents.
   Do NOT guess what the data is about from column names alone. If a dataset
   is called "school services" it does NOT mean "number of schools" — look
   at the actual values and description to determine what is being counted.
   ALWAYS check the sample data before writing the headline or lede.

   DATASET RELEVANCE: Compare the user's question with what the dataset
   actually contains. If the dataset does NOT directly answer the user's
   question (e.g., they asked about "number of schools" but the dataset
   is about "school support services"), you MUST acknowledge this honestly
   in the lede. For example: "While we don't have direct data on the
   number of schools, we do have data on school support services, which
   shows..." NEVER pretend the data answers a question it doesn't.

4. HEDGING: When the data doesn't definitively prove a cause, use language
   like "the data shows" or "this suggests" rather than absolute claims
   about why something happened. You CAN be direct about what the numbers
   themselves show — "waste dropped by 35%" is a fact, not a claim.

5. SCOPE: Only discuss what the data shows. Do not speculate about causes
   unless the data directly supports it.

6. NEUTRALITY: Present findings neutrally without political commentary or
   value judgments about city governance.

7. STRUCTURED OUTPUT: Respond ONLY with valid JSON matching the requested
   schema. No markdown, no extra text outside the JSON.
"""


# ---------------------------------------------------------------------------
# System prompts by audience
# ---------------------------------------------------------------------------

AUDIENCE_PROMPTS = {
    AudienceLevel.CITIZEN: """You are an editorial data storyteller for a city open data portal.
Your reader is someone who lives in this city — they may not have any
technical background, but they are smart and curious. They want to
understand what's happening in their city.

WRITING STYLE — this is critical:
- Write like a friendly, clear newspaper article, NOT a data report.
- Use short sentences. Use everyday words. No jargon.
- NEVER use terms like "dataset", "metrics", "cross-sectional",
  "aggregated", "statistically significant", or "correlation".
  Instead say "the numbers", "the data shows", "over time".
- Lead every section with the HUMAN story, then back it up with numbers.
  BAD: "The data shows a 35% decrease in per-capita waste generation."
  GOOD: "Residents are throwing away a third less rubbish than they did
  15 years ago — about 150 kg per person in 2012, down from 266 kg."
- Make numbers relatable. Round where it helps clarity.
  Say "about a third less" alongside "35% decrease".
  Do NOT invent real-world comparisons for weight or size
  (e.g. "the weight of a car") — these are almost always wrong.
- Each paragraph should prepare the reader for the chart that follows.
  Tell them what to look for: "In the chart below, you can see how the
  line drops steadily after 2005."
- End sections by connecting back to why it matters for daily life.""",

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
    AnalysisType.TREND: """Focus your story on HOW THINGS HAVE CHANGED:
- What's the big picture? Going up, going down, or staying flat?
- By how much? Give both the percentage and a tangible comparison.
- Was there a turning point — a year where things shifted?
- How does the most recent number compare to where things started?
- What should the reader take away about the direction things are heading?""",

    AnalysisType.COMPARISON: """Focus your story on HOW THINGS COMPARE:
- What's at the top? What's at the bottom? By how much?
- Are there any surprises — categories that are unexpectedly high or low?
- Is the gap between the top and bottom large or small?
- Are most categories clustered together, or are there clear standouts?
- What's the key takeaway a resident would care about?""",

    AnalysisType.DISTRIBUTION: """Focus your story on HOW THINGS ARE SPREAD OUT:
- Are most values bunched together, or spread across a wide range?
- What does a typical value look like?
- Are there any extreme outliers — unusually high or low?
- What pattern should the reader notice in the chart?""",

    AnalysisType.SUMMARY: """Focus your story on THE BIG PICTURE of this dataset:
- Start by explaining in plain language what this data actually tracks.
  Don't assume the reader knows — a title like "school services" could mean
  many things, so ground them using the actual values and description.
- What is the single most interesting or surprising finding? Lead with that.
- Give concrete numbers: totals, averages, highs, lows, and ranges.
  Make them relatable ("about twice as much as...", "roughly one per...").
- Are there any notable outliers or unexpected values?
- If there's a time dimension, briefly note whether things are going up,
  down, or staying flat — but keep the focus on the overall picture.
- End with a clear takeaway: what should the reader remember from this data?""",

    AnalysisType.SPATIAL: """Focus your story on WHERE THINGS ARE HAPPENING:
- Are there clusters — areas where values are concentrated?
- Which areas stand out as notably high or low?
- Are there gaps — areas with no data?
- What geographic pattern should the reader notice on the map?""",

    AnalysisType.CORRELATION: """Focus your story on HOW TWO THINGS RELATE:
- Do they move together? When one goes up, does the other?
- How strong is the relationship? A clear pattern or a loose one?
- Are there outliers that don't fit the pattern?
- Important: note that a pattern doesn't prove one causes the other.""",
}


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

NARRATIVE_OUTPUT_SCHEMA = {
    "headline": "A finding, not a topic. e.g. 'Residents Throw Away a Third Less Than 15 Years Ago'",
    "lede": "2-3 sentences. Start with why this matters to someone living here, then state the key finding with a concrete number.",
    "story_blocks": [
        {
            "type": "callout",
            "highlight_value": "35%",
            "highlight_label": "less waste per person since 2005",
            "body": "One sentence putting this number in context for the reader.",
        },
        {
            "type": "narrative",
            "heading": "A finding-based heading, not a topic label",
            "body": "2-4 sentences. Tell the reader what pattern to look for in the chart below. Use plain language.",
            "viz_index": "INTEGER — index of an available chart from the AVAILABLE CHARTS section. Only use indices that exist.",
        },
        {
            "type": "narrative",
            "heading": "Another angle on the same story",
            "body": "2-4 sentences introducing the next chart and what it reveals.",
            "viz_index": "INTEGER or omit — only include if another chart is available.",
        },
        {
            "type": "narrative",
            "heading": "What this means for residents",
            "body": "2-3 sentences connecting the findings back to everyday life. What's the takeaway?",
        },
    ],
    "data_note": "Plain-language note on what this data covers and any limitations. Written for non-technical readers.",
    "followup_question": "A suggested next topic to explore, framed as a prompt the reader could type. e.g. 'How has the city budget changed over the years?'",
}


STORY_STRUCTURE_GUIDE = """
STORY STRUCTURE — follow this carefully:

1. HEADLINE: Lead with the FINDING, not the topic.
   BAD:  "Waste Collection Trends Over Time"
   GOOD: "Residents Throw Away a Third Less Than 15 Years Ago"
   BAD:  "Bus Ridership Data Analysis"
   GOOD: "Bus Ridership Has Doubled Since 2015"
   Keep it under 12 words. Make it interesting enough that someone
   would want to read the rest.

2. LEDE: 2-3 sentences that hook the reader. Start with WHY this
   matters to someone who lives in the city, then weave in the most
   striking number. Do NOT start with "This data shows..." or
   "The dataset contains...". Start with the human angle.
   EXAMPLE: "Every week, the bins go out. But how much are we actually
   throwing away — and is it getting better? The numbers tell a
   surprisingly positive story: residents produce about 150 kg of
   waste per person per year, down 35% from 2005."

3. STORY BLOCKS: 3-5 blocks that form the body. Mix these types:

   a) "callout" block — USE THIS for the single most striking number.
      Place it early (often right after the lede) so the reader gets
      the headline stat immediately. Include:
      - highlight_value: the number (e.g. "35%", "150 kg")
      - highlight_label: short label (e.g. "less waste per person")
      - body: one sentence of plain-language context.
      Include a callout in MOST stories — if the data has a notable
      finding (and it usually does), highlight it. Only omit if there
      truly is no standout number.

   b) "narrative" blocks (required, at least 2):
      - heading: a FINDING-based heading, not a topic label.
        BAD:  "Yearly Waste Per Capita Statistics"
        GOOD: "Waste Per Person Dropped Steadily After 2005"
      - body: 2-4 sentences. FIRST tell the reader what the chart
        shows and what pattern to look for ("In the chart below,
        you'll see..."). THEN explain what it means. Use concrete
        numbers and make percentages relatable ("about a third less").
      - viz_index: assign each available chart to exactly one
        narrative block. The chart appears right after the text.
        Use 0 for primary, 1 for secondary, etc.

   c) "timeline" block (optional, max 1):
      - Include if the data shows clear turning points or if the
        evidence context mentions policy changes, new programs,
        or milestones. Do NOT invent milestones.
      - milestones: at least 2 entries with label and description.

   d) A CLOSING narrative block (recommended, no viz_index):
      - 2-3 sentences wrapping up the story. Connect findings back
        to daily life. What does this mean for someone living here?
        Keep it grounded — no grand conclusions beyond what the
        data supports.

4. DATA NOTE: Write this for a non-technical reader. Explain:
   - What time period the data covers.
   - If anything important might be missing or incomplete.
   - Any context that helps the reader interpret the numbers correctly.
   Do NOT use jargon like "cross-sectional", "single-period",
   "aggregated", or "limited sample size". Instead say things like
   "This data covers 1998 to 2012, so it doesn't include more recent
   years" or "The numbers are averages across all neighborhoods, so
   your area might differ."
   IMPORTANT: Check the column types before writing this — do NOT
   claim data is single-period if temporal columns exist.

5. FOLLOWUP QUESTION: Suggest a DIFFERENT topic the reader could
   explore next on the same portal. Frame it as something they could
   type into this system to get another data story.
   BAD:  "What initiatives have contributed to the reduction?"
         (too specific — the system can't answer this)
   BAD:  "How does this compare to other Nordic cities?"
         (the system only has data from this city's portal)
   GOOD: "How has the city budget changed over the years?"
   GOOD: "What do the city's school enrollment numbers look like?"
   GOOD: "How is public transport ridership trending?"
   The suggestion should feel like a natural "you might also be
   interested in..." nudge toward a completely different city topic.
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
                desc = f" — {viz.description}" if viz.description else ""
                parts.append(
                    f"Chart {i}{primary_tag}: {viz.chart_type} — \"{viz.title}\"{desc}"
                )
                parts.append(
                    f"  → Your narrative block for this chart should describe "
                    f"what THIS specific chart shows, not repeat the same "
                    f"analysis from another chart."
                )
            num_viz = len(bundle.visualizations)
            if num_viz == 1:
                parts.append(
                    "⚠ IMPORTANT: There is ONLY 1 chart (viz_index=0). "
                    "Set viz_index=0 on exactly ONE narrative block. "
                    "ALL other narrative blocks MUST NOT include viz_index at all — "
                    "omit the viz_index field entirely, do NOT set it to 1 or any "
                    "other number. You can still have 3-5 story blocks; most of "
                    "them simply won't have a chart attached."
                )
            else:
                parts.append(
                    f"There are {num_viz} charts available (viz_index 0 to {num_viz - 1}). "
                    f"Assign each chart to exactly one narrative block. "
                    f"Do NOT reuse the same viz_index in multiple blocks. "
                    f"Any remaining narrative blocks should omit viz_index entirely."
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
            lang_name = _LANGUAGE_NAMES.get(intent.language, intent.language)
            parts.extend([
                "",
                f"=== LANGUAGE ===",
                f"Write ALL narrative text (headline, lede, story blocks, data note, "
                f"and followup question) in {lang_name}. The JSON keys must stay in English.",
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

        # Build final instruction block, reinforcing language at the very end
        # so it's the last thing the model sees before generating.
        instruction_lines = [
            "",
            "=== INSTRUCTION ===",
            "Generate an editorial data story based on the evidence above.",
            "Write it as a story someone would actually enjoy reading — not a report.",
            "Lead with the most interesting finding. Use plain, everyday language.",
            "Make percentages relatable ('about a third less') but do NOT invent",
            "real-world size or weight comparisons — they are usually wrong.",
            "Prepare the reader for each chart by describing what to look for.",
            "Respond with ONLY the JSON object, no other text.",
        ]
        if intent.language != "en":
            lang_name = _LANGUAGE_NAMES.get(intent.language, intent.language)
            instruction_lines.append(
                f"REMINDER: Write ALL narrative text in {lang_name}. "
                f"Only JSON keys stay in English."
            )
        parts.extend(instruction_lines)

        return "\n".join(parts)


def _format_schema(schema: dict, indent: int = 2) -> str:
    """Format the output schema as a readable string."""
    import json
    return json.dumps(schema, indent=indent)
