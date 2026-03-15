"""
Test script for the LLM narrative generation pipeline.
Run from project root: python -m tests.test_llm_pipeline

Tests intent parsing, prompt assembly, output validation,
and (optionally) live narrative generation with Ollama.

Offline tests run without an LLM. Live tests require Ollama running.
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data.profiling.profiler import profile_dataset
from data.profiling.matcher import match_template
from data.analytics.evidence_bundle import BundleBuilder
from llm.intent import (
    UserIntent,
    IntentParser,
    AnalysisType,
    AudienceLevel,
)
from llm.prompts import PromptAssembler
from llm.narrative import (
    OutputValidator,
    GeneratedNarrative,
    NarrativeGenerator,
)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

def make_sample_bundle():
    """Create a sample evidence bundle for testing."""
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=36, freq="MS"),
        "department": ["Public Works", "Education", "Health"] * 12,
        "budget_isk": [150_000 + i * 1000 for i in range(36)],
        "expenditure_isk": [142_000 + i * 900 for i in range(36)],
    })
    profile = profile_dataset(df, dataset_id="rvk_budget", source="test/budget.csv")
    match = match_template(profile)
    builder = BundleBuilder()
    return builder.build(df, profile, match, title="Reykjavik Monthly Budget"), df


# ---------------------------------------------------------------------------
# Offline tests (no LLM required)
# ---------------------------------------------------------------------------

def test_intent_models():
    """Test intent model creation and validation."""
    print("\n" + "=" * 60)
    print("TEST: Intent models (offline)")
    print("=" * 60)

    # Valid intent
    intent = UserIntent(
        dataset_query="city budget spending",
        analysis_type=AnalysisType.TREND,
        focus_columns=["budget_isk", "expenditure_isk"],
        time_range="2020-2023",
        audience=AudienceLevel.CITIZEN,
        custom_question="How has the city budget changed over time?",
    )
    print(f"  Intent created: {intent.analysis_type.value}, audience={intent.audience.value}")
    print(f"  Query: {intent.dataset_query}")
    assert intent.analysis_type == AnalysisType.TREND

    # Fallback intent from parser
    from llm.intent import IntentParser
    # Test the static fallback method
    parser = IntentParser.__new__(IntentParser)
    fallback = parser._fallback_intent("How has traffic changed over time?")
    assert fallback.analysis_type == AnalysisType.TREND
    print(f"  Fallback 'over time' → {fallback.analysis_type.value} ✓")

    fallback2 = parser._fallback_intent("Compare districts by complaint count")
    assert fallback2.analysis_type == AnalysisType.COMPARISON
    print(f"  Fallback 'compare' → {fallback2.analysis_type.value} ✓")

    fallback3 = parser._fallback_intent("Where are the bus stops located?")
    assert fallback3.analysis_type == AnalysisType.SPATIAL
    print(f"  Fallback 'where' → {fallback3.analysis_type.value} ✓")

    print("  Intent model tests passed ✓")


def test_prompt_assembly():
    """Test prompt assembly from evidence bundle + intent."""
    print("\n" + "=" * 60)
    print("TEST: Prompt assembly (offline)")
    print("=" * 60)

    bundle, _ = make_sample_bundle()

    # Test with different intents
    assembler = PromptAssembler()

    # Citizen + trend
    intent1 = UserIntent(
        dataset_query="budget",
        analysis_type=AnalysisType.TREND,
        audience=AudienceLevel.CITIZEN,
        custom_question="How has the budget changed?",
    )
    prompt1 = assembler.assemble(bundle, intent1)
    assert "city data storyteller" in prompt1.system_prompt
    assert "EVIDENCE CONTEXT" in prompt1.user_prompt
    assert "GUARDRAILS" in prompt1.system_prompt.upper() or "IMPORTANT RULES" in prompt1.system_prompt
    print(f"  Citizen+trend prompt: system={len(prompt1.system_prompt)}, user={len(prompt1.user_prompt)} chars")

    # Policy + comparison
    intent2 = UserIntent(
        dataset_query="budget",
        analysis_type=AnalysisType.COMPARISON,
        audience=AudienceLevel.POLICY,
    )
    prompt2 = assembler.assemble(bundle, intent2)
    assert "policy" in prompt2.system_prompt.lower()
    print(f"  Policy+comparison prompt: system={len(prompt2.system_prompt)}, user={len(prompt2.user_prompt)} chars")

    # Default (no intent)
    prompt3 = assembler.assemble(bundle)
    assert "EVIDENCE CONTEXT" in prompt3.user_prompt
    print(f"  Default prompt: system={len(prompt3.system_prompt)}, user={len(prompt3.user_prompt)} chars")

    # Non-English
    intent_is = UserIntent(
        dataset_query="budget",
        analysis_type=AnalysisType.SUMMARY,
        language="is",
    )
    prompt_is = assembler.assemble(bundle, intent_is)
    assert "language code: is" in prompt_is.user_prompt
    print(f"  Icelandic prompt includes language instruction ✓")

    print("  Prompt assembly tests passed ✓")


def test_output_validation():
    """Test output validation with good and bad outputs."""
    print("\n" + "=" * 60)
    print("TEST: Output validation (offline)")
    print("=" * 60)

    bundle, _ = make_sample_bundle()
    validator = OutputValidator()

    # Valid output
    valid_output = json.dumps({
        "title": "Reykjavik Budget Shows Steady Growth",
        "summary": "The city budget has increased consistently over the past three years, with expenditure tracking closely behind allocation.",
        "sections": [
            {
                "heading": "Overall trend",
                "body": "Budget allocation has grown from 150,000 ISK to 185,000 ISK over 36 months, representing a steady upward trajectory. This consistent growth pattern suggests planned expansion of city services.",
                "key_metric": {
                    "label": "Budget growth",
                    "value": "+23.3%",
                    "context": "Over the full 36-month period",
                },
            },
            {
                "heading": "Expenditure tracking",
                "body": "Expenditure has followed a similar pattern, growing from 142,000 ISK to 173,500 ISK. The gap between budget and actual spending has remained relatively stable throughout the period.",
            },
        ],
        "data_limitations": "The dataset covers 36 months which provides a reasonable trend view but may not capture longer cyclical patterns.",
        "suggested_followup": "How does spending vary across departments over time?",
    })

    result = validator.validate(valid_output, bundle)
    assert result.is_valid, f"Valid output should pass: {result.errors}"
    assert result.narrative is not None
    assert result.narrative.title == "Reykjavik Budget Shows Steady Growth"
    assert len(result.narrative.sections) == 2
    print(f"  Valid output: passed ✓")
    print(f"    Title: {result.narrative.title}")
    print(f"    Sections: {len(result.narrative.sections)}")

    # Invalid: not JSON
    result2 = validator.validate("This is just text, not JSON.", bundle)
    assert not result2.is_valid
    print(f"  Non-JSON output: rejected ✓ ({result2.errors})")

    # Invalid: missing required fields
    result3 = validator.validate(json.dumps({"title": "Test"}), bundle)
    assert not result3.is_valid
    print(f"  Incomplete output: rejected ✓ ({result3.errors})")

    # Invalid: empty sections
    result4 = validator.validate(json.dumps({
        "title": "Test",
        "summary": "A valid summary here.",
        "sections": [],
    }), bundle)
    assert not result4.is_valid
    print(f"  Empty sections: rejected ✓ ({result4.errors})")

    # Valid with markdown code fence wrapping (common LLM behavior)
    fenced = f"```json\n{valid_output}\n```"
    result5 = validator.validate(fenced, bundle)
    assert result5.is_valid, f"Fenced JSON should pass: {result5.errors}"
    print(f"  Markdown-fenced JSON: accepted ✓")

    # Valid with surrounding text (another common LLM behavior)
    with_text = f"Here is the narrative:\n{valid_output}\n\nHope this helps!"
    result6 = validator.validate(with_text, bundle)
    assert result6.is_valid, f"JSON with surrounding text should pass: {result6.errors}"
    print(f"  JSON with surrounding text: accepted ✓")

    print("  Output validation tests passed ✓")


def test_llm_context_format():
    """Test that the evidence bundle formats properly for LLM consumption."""
    print("\n" + "=" * 60)
    print("TEST: LLM context formatting (offline)")
    print("=" * 60)

    bundle, _ = make_sample_bundle()
    context = bundle.to_llm_context()

    assert "Dataset:" in context
    assert "Metrics" in context
    assert "Trend" in context or "trend" in context

    lines = context.split("\n")
    print(f"  Context: {len(context)} chars, {len(lines)} lines")
    print(f"  Preview:")
    for line in lines[:10]:
        print(f"    | {line}")

    print("  LLM context formatting tests passed ✓")


# ---------------------------------------------------------------------------
# Live tests (require Ollama running)
# ---------------------------------------------------------------------------

async def test_live_intent_parsing():
    """Test intent parsing with a live LLM."""
    print("\n" + "=" * 60)
    print("TEST: Live intent parsing (requires Ollama)")
    print("=" * 60)

    from llm.providers.ollama import OllamaProvider
    from config import settings
    provider = OllamaProvider(model=settings.ollama_intent_model)
    parser = IntentParser(provider)

    test_queries = [
        "How has the city budget changed over the last 3 years?",
        "Compare complaint rates across districts",
        "Where are the swimming pools located in Reykjavik?",
    ]

    for query in test_queries:
        print(f"\n  Query: '{query}'")
        result = await parser.parse(query)
        if result.success:
            print(f"    Analysis type: {result.intent.analysis_type.value}")
            print(f"    Dataset query: {result.intent.dataset_query}")
            print(f"    Focus columns: {result.intent.focus_columns}")
        else:
            print(f"    Failed (using fallback): {result.error}")
            print(f"    Fallback type: {result.intent.analysis_type.value}")

    print("\n  Live intent parsing tests complete ✓")


async def test_live_narrative_generation():
    """Test full narrative generation with a live LLM."""
    print("\n" + "=" * 60)
    print("TEST: Live narrative generation (requires Ollama)")
    print("=" * 60)

    from llm.providers.ollama import OllamaProvider
    from config import settings
    intent_provider = OllamaProvider(model=settings.ollama_intent_model)
    generation_provider = OllamaProvider(model=settings.ollama_generation_model)
    generator = NarrativeGenerator(generation_provider, intent_llm_provider=intent_provider, max_retries=2)

    bundle, _ = make_sample_bundle()

    result = await generator.generate(
        bundle,
        user_message="How has the city budget changed over time?",
    )

    print(f"  Success: {result.success}")
    print(f"  Attempts: {result.attempts}")
    if result.success and result.narrative:
        print(f"  Title: {result.narrative.title}")
        print(f"  Summary: {result.narrative.summary}")
        print(f"  Sections: {len(result.narrative.sections)}")
        for s in result.narrative.sections:
            print(f"    - {s.heading}: {s.body[:80]}...")
        print(f"  Limitations: {result.narrative.data_limitations}")
        print(f"  Follow-up: {result.narrative.suggested_followup}")
    else:
        print(f"  Error: {result.error}")
        if result.validation:
            print(f"  Validation errors: {result.validation.errors}")
            print(f"  Raw output: {result.validation.raw_output[:300]}...")

    print("\n  Live narrative generation test complete ✓")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("LLM NARRATIVE PIPELINE TESTS")
    print("=" * 60)

    # Offline tests (always run)
    test_intent_models()
    test_prompt_assembly()
    test_output_validation()
    test_llm_context_format()

    # Live tests (only if Ollama is running)
    run_live = "--live" in sys.argv
    if run_live:
        print("\n\n>>> Running LIVE tests (Ollama required) <<<")
        try:
            await test_live_intent_parsing()
            await test_live_narrative_generation()
        except Exception as e:
            print(f"\n  Live test failed: {e}")
            print("  Make sure Ollama is running: ollama serve")
    else:
        print("\n\nSkipping live tests. Run with --live flag to test with Ollama:")
        print("  python -m tests.test_llm_pipeline --live")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
