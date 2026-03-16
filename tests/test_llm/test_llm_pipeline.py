"""
Offline tests for the LLM narrative generation pipeline.
Run from project root: python -m tests.test_llm.test_llm_pipeline

Tests intent parsing, prompt assembly, output validation,
and LLM context formatting — all without needing a running LLM.

For live tests see:
  python -m tests.test_llm.test_intent
  python -m tests.test_llm.test_narrative
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
    assert "data storyteller" in prompt1.system_prompt
    assert "EVIDENCE CONTEXT" in prompt1.user_prompt
    assert "AVAILABLE CHARTS" in prompt1.user_prompt
    assert "GUARDRAILS" in prompt1.system_prompt.upper() or "IMPORTANT RULES" in prompt1.system_prompt
    assert "STORY STRUCTURE" in prompt1.system_prompt
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

    # Valid output — editorial data story format
    valid_output = json.dumps({
        "headline": "Reykjavik Budget Shows Steady Growth",
        "lede": "Over the past three years, the city budget has increased consistently, with expenditure tracking closely behind allocation. This steady pattern reflects planned expansion of city services.",
        "story_blocks": [
            {
                "type": "narrative",
                "heading": "Overall trend",
                "body": "Budget allocation has grown from 150,000 ISK to 185,000 ISK over 36 months, representing a steady upward trajectory. This consistent growth pattern suggests planned expansion of city services.",
                "viz_index": 0,
            },
            {
                "type": "narrative",
                "heading": "Expenditure tracking",
                "body": "Expenditure has followed a similar pattern, growing from 142,000 ISK to 173,500 ISK. The gap between budget and actual spending has remained relatively stable throughout the period.",
            },
            {
                "type": "callout",
                "body": "Budget grew by nearly a quarter over the full period.",
                "highlight_value": "+23.3%",
                "highlight_label": "Budget growth over 36 months",
            },
        ],
        "data_note": "The dataset covers 36 months which provides a reasonable trend view but may not capture longer cyclical patterns.",
        "followup_question": "How does spending vary across departments over time?",
    })

    result = validator.validate(valid_output, bundle)
    assert result.is_valid, f"Valid output should pass: {result.errors}"
    assert result.narrative is not None
    assert result.narrative.headline == "Reykjavik Budget Shows Steady Growth"
    assert len(result.narrative.story_blocks) == 3
    print(f"  Valid output: passed ✓")
    print(f"    Headline: {result.narrative.headline}")
    print(f"    Story blocks: {len(result.narrative.story_blocks)}")

    # Invalid: not JSON
    result2 = validator.validate("This is just text, not JSON.", bundle)
    assert not result2.is_valid
    print(f"  Non-JSON output: rejected ✓ ({result2.errors})")

    # Invalid: missing required fields
    result3 = validator.validate(json.dumps({"headline": "Test"}), bundle)
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
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("LLM PIPELINE OFFLINE TESTS")
    print("=" * 60)

    test_intent_models()
    test_prompt_assembly()
    test_output_validation()
    test_llm_context_format()

    print("\n" + "=" * 60)
    print("ALL OFFLINE TESTS PASSED")
    print("=" * 60)

    print("\nFor live tests run separately:")
    print("  python -m tests.test_llm.test_intent      # intent parsing")
    print("  python -m tests.test_llm.test_narrative    # narrative generation")


if __name__ == "__main__":
    main()
