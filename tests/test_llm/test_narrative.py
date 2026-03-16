"""
Live test for LLM narrative generation.
Run from project root: python -m tests.test_llm.test_narrative

Requires Ollama running with both models:
  - Intent model (default: qwen3:4b)
  - Generation model (default: qwen3:8b)
"""

import asyncio
import sys
import os
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings
from llm.providers.ollama import OllamaProvider
from llm.narrative import NarrativeGenerator
from data.profiling.profiler import profile_dataset
from data.profiling.matcher import match_template
from data.analytics.evidence_bundle import BundleBuilder


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


async def main():
    print("=" * 60)
    print("LIVE NARRATIVE GENERATION TEST")
    print(f"  Intent model:     {settings.ollama_intent_model}")
    print(f"  Generation model: {settings.ollama_generation_model}")
    print(f"  Timeout: {settings.ollama_timeout}s")
    print("=" * 60)

    intent_provider = OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_intent_model,
        timeout=settings.ollama_timeout,
    )
    generation_provider = OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_generation_model,
        timeout=settings.ollama_timeout,
    )
    generator = NarrativeGenerator(
        generation_provider,
        intent_llm_provider=intent_provider,
        max_retries=2,
    )

    bundle, _ = make_sample_bundle()

    print("\n  Generating narrative for: 'How has the city budget changed over time?'")
    print("  (this may take a few minutes with thinking models...)\n")

    t0 = time.time()
    result = await generator.generate(
        bundle,
        user_message="How has the city budget changed over time?",
    )
    elapsed = time.time() - t0

    print(f"  Success:  {result.success}")
    print(f"  Attempts: {result.attempts}")
    print(f"  Time:     {elapsed:.1f}s")

    if result.success and result.narrative:
        print(f"  Headline:      {result.narrative.headline}")
        print(f"  Lede:          {result.narrative.lede[:120]}...")
        print(f"  Story blocks:  {len(result.narrative.story_blocks)}")
        for b in result.narrative.story_blocks:
            heading = b.heading or b.type
            body_preview = (b.body or "")[:80]
            viz_tag = f" [chart {b.viz_index}]" if b.viz_index is not None else ""
            print(f"    - [{b.type}] {heading}: {body_preview}...{viz_tag}")
        if result.narrative.data_note:
            print(f"  Data note:     {result.narrative.data_note[:100]}...")
        if result.narrative.followup_question:
            print(f"  Follow-up:     {result.narrative.followup_question[:100]}...")
        print("\n" + "=" * 60)
        print("NARRATIVE GENERATION PASSED ✓")
        print("=" * 60)
    else:
        print(f"\n  Error: {result.error}")
        if result.validation:
            print(f"  Validation errors: {result.validation.errors}")
            raw = result.validation.raw_output
            if raw:
                print(f"  Raw output (first 500 chars):\n{raw[:500]}")
        print("\n" + "=" * 60)
        print("NARRATIVE GENERATION FAILED ✗")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
