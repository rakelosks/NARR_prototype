"""
Live test for LLM intent parsing.
Run from project root: python -m tests.test_llm.test_intent

Requires Ollama running with the intent model (default: qwen3:4b).
"""

import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings
from llm.providers.ollama import OllamaProvider
from llm.intent import IntentParser, AnalysisType


# ---------------------------------------------------------------------------
# Expected analysis types for test queries
# ---------------------------------------------------------------------------

TEST_CASES = [
    ("How has the city budget changed over the last 3 years?", AnalysisType.TREND),
    ("Compare complaint rates across districts", AnalysisType.COMPARISON),
    ("Where are the swimming pools located in Reykjavik?", AnalysisType.SPATIAL),
]


async def main():
    print("=" * 60)
    print(f"LIVE INTENT PARSING TEST  (model: {settings.ollama_intent_model})")
    print(f"  timeout={settings.ollama_timeout}s")
    print("=" * 60)

    provider = OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_intent_model,
        timeout=settings.ollama_timeout,
    )
    parser = IntentParser(provider)

    passed = 0
    failed = 0

    for query, expected_type in TEST_CASES:
        print(f"\n  Query: '{query}'")
        print(f"  Expected: {expected_type.value}")

        t0 = time.time()
        result = await parser.parse(query)
        elapsed = time.time() - t0

        if result.success:
            actual = result.intent.analysis_type
            ok = actual == expected_type
            status = "PASS ✓" if ok else f"FAIL ✗ (got {actual.value})"
            print(f"  LLM parsed → {actual.value}  [{status}]  ({elapsed:.1f}s)")
            print(f"    dataset_query: {result.intent.dataset_query}")
            if ok:
                passed += 1
            else:
                failed += 1
        else:
            print(f"  LLM error: {result.error}")
            fallback = result.intent.analysis_type
            ok = fallback == expected_type
            status = "PASS ✓" if ok else f"FAIL ✗ (got {fallback.value})"
            print(f"  Fallback → {fallback.value}  [{status}]  ({elapsed:.1f}s)")
            if ok:
                passed += 1
            else:
                failed += 1

    print("\n" + "=" * 60)
    total = len(TEST_CASES)
    if failed == 0:
        print(f"ALL {total} INTENT TESTS PASSED ✓")
    else:
        print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
