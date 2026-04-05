"""
Re-collect only the corpus files for query 10 (run_10a and run_10b).

Same endpoint and JSON shape as collect_corpus.py, but sends the 10th
predefined query twice and writes:
  corpus/run_10a_query.json
  corpus/run_10b_query.json

Use when those runs were rate-limited or fell back to preview narrative.
Optional delay between the two POSTs may help avoid rate limits.

Usage:
    python scripts/collect_corpus_run10_ab.py
    python scripts/collect_corpus_run10_ab.py --base-url http://localhost:8000
    python scripts/collect_corpus_run10_ab.py --delay-seconds 15
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from collect_corpus import QUERIES, TIMEOUT

QUERY_INDEX = 10  # 1-based, matches run_10a / run_10b
RUN_IDS = ("run_10a", "run_10b")


def collect_one(
    endpoint: str,
    run_id: str,
    query: str,
) -> dict:
    start = time.time()
    try:
        resp = requests.post(
            endpoint,
            json={"user_message": query},
            timeout=TIMEOUT,
        )
        elapsed = round(time.time() - start, 2)

        if resp.status_code == 200:
            print(f"  ✓ {run_id} — HTTP 200 ({elapsed}s)")
            return {
                "run_id": run_id,
                "query": query,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status_code": resp.status_code,
                "response_time_seconds": elapsed,
                "response": resp.json(),
            }
        print(f"  ✗ {run_id} — HTTP {resp.status_code} ({elapsed}s)")
        return {
            "run_id": run_id,
            "query": query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status_code": resp.status_code,
            "response_time_seconds": elapsed,
            "error": resp.text,
        }
    except requests.RequestException as exc:
        elapsed = round(time.time() - start, 2)
        print(f"  ✗ {run_id} — {exc} ({elapsed}s)")
        return {
            "run_id": run_id,
            "query": query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status_code": None,
            "response_time_seconds": elapsed,
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-collect corpus JSON for run_10a and run_10b only",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the NARR API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Wait this many seconds between the a and b requests (default: 0)",
    )
    args = parser.parse_args()

    if QUERY_INDEX < 1 or QUERY_INDEX > len(QUERIES):
        raise SystemExit(f"QUERY_INDEX {QUERY_INDEX} out of range (1–{len(QUERIES)})")

    query = QUERIES[QUERY_INDEX - 1]
    endpoint = f"{args.base_url.rstrip('/')}/narratives/ask"
    corpus_dir = Path("corpus")
    corpus_dir.mkdir(exist_ok=True)

    print(f"Query {QUERY_INDEX}: {query}")
    print(f"POST {endpoint}")
    total_start = time.time()

    for i, run_id in enumerate(RUN_IDS):
        if i > 0 and args.delay_seconds > 0:
            print(f"  … waiting {args.delay_seconds}s before {run_id}")
            time.sleep(args.delay_seconds)
        print(f"[{run_id}]")
        result = collect_one(endpoint, run_id, query)
        out_path = corpus_dir / f"{run_id}_query.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"  → {out_path}")

    total_elapsed = round(time.time() - total_start, 2)
    print(f"\nDone in {total_elapsed}s.")


if __name__ == "__main__":
    main()
