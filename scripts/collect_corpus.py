"""
Collect a corpus of NARR narrative responses for later analysis.

Sends 10 predefined queries to the /narratives/ask endpoint, twice each (a/b),
and saves each full response (or error) as JSON under corpus/ — 20 files:
run_01a_query.json, run_01b_query.json, … run_10b_query.json.

Usage:
    python scripts/collect_corpus.py
    python scripts/collect_corpus.py --base-url http://localhost:8000
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

QUERIES = [
    "What are the trends in waste collection?",
    "Compare the number of children on the kindergarten waiting list by district and age group",
    "What are the air quality trends in the city?",
    "What is the population distribution across neighborhoods?",
    "How has the city budget changed?",
    "How many traffic accidents occurred these past years?",
    "How many cruise ships have come to the city?",
    "How has unemployment developed these past years?",
    "Which district has the most foreign citizens?",
    "Compare population by age and gender",
]

TIMEOUT = 300  # seconds per request
REPLICATES = ("a", "b")  # two runs per query → 20 JSON files


def run(base_url: str) -> None:
    endpoint = f"{base_url}/narratives/ask"
    corpus_dir = Path("corpus")
    corpus_dir.mkdir(exist_ok=True)

    n_queries = len(QUERIES)
    total_runs = n_queries * len(REPLICATES)
    succeeded = 0
    failed = 0
    total_start = time.time()
    run_counter = 0

    for idx, query in enumerate(QUERIES, start=1):
        for rep in REPLICATES:
            run_counter += 1
            run_id = f"run_{idx:02d}{rep}"
            print(f"[{run_counter}/{total_runs}] {run_id}: {query}")

            start = time.time()
            try:
                resp = requests.post(
                    endpoint,
                    json={"user_message": query},
                    timeout=TIMEOUT,
                )
                elapsed = round(time.time() - start, 2)

                if resp.status_code == 200:
                    result = {
                        "run_id": run_id,
                        "query": query,
                        "replicate": rep,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "status_code": resp.status_code,
                        "response_time_seconds": elapsed,
                        "response": resp.json(),
                    }
                    succeeded += 1
                    print(f"  ✓ Success ({elapsed}s)")
                else:
                    result = {
                        "run_id": run_id,
                        "query": query,
                        "replicate": rep,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "status_code": resp.status_code,
                        "response_time_seconds": elapsed,
                        "error": resp.text,
                    }
                    failed += 1
                    print(f"  ✗ Failed — HTTP {resp.status_code} ({elapsed}s)")

            except requests.RequestException as exc:
                elapsed = round(time.time() - start, 2)
                result = {
                    "run_id": run_id,
                    "query": query,
                    "replicate": rep,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status_code": None,
                    "response_time_seconds": elapsed,
                    "error": str(exc),
                }
                failed += 1
                print(f"  ✗ Error — {exc} ({elapsed}s)")

            out_path = corpus_dir / f"{run_id}_query.json"
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    total_elapsed = round(time.time() - total_start, 2)
    print(
        f"\nDone. {total_runs} files written ({n_queries} queries × {len(REPLICATES)}). "
        f"{succeeded} succeeded, {failed} failed, {total_elapsed}s total."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect NARR narrative corpus")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the NARR API (default: http://localhost:8000)",
    )
    run(parser.parse_args().base_url)
