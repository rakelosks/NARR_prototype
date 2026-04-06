"""
Collect a corpus of NARR narrative responses for later analysis.

Sends 10 predefined queries to the /narratives/ask endpoint, twice each (a/b),
and saves each full response (or error) as JSON under a portal-specific folder.

Usage:
    python scripts/collect_corpus.py
    python scripts/collect_corpus.py --base-url http://localhost:8000
    python scripts/collect_corpus.py --portal-name madrid --portal-url https://datos.madrid.es/egob/catalogo/api/3
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


def run(
    base_url: str,
    portal_name: str,
    portal_url: str,
    output_dir: str,
    replicates: int,
) -> None:
    endpoint = f"{base_url}/narratives/ask"
    normalized_portal = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in portal_name.lower()).strip("_")
    normalized_portal = normalized_portal or "portal"
    corpus_dir = Path(output_dir) / normalized_portal
    corpus_dir.mkdir(parents=True, exist_ok=True)
    replicate_labels = tuple(chr(ord("a") + i) for i in range(replicates))

    n_queries = len(QUERIES)
    total_runs = n_queries * len(replicate_labels)
    succeeded = 0
    failed = 0
    total_start = time.time()
    run_counter = 0

    # Capture active API portal configuration for traceability.
    health = None
    try:
        health_resp = requests.get(f"{base_url}/health", timeout=30)
        if health_resp.status_code == 200:
            health = health_resp.json()
            print(
                f"Health check: portal_url={health.get('portal_url')} "
                f"portal_language={health.get('portal_language')}"
            )
    except requests.RequestException as exc:
        print(f"Health check unavailable: {exc}")

    for idx, query in enumerate(QUERIES, start=1):
        for rep in replicate_labels:
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
                        "portal_name": portal_name,
                        "portal_url": portal_url,
                        "api_base_url": base_url,
                        "health": health,
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
                        "portal_name": portal_name,
                        "portal_url": portal_url,
                        "api_base_url": base_url,
                        "health": health,
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
                    "portal_name": portal_name,
                    "portal_url": portal_url,
                    "api_base_url": base_url,
                    "health": health,
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
    print(f"\nOutput folder: {corpus_dir}")
    print(f"Done. {total_runs} files written ({n_queries} queries × {len(replicate_labels)}). {succeeded} succeeded, {failed} failed, {total_elapsed}s total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect NARR narrative corpus")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the NARR API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--portal-name",
        default="reykjavik",
        help="Portal label used for output folder naming (default: reykjavik)",
    )
    parser.add_argument(
        "--portal-url",
        default="https://gagnagatt.reykjavik.is/en/api/3",
        help="CKAN portal URL for traceability metadata in each run file",
    )
    parser.add_argument(
        "--output-dir",
        default="corpus",
        help="Base directory where portal-specific corpus folders are written",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=2,
        help="Replicates per query, labeled a/b/c... (default: 2)",
    )
    args = parser.parse_args()
    if args.replicates < 1 or args.replicates > 26:
        raise SystemExit("--replicates must be between 1 and 26")
    run(
        base_url=args.base_url,
        portal_name=args.portal_name,
        portal_url=args.portal_url,
        output_dir=args.output_dir,
        replicates=args.replicates,
    )
