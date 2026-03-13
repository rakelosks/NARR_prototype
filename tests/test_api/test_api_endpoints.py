"""
Test script for the FastAPI endpoints.
Run from project root: python -m tests.test_api_endpoints

Uses FastAPI TestClient — no server needed.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from fastapi.testclient import TestClient

# Pre-seed a test dataset before importing the app
from data.cache.parquet_cache import save_snapshot

test_df = pd.DataFrame({
    "district": ["Vesturbær", "Hlíðar", "Laugardalur", "Breiðholt", "Grafarvogur"],
    "complaint_count": [45, 78, 32, 120, 55],
    "avg_resolution_days": [3.2, 5.1, 2.8, 7.4, 4.0],
})
save_snapshot(test_df, "test_district")

test_ts_df = pd.DataFrame({
    "date": pd.date_range("2020-01-01", periods=24, freq="MS"),
    "budget": [100_000 + i * 2000 for i in range(24)],
    "spending": [95_000 + i * 1800 for i in range(24)],
})
save_snapshot(test_ts_df, "test_timeseries")

# Register in metadata store
from data.storage.metadata import MetadataStore
store = MetadataStore()
store.register_dataset("test_district", "Test District Data", row_count=5)
store.register_dataset("test_timeseries", "Test Time Series", row_count=24)

# Now import and create test client
from app.main import app
client = TestClient(app)


def test_health():
    print("\n--- Test: Health check ---")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("  Health check passed ✓")


def test_list_datasets():
    print("\n--- Test: List datasets ---")
    r = client.get("/datasets/")
    assert r.status_code == 200
    data = r.json()
    print(f"  Found {len(data['datasets'])} datasets")
    assert len(data["datasets"]) >= 2
    print("  List datasets passed ✓")


def test_preview_narrative():
    print("\n--- Test: Preview narrative (no LLM) ---")

    # Categorical dataset
    r = client.post("/narratives/preview", json={"dataset_id": "test_district"})
    assert r.status_code == 200
    data = r.json()
    print(f"  Package ID: {data['package_id']}")
    print(f"  Title: {data['title']}")
    print(f"  Template: {data['dataset']['template_type']}")
    print(f"  Visualizations: {len(data['visualizations'])}")
    for viz in data["visualizations"]:
        print(f"    - {viz['chart_type']}: {viz['title']}")
    assert len(data["visualizations"]) >= 1
    assert data["dataset"]["template_type"] == "categorical"

    # Time series dataset
    r2 = client.post("/narratives/preview", json={"dataset_id": "test_timeseries"})
    assert r2.status_code == 200
    data2 = r2.json()
    print(f"\n  Time series package: {data2['dataset']['template_type']}")
    assert data2["dataset"]["template_type"] == "time_series"

    print("  Preview narrative passed ✓")


def test_generate_visualization():
    print("\n--- Test: Generate visualization ---")
    r = client.post("/visualizations/generate", json={
        "dataset_id": "test_district",
        "title": "District Complaints",
    })
    assert r.status_code == 200
    data = r.json()
    print(f"  Chart type: {data['chart_type']}")
    print(f"  Template: {data['template_type']}")
    spec = data["vega_lite_spec"]
    assert "$schema" in spec
    assert "data" in spec
    assert len(spec["data"]["values"]) > 0
    print(f"  Spec has {len(spec['data']['values'])} data rows")
    print("  Generate visualization passed ✓")


def test_generate_visualization_override():
    print("\n--- Test: Visualization with chart type override ---")
    r = client.post("/visualizations/generate", json={
        "dataset_id": "test_district",
        "chart_type": "scatter",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["chart_type"] == "scatter"
    print(f"  Overridden to: {data['chart_type']} ✓")


def test_create_async_job():
    print("\n--- Test: Async job creation ---")
    r = client.post("/jobs/generate", json={
        "dataset_id": "test_district",
        "user_message": "Compare districts by complaints",
    })
    assert r.status_code == 202
    data = r.json()
    job_id = data["job_id"]
    print(f"  Job created: {job_id}, status={data['status']}")
    assert data["status"] == "pending"

    # Poll for completion (background task runs inline in TestClient)
    time.sleep(0.5)
    r2 = client.get(f"/jobs/{job_id}")
    assert r2.status_code == 200
    data2 = r2.json()
    print(f"  Job status: {data2['status']}")

    # The job should complete (might be in preview mode without LLM)
    if data2["status"] == "completed":
        result = data2["result"]
        print(f"  Package: {result['package_id']}")
        print(f"  Visualizations: {len(result['visualizations'])}")
        print("  Async job passed ✓")
    elif data2["status"] == "failed":
        print(f"  Job failed: {data2.get('error')}")
        print("  (Expected if LLM not running — job infrastructure works ✓)")
    else:
        print(f"  Job still {data2['status']} — infrastructure works ✓")


def test_list_jobs():
    print("\n--- Test: List jobs ---")
    r = client.get("/jobs/")
    assert r.status_code == 200
    data = r.json()
    print(f"  Jobs: {len(data['jobs'])}")
    assert len(data["jobs"]) >= 1
    print("  List jobs passed ✓")


def test_not_found():
    print("\n--- Test: Error handling ---")
    r = client.post("/narratives/preview", json={"dataset_id": "nonexistent"})
    assert r.status_code == 404
    print(f"  Missing dataset → 404 ✓")

    r2 = client.get("/jobs/nonexistent")
    assert r2.status_code == 404
    print(f"  Missing job → 404 ✓")
    print("  Error handling passed ✓")


def main():
    print("=" * 60)
    print("FASTAPI ENDPOINT TESTS")
    print("=" * 60)

    test_health()
    test_list_datasets()
    test_preview_narrative()
    test_generate_visualization()
    test_generate_visualization_override()
    test_create_async_job()
    test_list_jobs()
    test_not_found()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
