"""
Test script for the analytics and visualization pipeline.
Run from project root: python -m tests.test_analytics_pipeline

Tests: analytics engine → chart selection → spec generation → evidence bundle
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data.profiling.profiler import profile_dataset
from data.profiling.matcher import match_template
from data.analytics.analytics import AnalyticsEngine
from visualization.charts import select_chart_type, generate_spec
from data.analytics.evidence_bundle import BundleBuilder


# ---------------------------------------------------------------------------
# Sample datasets (same as profiling tests)
# ---------------------------------------------------------------------------

def make_time_series_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=36, freq="MS"),
        "department": ["Public Works", "Education", "Health"] * 12,
        "budget_isk": [150_000 + i * 1000 for i in range(36)],
        "expenditure_isk": [142_000 + i * 900 for i in range(36)],
    })


def make_categorical_df() -> pd.DataFrame:
    return pd.DataFrame({
        "district": ["Vesturbær", "Hlíðar", "Laugardalur", "Breiðholt", "Grafarvogur", "Háaleiti"],
        "complaint_count": [45, 78, 32, 120, 55, 67],
        "avg_resolution_days": [3.2, 5.1, 2.8, 7.4, 4.0, 3.9],
        "satisfaction_score": [4.2, 3.5, 4.6, 2.9, 3.8, 4.0],
    })


def make_geospatial_df() -> pd.DataFrame:
    return pd.DataFrame({
        "name": ["City Hall", "Harpa", "Laugardalslaug", "Kringlan", "BSÍ Bus Terminal"],
        "type": ["Government", "Culture", "Recreation", "Shopping", "Transport"],
        "latitude": [64.1466, 64.1504, 64.1447, 64.1305, 64.1373],
        "longitude": [-21.9426, -21.9327, -21.8748, -21.8954, -21.9218],
        "visitor_count": [500, 1200, 800, 3000, 2500],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_analytics_engine():
    """Test the analytics engine for each template type."""
    print("\n" + "=" * 60)
    print("TEST: Analytics engine")
    print("=" * 60)

    engine = AnalyticsEngine()
    datasets = {
        "time_series": make_time_series_df(),
        "categorical": make_categorical_df(),
        "geospatial": make_geospatial_df(),
    }

    for name, df in datasets.items():
        print(f"\n--- Analytics: {name} ---")
        profile = profile_dataset(df, dataset_id=name)
        match = match_template(profile)

        result = engine.analyze(df, match)
        print(f"  Template: {result.template_type.value}")
        print(f"  Columns: {result.matched_columns}")
        print(f"  Aggregation rows: {len(result.aggregation_table)}")

        # Print some key metrics
        metrics = result.metrics
        if name == "time_series":
            print(f"  Date range: {metrics.get('date_range')}")
            print(f"  Trends: {list(metrics.get('trend', {}).keys())}")
            for col, trend in metrics.get("trend", {}).items():
                print(f"    {col}: {trend['direction']} ({trend['pct_change']:+.1f}%)")

        elif name == "categorical":
            print(f"  Categories: {metrics.get('total_categories')}")
            for col, ranking in metrics.get("rankings", {}).items():
                if ranking:
                    print(f"    {col} top: {ranking[0]['category']} = {ranking[0]['value']}")

        elif name == "geospatial":
            print(f"  Points: {metrics.get('total_points')}")
            print(f"  Bounding box: {metrics.get('bounding_box')}")

    print("\n  Analytics engine tests passed ✓")


def test_chart_selection():
    """Test chart type selection for each template."""
    print("\n" + "=" * 60)
    print("TEST: Chart type selection")
    print("=" * 60)

    engine = AnalyticsEngine()
    datasets = {
        "time_series": make_time_series_df(),
        "categorical": make_categorical_df(),
        "geospatial": make_geospatial_df(),
    }

    for name, df in datasets.items():
        print(f"\n--- Chart selection: {name} ---")
        profile = profile_dataset(df, dataset_id=name)
        match = match_template(profile)
        result = engine.analyze(df, match)

        selection = select_chart_type(result.template_type, result.metrics)
        print(f"  Primary: {selection.primary_chart}")
        if selection.secondary_chart:
            print(f"  Secondary: {selection.secondary_chart}")
        print(f"  Reason: {selection.reason}")

    print("\n  Chart selection tests passed ✓")


def test_spec_generation():
    """Test Vega-Lite spec generation."""
    print("\n" + "=" * 60)
    print("TEST: Vega-Lite spec generation")
    print("=" * 60)

    engine = AnalyticsEngine()
    datasets = {
        "time_series": make_time_series_df(),
        "categorical": make_categorical_df(),
        "geospatial": make_geospatial_df(),
    }

    for name, df in datasets.items():
        print(f"\n--- Spec generation: {name} ---")
        profile = profile_dataset(df, dataset_id=name)
        match = match_template(profile)
        result = engine.analyze(df, match)
        selection = select_chart_type(result.template_type, result.metrics)

        spec = generate_spec(
            chart_type=selection.primary_chart,
            data=result.aggregation_table,
            columns=result.matched_columns,
            metrics=result.metrics,
            title=f"Test {name} chart",
        )

        assert "$schema" in spec, "Spec should have $schema"
        assert "data" in spec, "Spec should have data"
        assert "mark" in spec, "Spec should have mark"
        assert "encoding" in spec, "Spec should have encoding"
        assert len(spec["data"]["values"]) > 0, "Spec should have data values"

        print(f"  Chart type: {selection.primary_chart}")
        print(f"  Mark: {spec['mark']}")
        print(f"  Encoding fields: {list(spec['encoding'].keys())}")
        print(f"  Data rows: {len(spec['data']['values'])}")

    print("\n  Spec generation tests passed ✓")


def test_evidence_bundle():
    """Test the full evidence bundle builder."""
    print("\n" + "=" * 60)
    print("TEST: Evidence bundle builder")
    print("=" * 60)

    builder = BundleBuilder()
    datasets = {
        "time_series": ("Reykjavik Monthly Budget", make_time_series_df()),
        "categorical": ("District Complaints", make_categorical_df()),
        "geospatial": ("City Service Locations", make_geospatial_df()),
    }

    for name, (title, df) in datasets.items():
        print(f"\n--- Bundle: {name} ---")
        profile = profile_dataset(df, dataset_id=name, source=f"test/{name}.csv")
        match = match_template(profile)

        bundle = builder.build(df, profile, match, title=title)

        print(f"  Template: {bundle.template_type}")
        print(f"  Size: {bundle.row_count} × {bundle.column_count}")
        print(f"  Column types: {bundle.column_summary}")
        print(f"  Mapped columns: {bundle.matched_columns}")
        print(f"  Visualizations: {len(bundle.visualizations)}")
        for viz in bundle.visualizations:
            primary = "PRIMARY" if viz.is_primary else "secondary"
            print(f"    [{primary}] {viz.chart_type}: {viz.title}")
        print(f"  Key findings: {len(bundle.narrative_context.key_findings)}")
        for finding in bundle.narrative_context.key_findings:
            print(f"    - {finding}")

        # Test LLM context formatting
        llm_context = bundle.to_llm_context()
        assert len(llm_context) > 100, "LLM context should be substantial"
        print(f"  LLM context length: {len(llm_context)} chars")
        print(f"  --- LLM Context Preview ---")
        preview_lines = llm_context.split("\n")[:15]
        for line in preview_lines:
            print(f"  | {line}")
        print(f"  | ... ({len(llm_context.split(chr(10)))} lines total)")

        # Verify bundle structure
        assert bundle.dataset_id == name
        assert bundle.template_type in ["time_series", "categorical", "geospatial"]
        assert len(bundle.visualizations) >= 1
        assert bundle.narrative_context is not None
        assert len(bundle.narrative_context.key_findings) >= 1

    print("\n  Evidence bundle tests passed ✓")


def test_spec_json_export():
    """Test that generated specs are valid JSON."""
    print("\n" + "=" * 60)
    print("TEST: Spec JSON export")
    print("=" * 60)

    builder = BundleBuilder()
    df = make_categorical_df()
    profile = profile_dataset(df, dataset_id="json_test")
    match = match_template(profile)
    bundle = builder.build(df, profile, match)

    for viz in bundle.visualizations:
        json_str = json.dumps(viz.vega_lite_spec, indent=2, default=str)
        parsed = json.loads(json_str)
        assert parsed["$schema"].startswith("https://vega.github.io")
        print(f"  {viz.chart_type} spec: {len(json_str)} bytes, valid JSON ✓")

    print("  JSON export tests passed ✓")


def main():
    print("=" * 60)
    print("ANALYTICS & VISUALIZATION PIPELINE TESTS")
    print("=" * 60)

    test_analytics_engine()
    test_chart_selection()
    test_spec_generation()
    test_evidence_bundle()
    test_spec_json_export()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
