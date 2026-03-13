"""
Test script for the data profiling pipeline.
Run from project root: python -m tests.test_profiling_pipeline

Tests: profiling → template matching → caching
"""

import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data.profiling.profiler import profile_dataset, profile_dataset_duckdb
from data.profiling.matcher import match_template
from data.profiling.profile_cache import ProfileCache
from data.profiling.template_definitions import ALL_TEMPLATES, TemplateType


# ---------------------------------------------------------------------------
# Sample datasets
# ---------------------------------------------------------------------------

def make_time_series_df() -> pd.DataFrame:
    """Simulates a monthly city budget time series."""
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=36, freq="MS"),
        "department": ["Public Works", "Education", "Health"] * 12,
        "budget_isk": [150_000, 320_000, 210_000] * 12,
        "expenditure_isk": [142_000, 305_000, 198_000] * 12,
    })


def make_categorical_df() -> pd.DataFrame:
    """Simulates district-level service complaints."""
    return pd.DataFrame({
        "district": ["Vesturbær", "Hlíðar", "Laugardalur", "Breiðholt", "Grafarvogur", "Háaleiti"],
        "complaint_count": [45, 78, 32, 120, 55, 67],
        "avg_resolution_days": [3.2, 5.1, 2.8, 7.4, 4.0, 3.9],
        "satisfaction_score": [4.2, 3.5, 4.6, 2.9, 3.8, 4.0],
    })


def make_geospatial_df() -> pd.DataFrame:
    """Simulates locations of city services."""
    return pd.DataFrame({
        "name": ["City Hall", "Harpa", "Laugardalslaug", "Kringlan", "BSÍ Bus Terminal"],
        "type": ["Government", "Culture", "Recreation", "Shopping", "Transport"],
        "latitude": [64.1466, 64.1504, 64.1447, 64.1305, 64.1373],
        "longitude": [-21.9426, -21.9327, -21.8748, -21.8954, -21.9218],
        "visitor_count": [500, 1200, 800, 3000, 2500],
    })


def make_ambiguous_df() -> pd.DataFrame:
    """A dataset that could match multiple templates."""
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=12, freq="MS"),
        "district": ["Vesturbær", "Hlíðar", "Laugardalur"] * 4,
        "latitude": [64.15] * 12,
        "longitude": [-21.94] * 12,
        "value": range(100, 112),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_profiling():
    """Test the profiling engine with different dataset types."""
    print("\n" + "=" * 60)
    print("TEST: Data profiling engine")
    print("=" * 60)

    datasets = {
        "time_series": make_time_series_df(),
        "categorical": make_categorical_df(),
        "geospatial": make_geospatial_df(),
    }

    profiles = {}
    for name, df in datasets.items():
        print(f"\n--- Profiling: {name} ---")
        profile = profile_dataset(df, dataset_id=name, source=f"test_{name}")
        profiles[name] = profile

        print(f"  Rows: {profile.row_count}, Columns: {profile.column_count}")
        print(f"  Null rate: {profile.total_null_rate:.1%}")
        print(f"  Types: {profile.column_types_summary}")
        for col in profile.columns:
            stats_type = ""
            if col.numeric_stats:
                stats_type = f"range=[{col.numeric_stats.min}, {col.numeric_stats.max}]"
            elif col.categorical_stats:
                stats_type = f"unique={col.categorical_stats.unique_count}"
            elif col.temporal_stats:
                stats_type = f"range={col.temporal_stats.min_date} to {col.temporal_stats.max_date}"
            elif col.geospatial_stats:
                stats_type = "geo"
            print(f"    {col.name}: {col.semantic_type} (null={col.null_rate:.0%}) {stats_type}")

    # Verify type inference
    assert profiles["time_series"].has_temporal, "Time series should have temporal"
    assert profiles["time_series"].has_numerical, "Time series should have numerical"
    assert profiles["categorical"].has_categorical, "Categorical should have categorical"
    assert profiles["categorical"].has_numerical, "Categorical should have numerical"
    assert profiles["geospatial"].has_geospatial, "Geospatial should have geospatial"

    print("\n  Profiling tests passed ✓")
    return profiles


def test_duckdb_profiling():
    """Test DuckDB-based profiling matches pandas profiling."""
    print("\n" + "=" * 60)
    print("TEST: DuckDB profiling")
    print("=" * 60)

    df = make_time_series_df()

    pandas_profile = profile_dataset(df, dataset_id="test_pandas")
    duckdb_profile = profile_dataset_duckdb(df, dataset_id="test_duckdb")

    print(f"  Pandas: {pandas_profile.column_types_summary}")
    print(f"  DuckDB: {duckdb_profile.column_types_summary}")

    assert pandas_profile.row_count == duckdb_profile.row_count
    assert pandas_profile.column_count == duckdb_profile.column_count

    # Check that numeric stats are close
    for p_col, d_col in zip(pandas_profile.columns, duckdb_profile.columns):
        assert p_col.semantic_type == d_col.semantic_type, (
            f"Type mismatch for {p_col.name}: {p_col.semantic_type} vs {d_col.semantic_type}"
        )
        if p_col.numeric_stats and d_col.numeric_stats:
            assert abs(p_col.numeric_stats.mean - d_col.numeric_stats.mean) < 0.01

    print("  DuckDB profiling matches pandas ✓")


def test_template_matching(profiles: dict):
    """Test template matching against profiled datasets."""
    print("\n" + "=" * 60)
    print("TEST: Template matching")
    print("=" * 60)

    expected = {
        "time_series": TemplateType.TIME_SERIES,
        "categorical": TemplateType.CATEGORICAL,
        "geospatial": TemplateType.GEOSPATIAL,
    }

    for name, profile in profiles.items():
        print(f"\n--- Matching: {name} ---")
        result = match_template(profile)

        print(f"  Profile types: {result.profile_summary}")
        for m in result.all_matches:
            status = "✓ BEST" if result.best_match and m.template_id == result.best_match.template_id else ""
            viable = "viable" if m.is_viable else "not viable"
            print(f"    {m.template_name}: score={m.score:.3f} ({viable}) {status}")
            if m.matched_columns:
                print(f"      Columns: {m.matched_columns}")
            if m.missing_required:
                print(f"      Missing: {m.missing_required}")

        assert result.best_match is not None, f"No match for {name}"
        assert result.best_match.template_id == expected[name], (
            f"Expected {expected[name]} for {name}, got {result.best_match.template_id}"
        )

    print("\n  Template matching tests passed ✓")


def test_ambiguous_matching():
    """Test matching with a dataset that fits multiple templates."""
    print("\n" + "=" * 60)
    print("TEST: Ambiguous template matching")
    print("=" * 60)

    df = make_ambiguous_df()
    profile = profile_dataset(df, dataset_id="ambiguous")
    result = match_template(profile)

    print(f"  Profile types: {result.profile_summary}")
    viable_count = sum(1 for m in result.all_matches if m.is_viable)
    print(f"  Viable matches: {viable_count}")
    for m in result.all_matches:
        viable = "viable" if m.is_viable else "not viable"
        best = "← BEST" if result.best_match and m.template_id == result.best_match.template_id else ""
        print(f"    {m.template_name}: score={m.score:.3f} ({viable}) {best}")

    assert result.best_match is not None, "Should find at least one viable match"
    assert viable_count >= 2, "Ambiguous dataset should match multiple templates"
    print("  Ambiguous matching test passed ✓")


def test_caching():
    """Test caching profiles and match results as Parquet."""
    print("\n" + "=" * 60)
    print("TEST: Profile caching")
    print("=" * 60)

    tmp_dir = tempfile.mkdtemp()
    try:
        cache = ProfileCache(cache_dir=tmp_dir)

        # Profile and cache
        df = make_time_series_df()
        profile = profile_dataset(df, dataset_id="cache_test", source="test")
        result = match_template(profile)

        cache.save_profile(profile)
        cache.save_match(result)

        # Verify files exist
        assert cache.has_profile("cache_test"), "Profile should be cached"
        assert cache.has_match("cache_test"), "Match should be cached"
        print(f"  Cached files: {os.listdir(tmp_dir)}")

        # Load profile table
        profile_df = cache.load_profile_table("cache_test")
        assert profile_df is not None
        assert len(profile_df) == profile.column_count
        print(f"  Profile table: {len(profile_df)} columns")
        print(f"  Columns: {list(profile_df['column_name'])}")

        # Load metadata
        meta = cache.load_profile_meta("cache_test")
        assert meta is not None
        assert meta["row_count"] == profile.row_count
        print(f"  Metadata: row_count={meta['row_count']}, types={meta['column_types_summary']}")

        # Load match table
        match_df = cache.load_match_table("cache_test")
        assert match_df is not None
        print(f"  Match table: {len(match_df)} template scores")

        # Get best match shortcut
        best = cache.get_best_match("cache_test")
        assert best is not None
        assert best["template_id"] == "time_series"
        print(f"  Best match from cache: {best['template_name']} (score={best['score']:.3f})")

        # List cached
        cached = cache.list_cached()
        assert "cache_test" in cached
        print(f"  Cached datasets: {cached}")

        # Clear
        cache.clear("cache_test")
        assert not cache.has_profile("cache_test")
        print("  Cache cleared successfully")

        print("  Caching tests passed ✓")

    finally:
        shutil.rmtree(tmp_dir)


def main():
    print("=" * 60)
    print("DATA PROFILING PIPELINE TESTS")
    print("=" * 60)

    profiles = test_profiling()
    test_duckdb_profiling()
    test_template_matching(profiles)
    test_ambiguous_matching()
    test_caching()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
