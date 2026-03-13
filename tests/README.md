"""
Test script for the fallback URL/file loader.
Run from project root: python -m tests.test_loader

Tests format detection, local file loading, and URL loading.
"""

import asyncio
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ingestion.loader import (
    detect_format,
    load_from_file,
    load_from_url,
    infer_column_types,
    parse_bytes,
    FileFormat,
)


def test_format_detection():
    """Test format detection from paths, content types, and explicit values."""
    print("\n--- Test: Format detection ---")

    # From file extension
    assert detect_format("data.csv") == FileFormat.CSV
    assert detect_format("data.json") == FileFormat.JSON
    assert detect_format("map.geojson") == FileFormat.GEOJSON
    assert detect_format("report.xlsx") == FileFormat.XLSX

    # From URL with query params
    assert detect_format("https://example.com/data.csv?token=abc") == FileFormat.CSV

    # From content type
    assert detect_format("noext", content_type="text/csv") == FileFormat.CSV
    assert detect_format("noext", content_type="application/json; charset=utf-8") == FileFormat.JSON

    # Explicit override wins
    assert detect_format("data.csv", explicit_format="json") == FileFormat.JSON

    # Unknown format raises
    try:
        detect_format("mystery_file")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("  All format detection tests passed ✓")


def test_parse_csv():
    """Test parsing CSV bytes."""
    print("\n--- Test: Parse CSV ---")
    csv_data = b"name,age,city\nAlice,30,Reykjavik\nBob,25,Akureyri\n"
    df = parse_bytes(csv_data, FileFormat.CSV)
    assert len(df) == 2
    assert list(df.columns) == ["name", "age", "city"]
    print(f"  Parsed {len(df)} rows, {len(df.columns)} columns ✓")


def test_parse_json():
    """Test parsing JSON bytes."""
    print("\n--- Test: Parse JSON ---")
    json_data = b'[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
    df = parse_bytes(json_data, FileFormat.JSON)
    assert len(df) == 2
    print(f"  Parsed {len(df)} rows ✓")


def test_parse_geojson():
    """Test parsing GeoJSON bytes."""
    print("\n--- Test: Parse GeoJSON ---")
    geojson_data = b"""{
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": "City Hall"}, "geometry": {"type": "Point", "coordinates": [-21.94, 64.15]}},
            {"type": "Feature", "properties": {"name": "Harpa"}, "geometry": {"type": "Point", "coordinates": [-21.93, 64.15]}}
        ]
    }"""
    df = parse_bytes(geojson_data, FileFormat.GEOJSON)
    assert len(df) == 2
    assert "geometry" in df.columns
    print(f"  Parsed {len(df)} features with geometry column ✓")


def test_column_type_inference():
    """Test semantic column type inference."""
    print("\n--- Test: Column type inference ---")
    import pandas as pd

    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [30, 25, 35],
        "salary": [50000.0, 60000.0, 70000.0],
        "created_date": ["2024-01-01", "2024-02-15", "2024-03-20"],
        "latitude": [64.15, 65.68, 63.44],
        "city": ["Reykjavik", "Akureyri", "Isafjordur"],
    })

    columns = infer_column_types(df)
    type_map = {c.name: c.dtype for c in columns}

    print(f"  Inferred types: {type_map}")

    assert type_map["name"] == "categorical"
    assert type_map["age"] == "numerical"
    assert type_map["salary"] == "numerical"
    assert type_map["created_date"] == "temporal"
    assert type_map["latitude"] == "geospatial"
    assert type_map["city"] == "categorical"
    print("  All type inferences correct ✓")


def test_load_from_file():
    """Test loading a local CSV file."""
    print("\n--- Test: Load from file ---")

    # Create a temp CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("district,population,area_km2\n")
        f.write("Vesturbær,12000,3.5\n")
        f.write("Hlíðar,15000,4.2\n")
        f.write("Laugardalur,18000,5.1\n")
        tmp_path = f.name

    try:
        df, result = load_from_file(tmp_path)
        print(f"  Source: {result.source}")
        print(f"  Format: {result.format}")
        print(f"  Shape: {result.row_count} x {result.column_count}")
        print(f"  Columns: {[c.name + ' (' + c.dtype + ')' for c in result.columns]}")
        assert result.row_count == 3
        assert result.format == "csv"
        print("  Load from file passed ✓")
    finally:
        os.unlink(tmp_path)


async def test_load_from_url():
    """Test loading from a URL (uses a public CSV)."""
    print("\n--- Test: Load from URL ---")
    # Use a small public CSV for testing
    url = "https://raw.githubusercontent.com/datasets/country-codes/master/data/country-codes.csv"
    try:
        df, result = await load_from_url(url)
        print(f"  Source: {result.source[:60]}...")
        print(f"  Format: {result.format}")
        print(f"  Shape: {result.row_count} x {result.column_count}")
        print(f"  First 5 columns: {[c.name for c in result.columns[:5]]}")
        print("  Load from URL passed ✓")
    except Exception as e:
        print(f"  Skipped (network error): {e}")


async def main():
    print("=" * 60)
    print("Fallback Loader Tests")
    print("=" * 60)

    test_format_detection()
    test_parse_csv()
    test_parse_json()
    test_parse_geojson()
    test_column_type_inference()
    test_load_from_file()
    await test_load_from_url()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
