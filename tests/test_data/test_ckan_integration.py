"""
Quick test script for the CKAN connector.
Run from project root: python -m tests.test_data.test_ckan_integration

Defaults to Reykjavik but supports any CKAN portal URL.
"""

import asyncio
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ingestion.ckan_client import CKANClient, CKANError
from data.storage.catalog_index import CatalogIndex

DEFAULT_PORTAL_URL = os.getenv(
    "CKAN_TEST_PORTAL_URL",
    "https://gagnagatt.reykjavik.is/en/api/3",
)
DEFAULT_PORTAL_LANGUAGE = os.getenv("CKAN_TEST_PORTAL_LANGUAGE", "is")


async def test_list_datasets(portal_url: str):
    """Test listing dataset names from the portal."""
    print("\n--- Test: List datasets ---")
    client = CKANClient(portal_url)
    names = await client.list_datasets(limit=10)
    print(f"Found {len(names)} dataset names (showing first 10):")
    for name in names:
        print(f"  - {name}")
    assert len(names) > 0, "Expected at least one dataset"
    return names


async def test_search_datasets(portal_url: str):
    """Test searching datasets by keyword."""
    print("\n--- Test: Search datasets ---")
    client = CKANClient(portal_url)
    datasets = await client.search_datasets("transport", rows=5)
    print(f"Search 'transport' returned {len(datasets)} results:")
    for ds in datasets:
        formats = [r.normalized_format for r in ds.resources]
        print(f"  - {ds.title} ({ds.num_resources} resources: {formats})")
    return datasets


async def test_get_dataset(portal_url: str, dataset_name: str):
    """Test fetching full metadata for a single dataset."""
    print(f"\n--- Test: Get dataset '{dataset_name}' ---")
    client = CKANClient(portal_url)
    dataset = await client.get_dataset(dataset_name)
    print(f"Title: {dataset.title}")
    print(f"Description: {dataset.notes[:200]}..." if len(dataset.notes) > 200 else f"Description: {dataset.notes}")
    print(f"Organization: {dataset.organization}")
    print(f"Tags: {dataset.tags}")
    print(f"Resources ({dataset.num_resources}):")
    for r in dataset.resources:
        supported = "✓" if r.is_supported else "✗"
        print(f"  [{supported}] {r.name or r.id} — {r.format} — {r.url[:80]}...")
    print(f"Supported resources: {len(dataset.supported_resources)}")
    return dataset


async def test_download_resource(portal_url: str, dataset_name: str):
    """Test downloading a CSV resource and parsing to DataFrame."""
    print(f"\n--- Test: Download resource from '{dataset_name}' ---")
    client = CKANClient(portal_url)
    dataset = await client.get_dataset(dataset_name)

    csv_resources = [r for r in dataset.resources if r.normalized_format == "csv"]
    if not csv_resources:
        print("No CSV resources found, trying any supported format...")
        csv_resources = dataset.supported_resources

    if not csv_resources:
        print("No supported resources found in this dataset. Skipping.")
        return None

    resource = csv_resources[0]
    print(f"Downloading: {resource.name or resource.id} ({resource.format})")

    df = await client.download_resource_as_dataframe(resource)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"First 3 rows:")
    print(df.head(3).to_string())
    return df


async def test_catalog_index(portal_url: str):
    """Test building and querying the catalog index."""
    print("\n--- Test: Catalog index ---")
    client = CKANClient(portal_url)

    # Use a temp database for testing
    index = CatalogIndex(db_path=":memory:")

    # Build catalog (fetches all datasets)
    print("Building catalog index...")
    await index.refresh(client, portal_url=portal_url)

    count = index.count(portal_url=portal_url)
    print(f"Total datasets indexed: {count}")

    # Search the local index
    results = index.search(query="transport", portal_url=portal_url, limit=5)
    print(f"\nLocal search for 'transport': {len(results)} results")
    for r in results:
        print(f"  - {r['title']} (formats: {r['resource_formats']})")

    # Filter by format
    csv_results = index.search(format_filter="csv", portal_url=portal_url, limit=5)
    print(f"\nDatasets with CSV resources: {len(csv_results)}")
    for r in csv_results:
        print(f"  - {r['title']}")

    # Check sync log
    last = index.last_sync(portal_url)
    print(f"\nLast sync: {last}")

    return index


async def main(portal_url: str, portal_language: str):
    print("=" * 60)
    print("CKAN Connector Integration Test")
    print(f"Portal: {portal_url}")
    print(f"Language: {portal_language}")
    print("=" * 60)

    try:
        # 1. List datasets
        names = await test_list_datasets(portal_url)

        # 2. Search
        await test_search_datasets(portal_url)

        # 3. Get single dataset
        if names:
            dataset = await test_get_dataset(portal_url, names[0])

            # 4. Download a resource
            await test_download_resource(portal_url, names[0])

        # 5. Catalog index
        await test_catalog_index(portal_url)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)

    except CKANError as e:
        print(f"\nCKAN Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CKAN integration checks")
    parser.add_argument(
        "--portal-url",
        default=DEFAULT_PORTAL_URL,
        help=f"CKAN API base URL (default: {DEFAULT_PORTAL_URL})",
    )
    parser.add_argument(
        "--portal-language",
        default=DEFAULT_PORTAL_LANGUAGE,
        help=f"Portal language code (default: {DEFAULT_PORTAL_LANGUAGE})",
    )
    args = parser.parse_args()
    asyncio.run(main(args.portal_url, args.portal_language))
