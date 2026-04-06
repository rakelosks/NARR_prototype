# Test Suite Overview

This directory contains pytest-based automated tests for the NARR prototype.

## Structure

- `test_api/` — API endpoint behavior and health checks
- `test_data/` — ingestion, profiling, analytics, and CKAN integration
- `test_llm/` — intent parsing and narrative generation behavior

## Running Tests

From the project root:

```bash
python -m pytest -q
```

If you are using a virtual environment:

```bash
source venv/bin/activate
python -m pytest -q
```

## Notes

- Some tests are integration-like and may rely on optional external services.
- Prefer deterministic/unit tests for CI and keep network-dependent tests clearly marked.
