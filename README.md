# Smart City Narrative Visualization Platform

AI-powered open data narrative visualization system for smart cities.

## Architecture

- **FastAPI** backend with async request handling (Uvicorn)
- **DuckDB** for analytical queries, **SQLite** for metadata, **Parquet** for caching
- **Frictionless** for dataset validation, **Pydantic** for schema enforcement
- **Provider-agnostic LLM** interface (Ollama default, OpenAI optional)
- **Vega-Lite** for interactive visualizations
- **Streamlit** proof-of-concept client

## Quick Start

### Local Development

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment config
cp .env.example .env

# 4. Start the API server
uvicorn app.main:app --reload

# 5. (In a separate terminal) Start the Streamlit client
streamlit run client/streamlit_app.py
```

### Docker

```bash
docker compose up
```

This starts the API (port 8000), Streamlit client (port 8501), and Ollama (port 11434).

## Project Structure

```
├── app/                    # FastAPI backend
│   ├── main.py             # Application entry point
│   ├── api/                # Route definitions
│   ├── models/             # Pydantic schemas
│   └── services/           # Business logic
├── data/                   # Data processing layer
│   ├── ingestion/          # Loaders, parsers
│   ├── validation/         # Frictionless validation
│   ├── storage/            # DuckDB, SQLite
│   └── cache/              # Parquet snapshots
├── llm/                    # LLM integration
│   ├── interface.py        # Provider-agnostic interface
│   └── providers/          # Ollama, OpenAI implementations
├── visualization/          # Vega-Lite spec generation
│   ├── charts.py           # Chart type selection & specs
│   └── templates/          # Chart configurations
├── client/                 # Streamlit proof-of-concept
├── tests/                  # Test suite
├── config.py               # Application settings
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container image
└── docker-compose.yml      # Full stack orchestration
```

## API Endpoints

- `GET /health` — Health check
- `GET /datasets/` — List registered datasets
- `POST /datasets/ingest` — Ingest a new dataset
- `POST /narratives/generate` — Generate a narrative
- `POST /visualizations/generate` — Generate a Vega-Lite spec

## License

TBD
