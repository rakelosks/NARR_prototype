# NARR — Narrative Visualization for Open Data

NARR is an AI-powered pipeline that connects to city open data portals (CKAN) and
produces editorial data stories from natural language prompts. It profiles datasets
automatically, selects appropriate visualizations, and uses an LLM to write
human-readable narratives grounded in the actual data.

Built as an undergraduate capstone project at IE University exploring how
generative AI can make public data more accessible to non-technical audiences.

## Architecture

| Layer | Technology |
|-------|-----------|
| **API** | FastAPI, Uvicorn, async throughout |
| **Analytics** | DuckDB (columnar queries), Pandas, PyArrow |
| **Storage** | SQLite (catalog & metadata), Parquet (dataset cache) |
| **Validation** | Frictionless (data quality), Pydantic (schemas) |
| **LLM** | Provider-agnostic — Ollama (default, local) or OpenAI |
| **Visualization** | Vega-Lite JSON specs, rendered client-side |
| **Client** | Streamlit proof-of-concept |
| **Deployment** | Docker & Docker Compose |

## Quick Start

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai) running locally (or an OpenAI API key)

### Local Development

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies (use python -m pip to avoid PATH issues)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Copy environment config and edit as needed
cp .env.example .env

# 4. Sanity check (verifies dependencies + imports)
python -c "import fastapi, uvicorn, pandas, duckdb; import app.main; print('OK: environment + imports')"

# 5. Pull the default LLM model (if using Ollama)
ollama pull qwen3:4b

# 6. Start the API server (use python -m so the venv’s Uvicorn is found)
python -m uvicorn app.main:app --reload

# 7. (In a separate terminal) Start the Streamlit client
python -m streamlit run client/streamlit_app.py
```

The API will be available at `http://localhost:8000` and the Streamlit client at
`http://localhost:8501`.

### Tests (optional)

```bash
python -m pytest -q
```

### Docker

```bash
# Start API + Streamlit
docker compose up

# If using bundled Ollama, include the ollama profile:
docker compose --profile ollama up
```

Notes:
- In `docker-compose.yml`, the API container uses `OLLAMA_BASE_URL=http://ollama:11434` by default (Docker service hostname).
- For OpenAI mode, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` in `.env`.
- To use a host Ollama instance from Docker, override `OLLAMA_BASE_URL` accordingly for your host setup.

| Service | Port |
|---------|------|
| API | 8000 |
| Streamlit | 8501 |
| Ollama (optional) | 11434 |

## Project Structure

```
├── app/                    # FastAPI backend
│   ├── main.py             # Application entry point & middleware
│   ├── api/                # Route handlers
│   │   ├── datasets.py     # Catalog & dataset configuration routes
│   │   ├── narratives.py   # Narrative generation routes
│   │   ├── visualizations.py # Visualization generation routes
│   │   └── jobs.py         # Async job management routes
│   ├── middleware/          # Auth & rate limiting
│   └── models/             # Pydantic request/response schemas
├── data/                   # Data processing layer
│   ├── ingestion/          # CKAN client, loaders, parsers
│   ├── profiling/          # Column classification & keyword dictionary
│   ├── analytics/          # Evidence bundle builder, DuckDB analytics
│   ├── validation/         # Frictionless data validation
│   ├── storage/            # Catalog index & metadata store (SQLite)
│   └── cache/              # Parquet snapshot cache
├── llm/                    # LLM integration
│   ├── interface.py        # Provider-agnostic LLM interface
│   ├── providers/          # Ollama & OpenAI implementations
│   ├── intent.py           # Natural language → structured intent
│   ├── prompts.py          # Prompt assembly with guardrails
│   ├── narrative.py        # Narrative generation & validation
│   └── ...                 # Provider integration modules
├── visualization/          # Vega-Lite spec generation
│   ├── charts.py           # Chart type selection & spec building
│   └── templates/          # Chart template configurations
├── client/                 # Streamlit proof-of-concept client
│   └── streamlit_app.py    # Main client application
├── tests/                  # Test suite (pytest)
├── config.py               # Centralized settings (env vars)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container image
├── docker-compose.yml      # Full stack orchestration
├── .env.example            # Environment variable template
└── LICENSE                 # MIT License
```

## API Endpoints

All endpoints are documented interactively at `/docs` (Swagger UI) and `/redoc`
when the server is running.

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check — returns status and portal configuration |

### Datasets

| Method | Path | Description |
|--------|------|-------------|
| POST | `/datasets/catalog/refresh` | Refresh the local catalog index from a CKAN portal |
| GET | `/datasets/catalog/search` | Search the local catalog index |
| GET | `/datasets/catalog/{dataset_id}` | Get a single catalog entry by ID |
| GET | `/datasets/` | List datasets with saved template configurations |
| GET | `/datasets/{dataset_id}` | Get saved template configuration for a dataset |

### Narratives

| Method | Path | Description |
|--------|------|-------------|
| POST | `/narratives/preview` | Build evidence bundle without LLM (metrics + visualizations only) |
| POST | `/narratives/generate` | Full narrative generation with LLM |
| POST | `/narratives/ask` | End-to-end: natural language question → data story |

### Visualizations

| Method | Path | Description |
|--------|------|-------------|
| POST | `/visualizations/generate` | Generate Vega-Lite spec(s) for a cached dataset |

### Jobs

| Method | Path | Description |
|--------|------|-------------|
| POST | `/jobs/generate` | Create an async narrative generation job (returns job ID) |
| GET | `/jobs/{job_id}` | Poll job status and retrieve results |
| GET | `/jobs/` | List recent jobs |

## Configuration

Copy `.env.example` to `.env` and adjust. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `OPENAI_API_KEY` | _(empty)_ | Required when `LLM_PROVIDER=openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama base URL (local dev default) |
| `OLLAMA_INTENT_MODEL` | `qwen3:4b` | Model for intent parsing |
| `OLLAMA_GENERATION_MODEL` | `qwen3:4b` | Model for narrative generation |
| `OLLAMA_INTENT_THINK` | `false` | Enable/disable Ollama "thinking" mode for intent parsing |
| `OLLAMA_GENERATION_THINK` | `true` | Enable/disable Ollama "thinking" mode for narrative generation |
| `OLLAMA_TIMEOUT` | `180` | Timeout (seconds) for Ollama requests |
| `CKAN_PORTAL_URL` | Reykjavik portal | CKAN API base URL |
| `CKAN_PORTAL_LANGUAGE` | `is` | Portal metadata language (ISO 639-1) |
| `NARR_API_KEY` | _(empty)_ | API key for authentication (empty = disabled) |
| `RATE_LIMIT_RPM` | `60` | Rate limit per client per minute |
| `TRUST_PROXY_HEADERS` | `false` | Trust `X-Forwarded-For` for rate limiting (only behind trusted proxy) |

See `.env.example` for the full list.

## Prototype Security Posture

This repository is currently optimized for local development and proof-of-concept
demonstration. By default, authentication is disabled when `NARR_API_KEY` is empty,
and CORS is permissive. Before any public deployment, set `NARR_API_KEY` and
restrict allowed origins.

## How It Works

1. **Intent parsing** — The user's natural language prompt is parsed by an LLM
   into a structured intent (dataset keywords, analysis type, audience level,
   language).

2. **Dataset discovery** — The intent's keywords are used to search the local
   CKAN catalog index (in both the user's language and the portal's language).

3. **Data profiling** — The matched dataset is fetched, cached as Parquet, and
   profiled: column types are classified, temporal/categorical/measure roles
   are detected, and a keyword dictionary maps columns to canonical concepts.

4. **Template matching** — Based on the profile, an analysis template is selected
   (time series, categorical comparison, distribution, etc.) and the appropriate
   DuckDB analytics and Vega-Lite visualizations are generated.

5. **Evidence bundle** — Metrics, sample data, column metadata, and chart specs
   are packaged into an evidence bundle that serves as the LLM's grounded context.

6. **Narrative generation** — The evidence bundle is combined with audience-tuned
   system prompts, analysis-specific guidance, and guardrails (anti-hallucination,
   hedging, data interpretation rules). The LLM produces a structured JSON story
   with a headline, lede, story blocks, and data notes.

7. **Chart labeling** — Chart titles are generated deterministically in the
   analytics/evidence-bundle step (metric phrases, template type, and
   ``by year`` / ``by category`` patterns). Axis labels follow the CKAN portal
   language when available (e.g. Icelandic column names for Icelandic portals).

8. **Delivery** — The client renders the narrative text alongside interactive
   Vega-Lite charts.

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for
details. The MIT License is approved by the
[Open Source Initiative](https://opensource.org/licenses/MIT) and is compatible
with the [Digital Public Goods Standard](https://digitalpublicgoods.net/standard/).
