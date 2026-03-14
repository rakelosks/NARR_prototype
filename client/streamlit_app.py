"""
Streamlit proof-of-concept client.
Consumes the backend API to provide a complete narrative visualization interface.

Run with: streamlit run client/streamlit_app.py
"""

import time
import json

import streamlit as st
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Smart City Data Narratives",
    page_icon="🏙️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def api_get(path: str, params: dict = None) -> dict | None:
    """GET request to the API."""
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("Cannot connect to API. Make sure it's running: `uvicorn app.main:app --reload`")
        return None
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")
        return None


def api_post(path: str, data: dict) -> dict | None:
    """POST request to the API."""
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("Cannot connect to API. Make sure it's running: `uvicorn app.main:app --reload`")
        return None
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")
        return None


def render_vega_lite(spec: dict):
    """Render a Vega-Lite spec in Streamlit."""
    st.vega_lite_chart(spec["data"]["values"], spec, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar — Data source management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏙️ Data sources")

    # API health
    health = api_get("/health")
    if health:
        st.success("API connected")
    else:
        st.stop()

    st.divider()

    # CKAN catalog refresh
    st.subheader("CKAN portal")
    portal_url = st.text_input(
        "Portal API URL",
        value="https://gagnagatt.reykjavik.is/en/api/3",
    )
    if st.button("Refresh catalog"):
        with st.spinner("Fetching catalog..."):
            result = api_post("/datasets/catalog/refresh", {
                "portal_url": portal_url,
                "full": False,
            })
            if result:
                st.success(f"Indexed {result['datasets_indexed']} datasets")

    st.divider()

    # Catalog search
    st.subheader("Search catalog")
    search_query = st.text_input("Search datasets", placeholder="e.g. transport, budget")
    if search_query:
        results = api_get("/datasets/catalog/search", {"q": search_query, "limit": 10})
        if results and results["results"]:
            for entry in results["results"]:
                with st.expander(entry["title"] or entry["name"]):
                    st.write(entry.get("description", "")[:200])
                    st.caption(f"Formats: {entry.get('resource_formats', [])}")
                    if st.button("Ingest", key=f"ingest_{entry['dataset_id']}"):
                        with st.spinner("Ingesting..."):
                            ingest_result = api_post("/datasets/ingest/ckan", {
                                "portal_url": portal_url,
                                "dataset_id": entry["name"],
                            })
                            if ingest_result:
                                st.success(
                                    f"Ingested: {ingest_result['name']} "
                                    f"({ingest_result['row_count']} rows)"
                                )
                                st.rerun()
        elif results:
            st.info("No results found")

    st.divider()

    # Direct URL ingestion
    st.subheader("Direct URL")
    with st.form("url_ingest"):
        url = st.text_input("Data file URL")
        name = st.text_input("Dataset name")
        fmt = st.selectbox("Format", ["auto", "csv", "json", "geojson", "xlsx"])
        submitted = st.form_submit_button("Ingest")
        if submitted and url and name:
            with st.spinner("Downloading..."):
                payload = {"url": url, "name": name}
                if fmt != "auto":
                    payload["format"] = fmt
                result = api_post("/datasets/ingest/url", payload)
                if result:
                    st.success(f"Ingested: {result['row_count']} rows")
                    st.rerun()

    st.divider()

    # List ingested datasets
    st.subheader("Ingested datasets")
    datasets_resp = api_get("/datasets/")
    datasets = datasets_resp["datasets"] if datasets_resp else []

    if not datasets:
        st.info("No datasets ingested yet")

    selected_dataset_id = None
    for ds in datasets:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{ds['name']}**")
            st.caption(f"{ds.get('row_count', '?')} rows")
        with col2:
            if st.button("Select", key=f"sel_{ds['id']}"):
                st.session_state["selected_dataset"] = ds["id"]
                st.session_state["selected_name"] = ds["name"]
                st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Smart City Data Narratives")

if "selected_dataset" not in st.session_state:
    st.markdown(
        """
        Welcome! This platform generates AI-powered narrative visualizations
        from city open data.

        **Get started:**
        1. Refresh the CKAN catalog in the sidebar
        2. Search for a dataset and ingest it
        3. Select a dataset to explore
        """
    )
    st.stop()


# Dataset is selected
dataset_id = st.session_state["selected_dataset"]
dataset_name = st.session_state.get("selected_name", dataset_id)

st.header(dataset_name)

# ---------------------------------------------------------------------------
# Tabs: Preview | Narrative | Raw
# ---------------------------------------------------------------------------

tab_preview, tab_narrative, tab_raw = st.tabs(["📊 Preview", "📝 Narrative", "🔧 Raw data"])

# --- Preview tab ---
with tab_preview:
    st.subheader("Automated analysis")
    if st.button("Generate preview", key="preview_btn"):
        with st.spinner("Profiling and analyzing..."):
            result = api_post("/narratives/preview", {"dataset_id": dataset_id})
            if result:
                st.session_state["preview"] = result

    if "preview" in st.session_state:
        preview = st.session_state["preview"]

        # Dataset info
        ds_info = preview["dataset"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", ds_info["row_count"])
        col2.metric("Columns", ds_info["column_count"])
        col3.metric("Template", ds_info["template_type"].replace("_", " ").title())

        st.caption(f"Column types: {ds_info['column_types']}")
        st.caption(f"Matched columns: {ds_info['matched_columns']}")

        # Visualizations
        for viz in preview["visualizations"]:
            st.subheader(viz["title"])
            st.caption(viz["description"])
            try:
                render_vega_lite(viz["vega_lite_spec"])
            except Exception as e:
                st.error(f"Chart rendering error: {e}")
                with st.expander("Raw spec"):
                    st.json(viz["vega_lite_spec"])

        # Key findings
        if preview.get("sections"):
            st.subheader("Key findings")
            for section in preview["sections"]:
                st.markdown(f"**{section.get('heading', '')}**")
                st.write(section.get("body", ""))


# --- Narrative tab ---
with tab_narrative:
    st.subheader("AI-generated narrative")
    user_message = st.text_area(
        "Ask a question about this dataset",
        placeholder="e.g. How has the budget changed over time? Compare districts by complaints.",
    )
    title_override = st.text_input("Custom title (optional)")

    col_sync, col_async = st.columns(2)

    # Synchronous generation
    with col_sync:
        if st.button("Generate (sync)", key="gen_sync"):
            with st.spinner("Generating narrative... (this may take a minute)"):
                payload = {"dataset_id": dataset_id}
                if user_message:
                    payload["user_message"] = user_message
                if title_override:
                    payload["title"] = title_override
                result = api_post("/narratives/generate", payload)
                if result:
                    st.session_state["narrative"] = result

    # Async generation
    with col_async:
        if st.button("Generate (async)", key="gen_async"):
            payload = {"dataset_id": dataset_id}
            if user_message:
                payload["user_message"] = user_message
            if title_override:
                payload["title"] = title_override
            job = api_post("/jobs/generate", payload)
            if job:
                st.session_state["active_job"] = job["job_id"]
                st.info(f"Job created: {job['job_id']}")

    # Poll active job
    if "active_job" in st.session_state:
        job_id = st.session_state["active_job"]
        status = api_get(f"/jobs/{job_id}")
        if status:
            if status["status"] == "completed":
                st.success("Generation complete!")
                st.session_state["narrative"] = status["result"]
                del st.session_state["active_job"]
            elif status["status"] == "failed":
                st.error(f"Job failed: {status.get('error')}")
                del st.session_state["active_job"]
            else:
                st.info(f"Job status: {status['status']}...")
                time.sleep(2)
                st.rerun()

    # Display narrative
    if "narrative" in st.session_state:
        pkg = st.session_state["narrative"]

        st.markdown(f"## {pkg.get('title', '')}")
        st.markdown(f"*{pkg.get('summary', '')}*")

        # Sections
        for section in pkg.get("sections", []):
            st.markdown(f"### {section.get('heading', '')}")
            st.write(section.get("body", ""))
            if "key_metric" in section and section["key_metric"]:
                km = section["key_metric"]
                st.metric(
                    label=km.get("label", ""),
                    value=km.get("value", ""),
                    help=km.get("context", ""),
                )

        # Visualizations
        for viz in pkg.get("visualizations", []):
            st.markdown(f"#### {viz['title']}")
            try:
                render_vega_lite(viz["vega_lite_spec"])
            except Exception as e:
                st.error(f"Chart rendering error: {e}")

        # Footer
        if pkg.get("data_limitations"):
            st.caption(f"⚠️ Limitations: {pkg['data_limitations']}")
        if pkg.get("suggested_followup"):
            st.info(f"💡 Follow-up: {pkg['suggested_followup']}")

        # Provenance
        with st.expander("Provenance"):
            prov = pkg.get("provenance", {})
            st.json(prov)


# --- Raw data tab ---
with tab_raw:
    st.subheader("Raw API responses")

    if st.button("Fetch visualization spec"):
        result = api_post("/visualizations/generate", {"dataset_id": dataset_id})
        if result:
            st.json(result)

    if "preview" in st.session_state:
        with st.expander("Preview package (JSON)"):
            st.json(st.session_state["preview"])

    if "narrative" in st.session_state:
        with st.expander("Narrative package (JSON)"):
            st.json(st.session_state["narrative"])