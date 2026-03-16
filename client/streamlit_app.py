"""
Streamlit client — single-prompt data story interface.
A citizen types a question, the system finds the right dataset,
analyzes it, and returns an editorial data story with embedded charts.

Run with: streamlit run client/streamlit_app.py
"""

import pandas as pd
import streamlit as st
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Smart City Data Stories",
    page_icon="🏙️",
    layout="centered",
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


def api_post(path: str, data: dict, timeout: int = 600) -> dict | None:
    """POST request to the API."""
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("Cannot connect to API. Make sure it's running: `uvicorn app.main:app --reload`")
        return None
    except requests.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:
            detail = e.response.text
        st.error(f"Error: {detail}")
        return None
    except requests.ReadTimeout:
        st.error("The request timed out. The story generation may take a few minutes — please try again.")
        return None


def render_vega_lite(spec: dict):
    """Render a Vega-Lite spec in Streamlit."""
    st.vega_lite_chart(spec["data"]["values"], spec, use_container_width=True)


def ensure_catalog(portal_url: str):
    """Ensure the catalog is populated. Auto-refresh if empty."""
    if "catalog_ready" in st.session_state:
        return True

    # Check if catalog has entries
    result = api_get("/datasets/catalog/search", {"limit": 1})
    if result and result.get("count", 0) > 0:
        st.session_state["catalog_ready"] = True
        return True

    # Auto-refresh catalog from the configured portal
    with st.spinner("Connecting to open data portal..."):
        refresh = api_post("/datasets/catalog/refresh", {
            "portal_url": portal_url,
            "full": False,
        }, timeout=60)
        if refresh and refresh.get("datasets_indexed", 0) > 0:
            st.session_state["catalog_ready"] = True
            return True
        else:
            st.warning("Could not load dataset catalog. Please try again later.")
            return False


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("🏙️ Smart City Data Stories")
st.caption("Ask a question and get an AI-generated data story with visualizations")

# Check API health and get portal URL
health = api_get("/health")
if not health:
    st.stop()

portal_url = health.get("portal_url", "")


# ---------------------------------------------------------------------------
# Prompt input
# ---------------------------------------------------------------------------

user_message = st.text_area(
    "What would you like to know?",
    placeholder="e.g. How has waste collection changed over the years? What are the budget trends?",
    height=80,
    label_visibility="collapsed",
)

generate_clicked = st.button("Generate story", type="primary", use_container_width=True)

if generate_clicked and user_message.strip():
    # Ensure catalog is available
    if not ensure_catalog(portal_url):
        st.stop()

    with st.spinner("Finding the right dataset and generating your story..."):
        result = api_post("/narratives/ask", {
            "user_message": user_message.strip(),
        }, timeout=600)

        if result:
            st.session_state["story"] = result
            st.rerun()

elif generate_clicked and not user_message.strip():
    st.warning("Please type a question first.")


# ---------------------------------------------------------------------------
# Story display
# ---------------------------------------------------------------------------

if "story" in st.session_state:
    pkg = st.session_state["story"]
    vizs = pkg.get("visualizations", [])

    # Dataset info caption
    dataset_name = pkg.get("dataset_name", "")
    if dataset_name:
        st.caption(f"📊 Dataset: {dataset_name}")

    st.divider()

    # Headline + lede
    headline = pkg.get("headline", "")
    lede = pkg.get("lede", "")

    if headline:
        st.markdown(f"## {headline}")
    if lede:
        st.markdown(lede)

    # Story blocks — editorial flow with inline charts
    for block in pkg.get("story_blocks", []):
        block_type = block.get("type", "narrative")

        if block_type == "narrative":
            if block.get("heading"):
                st.markdown(f"### {block['heading']}")
            if block.get("body"):
                st.write(block["body"])
            # Render chart inline if viz_index is set
            viz_idx = block.get("viz_index")
            if viz_idx is not None and 0 <= viz_idx < len(vizs):
                try:
                    render_vega_lite(vizs[viz_idx]["vega_lite_spec"])
                except Exception as e:
                    st.error(f"Chart rendering error: {e}")

        elif block_type == "timeline":
            if block.get("heading"):
                st.markdown(f"### {block['heading']}")
            for ms in block.get("milestones", []):
                st.markdown(f"**{ms.get('label', '')}** — {ms.get('description', '')}")

        elif block_type == "callout":
            col_metric, col_text = st.columns([1, 3])
            with col_metric:
                st.metric(
                    label=block.get("highlight_label", ""),
                    value=block.get("highlight_value", ""),
                )
            with col_text:
                if block.get("body"):
                    st.write(block["body"])

    # Footer
    if pkg.get("data_note"):
        st.caption(f"⚠️ {pkg['data_note']}")
    if pkg.get("followup_question"):
        st.info(f"💡 {pkg['followup_question']}")

    st.divider()

    # Raw data — extract from primary visualization spec
    with st.expander("View raw data"):
        if vizs:
            raw_data = vizs[0].get("vega_lite_spec", {}).get("data", {}).get("values", [])
            if raw_data:
                st.dataframe(pd.DataFrame(raw_data), use_container_width=True)
            else:
                st.info("No tabular data available.")
        else:
            st.info("No data available.")

    # Provenance
    with st.expander("Provenance"):
        prov = pkg.get("provenance", {})
        st.json(prov)

    # Full JSON response
    with st.expander("Full API response"):
        st.json(pkg)

    # New question button
    if st.button("Ask another question"):
        del st.session_state["story"]
        st.rerun()
