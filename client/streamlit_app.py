"""
Streamlit client — single-prompt data story interface.
A citizen types a question, the system finds the right dataset,
analyzes it, and returns an editorial data story with embedded charts.

Run with: streamlit run client/streamlit_app.py
"""

import copy
import html as html_lib
import json
import os
import time
import threading

import pandas as pd
import streamlit as st
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("NARR_API_KEY", "")

EXAMPLE_QUESTIONS = [
    "How has public transport usage changed over the years?",
    "What are the trends in waste collection?",
    "How has the city budget changed?",
    "What does school enrollment look like?",
]

PIPELINE_STEPS = [
    "Understanding your question",
    "Finding the right dataset",
    "Analyzing the data",
    "Generating visualizations",
    "Writing the story",
]

# Cumulative seconds at which each step roughly starts
STEP_THRESHOLDS = [0, 3, 7, 11, 14]


def _headers() -> dict:
    h = {}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


st.set_page_config(
    page_title="Smart City Data Stories",
    page_icon="🏙️",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Theme state
# ---------------------------------------------------------------------------

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

is_dark = st.session_state["theme"] == "dark"


# ---------------------------------------------------------------------------
# Theme palettes
# ---------------------------------------------------------------------------

if is_dark:
    T = dict(
        bg="#0e1117", bg_card="#1a1d24", text="#e8e7e3", text_muted="#9c9a92",
        text_body="#b0aaa5", border="rgba(255,255,255,0.08)",
        border_light="rgba(255,255,255,0.06)", accent="#2563eb",
        accent_hover="#1d4ed8",
        callout_bg="linear-gradient(135deg, #1e3a5f 0%, #1a2e4a 100%)",
        callout_border="#60a5fa", callout_value="#60a5fa",
        callout_label="#94a3b8", callout_body="#cbd5e1",
        note_bg="linear-gradient(135deg, #1a2332 0%, #1a1d24 100%)",
        note_accent="#f59e0b", timeline_dot="#2563eb",
        input_bg="#1a1d24", input_border="rgba(255,255,255,0.08)",
        spinner_track="rgba(37,99,235,0.3)", spinner_head="#2563eb",
        btn2_border="rgba(255,255,255,0.1)", btn2_text="#9c9a92",
        # Vega-Lite chart theme — match page background
        vega_bg="#0e1117", vega_text="#b0aaa5", vega_grid="#2a2d34",
        vega_title="#e8e7e3",
    )
else:
    T = dict(
        bg="#ffffff", bg_card="#f7f7f6", text="#1a1a1a", text_muted="#6b7280",
        text_body="#4b5563", border="rgba(0,0,0,0.08)",
        border_light="rgba(0,0,0,0.06)", accent="#2563eb",
        accent_hover="#1d4ed8",
        callout_bg="linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)",
        callout_border="#2563eb", callout_value="#1d4ed8",
        callout_label="#6b7280", callout_body="#374151",
        note_bg="linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%)",
        note_accent="#d97706", timeline_dot="#2563eb",
        input_bg="#f7f7f6", input_border="rgba(0,0,0,0.1)",
        spinner_track="rgba(37,99,235,0.2)", spinner_head="#2563eb",
        btn2_border="rgba(0,0,0,0.12)", btn2_text="#6b7280",
        vega_bg="#ffffff", vega_text="#4b5563", vega_grid="#e5e7eb",
        vega_title="#1a1a1a",
    )


# ---------------------------------------------------------------------------
# CSS — comprehensive overrides for both custom elements and Streamlit
# internals (expanders, dataframes, json, captions, inputs, etc.)
# ---------------------------------------------------------------------------

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
}}

#MainMenu {{visibility: hidden;}}
header {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* ============================================================
   GLOBAL NUCLEAR THEME OVERRIDE
   Force background on every Streamlit element, then restore
   specific children (buttons, charts) as needed.
   ============================================================ */
.stApp,
.stApp > div,
.stApp > div > div,
.stApp > div > div > div,
.stApp section,
.stApp section > div {{
    background-color: {T['bg']} !important;
    color: {T['text']} !important;
}}

/* All generic containers, blocks, and element wrappers */
.stApp [data-testid="stVerticalBlock"],
.stApp [data-testid="stHorizontalBlock"],
.stApp [data-testid="column"],
.stApp [data-testid="stVerticalBlockBorderWrapper"],
.stApp .block-container,
.stApp .element-container {{
    background-color: {T['bg']} !important;
}}

/* Markdown text */
.stApp .stMarkdown, .stApp .stMarkdown p,
.stApp .stMarkdown li, .stApp .stMarkdown span {{
    color: {T['text']} !important;
}}
.stApp .stCaption, .stApp .stCaption p {{
    color: {T['text_muted']} !important;
}}

/* ---- Text area ---- */
.stTextArea textarea {{
    background: {T['input_bg']} !important;
    border: 1px solid {T['input_border']} !important;
    border-radius: 12px !important;
    color: {T['text']} !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
    outline: none !important;
}}
.stTextArea textarea:focus {{
    border-color: {T['accent']} !important;
    box-shadow: 0 0 0 1px {T['accent']} !important;
}}
.stTextArea textarea::placeholder {{
    color: {T['text_muted']} !important;
}}
.stTextArea label {{ display: none !important; }}
/* Kill every wrapper div border/outline/shadow around the textarea */
.stTextArea, .stTextArea div,
[data-testid="stTextArea"], [data-testid="stTextArea"] div {{
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background-color: {T['bg']} !important;
}}
/* The textarea itself gets its own border back (above rules nuke divs) */
.stTextArea textarea {{
    border: 1px solid {T['input_border']} !important;
}}

/* ---- Buttons ---- */
.stButton > button[kind="primary"] {{
    background: {T['accent']} !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: {T['accent_hover']} !important;
}}
.stButton > button[kind="secondary"] {{
    background: transparent !important;
    border: 1px solid {T['btn2_border']} !important;
    border-radius: 8px !important;
    color: {T['btn2_text']} !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}}
.stButton > button[kind="secondary"]:hover {{
    border-color: {T['accent']} !important;
    color: {T['text']} !important;
}}
.stDownloadButton > button {{
    background: transparent !important;
    border: 1px solid {T['btn2_border']} !important;
    border-radius: 8px !important;
    color: {T['btn2_text']} !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}}
.stDownloadButton > button:hover {{
    border-color: {T['accent']} !important;
    color: {T['text']} !important;
}}

/* ---- Expanders: every layer ---- */
details[data-testid="stExpander"],
details[data-testid="stExpander"] *:not(button):not(svg):not(path):not(canvas):not(iframe) {{
    background-color: {T['bg']} !important;
}}
details[data-testid="stExpander"] summary {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: {T['text_muted']} !important;
    border-radius: 10px !important;
}}
details[data-testid="stExpander"] summary:hover,
details[data-testid="stExpander"][open] summary {{
    color: {T['text']} !important;
}}
details[data-testid="stExpander"] summary span {{
    color: inherit !important;
}}
details[data-testid="stExpander"] summary svg {{
    fill: {T['text_muted']} !important;
}}
details[data-testid="stExpander"] {{
    border: 1px solid {T['border']} !important;
    border-radius: 10px !important;
}}

/* ---- JSON viewer ---- */
/* Set bg on container only; do NOT touch child colors so syntax
   highlighting (keys=teal, strings=green, numbers=purple) stays */
[data-testid="stJson"] {{
    background-color: {T['bg_card']} !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
}}
/* Undo the nuclear color override for json children */
[data-testid="stJson"] span,
[data-testid="stJson"] div {{
    color: unset !important;
}}

/* ---- Dataframe ---- */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrame"] [data-testid="glideDataEditor"] {{
    background-color: {T['bg']} !important;
}}

/* ---- Alerts ---- */
.stAlert {{
    background-color: {T['bg_card']} !important;
    color: {T['text']} !important;
}}

/* ---- Custom story elements ---- */
.app-title {{
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: {T['text']};
}}
.app-subtitle {{
    font-size: 0.95rem;
    color: {T['text_muted']};
    margin-bottom: 1.5rem;
}}

.story-headline {{
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    line-height: 1.25;
    color: {T['text']};
    margin-bottom: 0.5rem;
}}
.story-lede {{
    font-size: 1.1rem;
    line-height: 1.65;
    color: {T['text_body']};
    margin-bottom: 1.5rem;
}}

.callout-block {{
    background: {T['callout_bg']};
    border-left: 4px solid {T['callout_border']};
    border-radius: 12px;
    padding: 1.75rem 2rem;
    margin: 1.5rem 0;
}}
.callout-value {{
    font-size: 2.75rem;
    font-weight: 700;
    color: {T['callout_value']};
    letter-spacing: -1px;
    line-height: 1.1;
}}
.callout-label {{
    font-size: 0.95rem;
    color: {T['callout_label']};
    margin-top: 0.25rem;
    margin-bottom: 0.5rem;
}}
.callout-body {{
    font-size: 0.95rem;
    color: {T['callout_body']};
    line-height: 1.5;
}}

.narrative-heading {{
    font-size: 1.25rem;
    font-weight: 600;
    letter-spacing: -0.3px;
    color: {T['text']};
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}}
.narrative-body {{
    font-size: 1rem;
    line-height: 1.7;
    color: {T['text_body']};
    margin-bottom: 1rem;
}}

.timeline-block {{
    border-left: 2px solid {T['timeline_dot']};
    padding-left: 1.25rem;
    margin: 1rem 0 1.5rem 0.5rem;
}}
.timeline-milestone {{
    position: relative;
    padding-bottom: 1rem;
}}
.timeline-milestone::before {{
    content: '';
    position: absolute;
    left: -1.55rem;
    top: 0.45rem;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: {T['timeline_dot']};
}}
.timeline-label {{
    font-weight: 600;
    font-size: 0.9rem;
    color: {T['text']};
}}
.timeline-desc {{
    font-size: 0.9rem;
    color: {T['text_muted']};
    margin-top: 0.15rem;
}}

.data-note {{
    background: {T['note_bg']};
    border-left: 3px solid {T['note_accent']};
    border-radius: 10px;
    padding: 1rem 1.25rem;
    font-size: 0.9rem;
    color: {T['text_body']};
    line-height: 1.6;
    margin: 1.5rem 0;
}}
.data-note-label {{
    font-weight: 600;
    color: {T['note_accent']};
    margin-bottom: 0.4rem;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.followup-label {{
    font-size: 0.8rem;
    font-weight: 600;
    color: {T['text_muted']};
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}}

.dataset-chip {{
    display: inline-block;
    background: {T['bg_card']};
    border: 1px solid {T['border']};
    border-radius: 8px;
    padding: 0.35rem 0.75rem;
    font-size: 0.8rem;
    color: {T['text_muted']};
    margin-bottom: 1rem;
}}

.soft-divider {{
    border: none;
    border-top: 1px solid {T['border_light']};
    margin: 1.5rem 0;
}}

.pipeline-status-line {{
    background: {T['bg_card']};
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 0.9rem;
    color: {T['text']};
}}
.pipeline-spinner {{
    width: 16px;
    height: 16px;
    border: 2px solid {T['spinner_track']};
    border-top-color: {T['spinner_head']};
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
}}
@keyframes spin {{
    to {{ transform: rotate(360deg); }}
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def api_get(path: str, params: dict = None) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, headers=_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("Cannot connect to the API. Make sure it's running.")
        return None
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code}")
        return None


def api_post(path: str, data: dict, timeout: int = 600) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, headers=_headers(), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("Cannot connect to the API. Make sure it's running.")
        return None
    except requests.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:
            detail = e.response.text
        st.error(f"Could not generate story: {detail}")
        return None
    except requests.ReadTimeout:
        st.error("The request timed out. Story generation can take a minute or two — please try again.")
        return None


def api_post_threaded(path: str, data: dict, timeout: int = 600, result_holder: dict = None):
    """Run API POST in a background thread, storing result in result_holder."""
    try:
        r = requests.post(
            f"{API_BASE}{path}", json=data, headers=_headers(), timeout=timeout
        )
        r.raise_for_status()
        result_holder["result"] = r.json()
    except requests.ConnectionError:
        result_holder["error"] = "Cannot connect to the API. Make sure it's running."
    except requests.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:
            detail = e.response.text
        result_holder["error"] = f"Could not generate story: {detail}"
    except requests.ReadTimeout:
        result_holder["error"] = (
            "The request timed out. Story generation can take a minute or two — "
            "please try again."
        )
    except Exception as e:
        result_holder["error"] = str(e)
    finally:
        result_holder["done"] = True


def theme_vega_spec(spec: dict) -> dict:
    """Inject theme-appropriate colors into a Vega-Lite spec."""
    spec = copy.deepcopy(spec)
    spec["background"] = T["vega_bg"]
    cfg = spec.setdefault("config", {})
    cfg["style"] = {"guide-label": {"fill": T["vega_text"]}, "guide-title": {"fill": T["vega_text"]}}
    cfg["axis"] = {
        "labelColor": T["vega_text"],
        "titleColor": T["vega_text"],
        "gridColor": T["vega_grid"],
        "domainColor": T["vega_grid"],
        "tickColor": T["vega_grid"],
    }
    cfg["legend"] = {"labelColor": T["vega_text"], "titleColor": T["vega_text"]}
    cfg["title"] = {"color": T["vega_title"]}
    cfg["view"] = {"stroke": "transparent"}
    return spec


def render_vega_lite(spec: dict):
    themed = theme_vega_spec(spec)
    st.vega_lite_chart(themed, use_container_width=True)


def ensure_catalog(portal_url: str):
    if "catalog_ready" in st.session_state:
        return True
    result = api_get("/datasets/catalog/search", {"limit": 1})
    if result and result.get("count", 0) > 0:
        st.session_state["catalog_ready"] = True
        return True
    with st.spinner("Connecting to open data portal..."):
        refresh = api_post("/datasets/catalog/refresh", {
            "portal_url": portal_url, "full": False,
        }, timeout=60)
        if refresh and refresh.get("datasets_indexed", 0) > 0:
            st.session_state["catalog_ready"] = True
            return True
        else:
            st.warning("Could not load the dataset catalog. Please try again.")
            return False


# ---------------------------------------------------------------------------
# Story download — self-contained HTML with charts
# ---------------------------------------------------------------------------

def build_story_html(pkg: dict) -> str:
    """Build a self-contained HTML file of the story for download."""
    headline = html_lib.escape(pkg.get("headline", ""))
    lede = html_lib.escape(pkg.get("lede", ""))
    dataset_name = html_lib.escape(pkg.get("dataset_name", ""))
    data_note = html_lib.escape(pkg.get("data_note", ""))
    vizs = pkg.get("visualizations", [])

    blocks_html = ""
    chart_specs = {}  # id → spec JSON

    for block in pkg.get("story_blocks", []):
        btype = block.get("type", "narrative")

        if btype == "callout":
            val = html_lib.escape(block.get("highlight_value", ""))
            lab = html_lib.escape(block.get("highlight_label", ""))
            bod = html_lib.escape(block.get("body", ""))
            blocks_html += f"""
            <div class="callout">
                <div class="callout-value">{val}</div>
                <div class="callout-label">{lab}</div>
                <div class="callout-body">{bod}</div>
            </div>"""

        elif btype == "timeline":
            heading = html_lib.escape(block.get("heading", ""))
            if heading:
                blocks_html += f"<h3>{heading}</h3>"
            blocks_html += '<div class="timeline">'
            for ms in block.get("milestones", []):
                ml = html_lib.escape(ms.get("label", ""))
                md = html_lib.escape(ms.get("description", ""))
                blocks_html += (
                    f'<div class="milestone"><strong>{ml}</strong> — {md}</div>'
                )
            blocks_html += "</div>"

        else:
            heading = html_lib.escape(block.get("heading", ""))
            body = html_lib.escape(block.get("body", ""))
            if heading:
                blocks_html += f"<h3>{heading}</h3>"
            if body:
                blocks_html += f"<p>{body}</p>"

            viz_idx = block.get("viz_index")
            if viz_idx is not None and 0 <= viz_idx < len(vizs):
                chart_id = f"chart-{viz_idx}"
                spec = vizs[viz_idx].get("vega_lite_spec", {})
                # Force light theme for downloaded HTML
                spec_copy = copy.deepcopy(spec)
                spec_copy["background"] = "#ffffff"
                spec_copy.setdefault("config", {})["view"] = {"stroke": "transparent"}
                chart_specs[chart_id] = json.dumps(spec_copy)
                blocks_html += f'<div id="{chart_id}" class="chart"></div>'

    note_html = ""
    if data_note:
        note_html = f"""
        <div class="data-note">
            <div class="data-note-label">About this data</div>
            <p>{data_note}</p>
        </div>"""

    # Build chart rendering script — runs after libs load
    chart_script_lines = []
    for chart_id, spec_json in chart_specs.items():
        chart_script_lines.append(
            f"  vegaEmbed('#{chart_id}', {spec_json}, {{actions: false}}).catch(console.error);"
        )
    chart_script = "\n".join(chart_script_lines)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{headline}</title>
<style>
  body {{
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    max-width: 720px; margin: 2rem auto; padding: 0 1.5rem;
    color: #1a1a1a; line-height: 1.7; background: #fff;
  }}
  h1 {{ font-size: 1.75rem; font-weight: 700; letter-spacing: -0.5px; line-height: 1.25; }}
  h3 {{ font-size: 1.15rem; font-weight: 600; margin-top: 1.5rem; }}
  .lede {{ font-size: 1.1rem; color: #4b5563; margin-bottom: 1.5rem; }}
  .dataset {{ font-size: 0.8rem; color: #6b7280; margin-bottom: 1rem; }}
  .callout {{
    background: #eff6ff; border-left: 4px solid #2563eb;
    border-radius: 10px; padding: 1.5rem 2rem; margin: 1.5rem 0;
  }}
  .callout-value {{ font-size: 2.5rem; font-weight: 700; color: #1d4ed8; }}
  .callout-label {{ font-size: 0.9rem; color: #6b7280; }}
  .callout-body {{ font-size: 0.95rem; color: #374151; margin-top: 0.5rem; }}
  .timeline {{ border-left: 2px solid #2563eb; padding-left: 1.25rem; margin: 1rem 0 1.5rem 0.5rem; }}
  .milestone {{ padding-bottom: 0.75rem; }}
  .chart {{ margin: 1.5rem 0; }}
  .data-note {{
    background: #fffbeb; border-left: 3px solid #d97706;
    border-radius: 10px; padding: 1rem 1.25rem; margin: 1.5rem 0;
    font-size: 0.9rem; color: #4b5563;
  }}
  .data-note-label {{
    font-weight: 600; color: #d97706; font-size: 0.8rem;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.25rem;
  }}
  .footer {{
    font-size: 0.75rem; color: #9ca3af; margin-top: 2rem;
    border-top: 1px solid #e5e7eb; padding-top: 1rem;
  }}
</style>
</head>
<body>
<div class="dataset">{dataset_name}</div>
<h1>{headline}</h1>
<div class="lede">{lede}</div>
{blocks_html}
{note_html}
<div class="footer">Generated by Smart City Data Stories (NARR)</div>

<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
<script>
{chart_script}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------

def render_callout(block: dict):
    value = block.get("highlight_value", "")
    label = block.get("highlight_label", "")
    body = block.get("body", "")
    st.markdown(f"""
    <div class="callout-block">
        <div class="callout-value">{value}</div>
        <div class="callout-label">{label}</div>
        <div class="callout-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)


def render_narrative(block: dict, vizs: list):
    heading = block.get("heading", "")
    body = block.get("body", "")
    if heading:
        st.markdown(
            f'<div class="narrative-heading">{heading}</div>',
            unsafe_allow_html=True,
        )
    if body:
        st.markdown(
            f'<div class="narrative-body">{body}</div>',
            unsafe_allow_html=True,
        )
    viz_idx = block.get("viz_index")
    if viz_idx is not None and 0 <= viz_idx < len(vizs):
        try:
            render_vega_lite(vizs[viz_idx]["vega_lite_spec"])
        except Exception as e:
            st.error(f"Chart rendering error: {e}")


def render_timeline(block: dict):
    heading = block.get("heading", "")
    milestones = block.get("milestones", [])
    if heading:
        st.markdown(
            f'<div class="narrative-heading">{heading}</div>',
            unsafe_allow_html=True,
        )
    if milestones:
        markup = '<div class="timeline-block">'
        for ms in milestones:
            label = ms.get("label", "")
            desc = ms.get("description", "")
            markup += f"""
            <div class="timeline-milestone">
                <div class="timeline-label">{label}</div>
                <div class="timeline-desc">{desc}</div>
            </div>"""
        markup += "</div>"
        st.markdown(markup, unsafe_allow_html=True)


def render_story(pkg: dict, story_index: int = 0):
    """Render the complete data story."""
    vizs = pkg.get("visualizations", [])
    suffix = f"_{story_index}"

    dataset_name = pkg.get("dataset_name", "")
    if dataset_name:
        st.markdown(
            f'<div class="dataset-chip">Dataset: {dataset_name}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    headline = pkg.get("headline", "")
    if headline:
        st.markdown(
            f'<div class="story-headline">{headline}</div>',
            unsafe_allow_html=True,
        )

    lede = pkg.get("lede", "")
    if lede:
        st.markdown(
            f'<div class="story-lede">{lede}</div>',
            unsafe_allow_html=True,
        )

    for block in pkg.get("story_blocks", []):
        block_type = block.get("type", "narrative")
        if block_type == "callout":
            render_callout(block)
        elif block_type == "timeline":
            render_timeline(block)
        else:
            render_narrative(block, vizs)

    data_note = pkg.get("data_note", "")
    if data_note:
        st.markdown(f"""
        <div class="data-note">
            <div class="data-note-label">About this data</div>
            {data_note}
        </div>
        """, unsafe_allow_html=True)

    followup = pkg.get("followup_question", "")
    if followup:
        st.markdown(
            '<div class="followup-label">Explore next</div>',
            unsafe_allow_html=True,
        )
        if st.button(f"{followup}", key=f"followup{suffix}", type="secondary"):
            st.session_state["pending_question"] = followup
            st.rerun()

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # Download button
    story_html = build_story_html(pkg)
    safe_headline = (headline or "story").replace(" ", "_")[:40]
    st.download_button(
        label="Download story as HTML",
        data=story_html,
        file_name=f"{safe_headline}.html",
        mime="text/html",
        key=f"download{suffix}",
    )

    # Expandable details
    with st.expander("View raw data"):
        if vizs:
            raw_data = (
                vizs[0].get("vega_lite_spec", {}).get("data", {}).get("values", [])
            )
            if raw_data:
                st.dataframe(pd.DataFrame(raw_data), use_container_width=True)
            else:
                st.info("No tabular data available.")
        else:
            st.info("No data available.")

    with st.expander("Provenance"):
        prov = pkg.get("provenance", {})
        if prov:
            cols = st.columns(3)
            cols[0].markdown(f"**Template:** {prov.get('template_type', 'N/A')}")
            cols[1].markdown(f"**Model:** {prov.get('llm_model', 'N/A')}")
            cols[2].markdown(f"**Attempts:** {prov.get('generation_attempts', 'N/A')}")
            st.json(prov)

    with st.expander("Full API response"):
        st.json(pkg)


def run_generation(question: str, portal_url: str):
    """
    Run story generation with a live-updating progress indicator.
    API call runs in a background thread; main thread polls and updates UI.
    """
    if not ensure_catalog(portal_url):
        return

    status_container = st.empty()

    holder = {"result": None, "error": None, "done": False}
    thread = threading.Thread(
        target=api_post_threaded,
        args=("/narratives/ask", {"user_message": question}),
        kwargs={"timeout": 600, "result_holder": holder},
        daemon=True,
    )

    start_time = time.time()
    thread.start()

    last_step = -1
    while not holder["done"]:
        elapsed = time.time() - start_time
        current_step = 0
        for i, threshold in enumerate(STEP_THRESHOLDS):
            if elapsed >= threshold:
                current_step = i

        if current_step != last_step:
            step_text = PIPELINE_STEPS[current_step]
            status_container.markdown(
                f'<div class="pipeline-status-line">'
                f'<div class="pipeline-spinner"></div>{step_text}...'
                f"</div>",
                unsafe_allow_html=True,
            )
            last_step = current_step

        time.sleep(0.3)

    status_container.empty()
    elapsed = time.time() - start_time

    if holder["error"]:
        st.error(holder["error"])
        return

    result = holder["result"]
    if result:
        result["_generation_time"] = round(elapsed, 1)
        result["_question"] = question
        if "stories" not in st.session_state:
            st.session_state["stories"] = []
        st.session_state["stories"].insert(0, result)
        st.rerun()


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

# Header row with theme toggle
header_col, toggle_col = st.columns([5, 1])
with header_col:
    st.markdown(
        '<div class="app-title">Smart City Data Stories</div>',
        unsafe_allow_html=True,
    )
with toggle_col:
    st.markdown("")
    theme_icon = "☀️" if is_dark else "🌙"
    theme_label = "Light" if is_dark else "Dark"
    if st.button(
        f"{theme_icon} {theme_label}", key="theme_toggle", type="secondary"
    ):
        st.session_state["theme"] = "light" if is_dark else "dark"
        st.rerun()

st.markdown(
    '<div class="app-subtitle">'
    "Ask a question about your city and get an AI-generated data story "
    "with visualizations"
    "</div>",
    unsafe_allow_html=True,
)

# Check API health
health = api_get("/health")
if not health:
    st.stop()

portal_url = health.get("portal_url", "")

# ---------------------------------------------------------------------------
# Question input — always visible
# ---------------------------------------------------------------------------

pending = st.session_state.pop("pending_question", None)

if pending:
    st.text_area(
        "What would you like to know?",
        value=pending,
        height=80,
        label_visibility="collapsed",
        disabled=True,
    )
    run_generation(pending, portal_url)
else:
    user_message = st.text_area(
        "What would you like to know?",
        placeholder="e.g. How has waste collection changed over the years?",
        height=80,
        label_visibility="collapsed",
    )

    generate_clicked = st.button(
        "Generate story", type="primary", use_container_width=True
    )

    # Example questions — only show when no stories yet
    if "stories" not in st.session_state or len(st.session_state["stories"]) == 0:
        st.markdown("")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            col = cols[i % 2]
            with col:
                if st.button(
                    q, key=f"example_{i}", type="secondary", use_container_width=True
                ):
                    st.session_state["pending_question"] = q
                    st.rerun()

    if generate_clicked and user_message.strip():
        run_generation(user_message.strip(), portal_url)
    elif generate_clicked:
        st.warning("Please type a question first.")

# ---------------------------------------------------------------------------
# Display stories (newest first, all preserved)
# ---------------------------------------------------------------------------

stories = st.session_state.get("stories", [])

for i, pkg in enumerate(stories):
    question = pkg.get("_question", "")
    gen_time = pkg.get("_generation_time", "")
    if question:
        time_label = f" ({gen_time}s)" if gen_time else ""
        st.caption(f"Q: {question}{time_label}")

    render_story(pkg, story_index=i)

    if i < len(stories) - 1:
        st.markdown("")
        st.markdown("")
