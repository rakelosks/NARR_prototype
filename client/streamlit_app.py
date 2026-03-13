"""
Streamlit proof-of-concept client.
Consumes the backend API endpoints to display narrative visualizations.
This is a lightweight sample interface — the core system is frontend-agnostic.
"""

import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Smart City Data Narratives",
    page_icon="🏙️",
    layout="wide",
)

st.title("Smart City Narrative Visualization")
st.markdown("AI-powered open data narratives for smart cities.")


# Health check
try:
    health = requests.get(f"{API_BASE_URL}/health", timeout=5)
    if health.status_code == 200:
        st.success("Backend API is running")
    else:
        st.error("Backend API returned an error")
except requests.ConnectionError:
    st.warning("Backend API is not running. Start it with: `uvicorn app.main:app --reload`")


# Placeholder sections
st.header("Datasets")
st.info("No datasets loaded yet. Use the API to ingest a dataset.")

st.header("Narrative Visualizations")
st.info("Generate narratives after loading a dataset.")
