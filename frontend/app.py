"""Streamlit frontend for the Agentic AI Workshop pipeline."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure we can import the backend modules when launching from the frontend directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Load environment variables - try .env file first (local), then Streamlit secrets (cloud)
load_dotenv()

# Load Streamlit secrets into environment variables
if hasattr(st, 'secrets'):
    for key, value in st.secrets.items():
        if key not in os.environ:
            os.environ[key] = str(value)

from main import run_pipeline  # noqa: E402  # pylint: disable=wrong-import-position

st.set_page_config(page_title="Agentic AI Workshop", page_icon="🧠", layout="wide")

st.title("Agentic AI Workshop: Multi-Agent Systems From Idea to Deployment")
st.write(
    "Use this interface to run the full CrewAI pipeline. Enter a workshop theme and the agents will plan, "
    "research, write, and review a set of deliverables."
)

default_topic = "Designing a Multi-Agent Workshop for Cloud-Native Deployments"

with st.sidebar:
    st.header("Configuration")
    topic = st.text_input("Workshop Topic", value=default_topic)
    run_button = st.button("Run Pipeline", type="primary")

if run_button:
    st.info("Starting the agentic pipeline. This may take a few minutes depending on model latency.")
    with st.spinner("Agents working..."):
        try:
            output = run_pipeline(topic)
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
        else:
            st.success("Pipeline completed successfully!")
            st.markdown("### Final Output")
            st.markdown(output)

st.markdown("---")
st.caption(
    "Tip: Update agent prompts, tasks, or the RAG knowledge base to explore different workshop outcomes."
)