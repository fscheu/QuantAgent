"""
Streamlit MVP UI for QuantAgent

Focus: functionality over aesthetics. Reads from database if available.
Tabs: Dashboard, Configuration, Analyses, Backtesting, Replay, Orders & Positions, Logs.

Run: streamlit run apps/streamlit/app.py
"""

from __future__ import annotations

import json

import streamlit as st

from apps.streamlit.services.db import get_db_handle
from apps.streamlit.views.dashboard import render as render_dashboard
from apps.streamlit.views.configuration import render as render_configuration
from apps.streamlit.views.analyses import render as render_analyses
from apps.streamlit.views.backtesting import render as render_backtesting
from apps.streamlit.views.replay import render as render_replay
from apps.streamlit.views.orders_positions import render as render_orders_positions
from apps.streamlit.views.logs import render as render_logs


# -----------------------------
# UI State & Defaults
# -----------------------------


ENVIRONMENTS = ["backtest", "paper"]  # prod out of MVP scope for UI

st.set_page_config(page_title="QuantAgent UI (MVP)", layout="wide")
st.title("QuantAgent – Streamlit MVP")

# Initialize session-scoped defaults
if "ui_profiles" not in st.session_state:
    st.session_state.ui_profiles = {
        "portfolio": {},  # name -> json
        "risk": {},
        "combined": {},
    }

if "model_presets" not in st.session_state:
    st.session_state.model_presets = {
        "default": {"provider": "openai", "model_name": "gpt-4o-mini", "temperature": 0.1}
    }

if "backtest_runs" not in st.session_state:
    st.session_state.backtest_runs = []  # ephemeral placeholder until DB tables exist


col0, col1 = st.columns([1, 3])
with col0:
    environment = st.selectbox("Environment", ENVIRONMENTS, index=1)
with col1:
    st.caption("Set DATABASE_URL and start PostgreSQL via docker-compose for full functionality.")

db = get_db_handle()
if not db.ok:
    st.warning(db.error)


tabs = st.tabs(
    [
        "Dashboard",
        "Configuration",
        "Analyses",
        "Backtesting",
        "Replay",
        "Orders & Positions",
        "Logs",
    ]
)

with tabs[0]:
    render_dashboard(db, environment)

with tabs[1]:
    render_configuration()

with tabs[2]:
    render_analyses(db, environment)

with tabs[3]:
    render_backtesting()

with tabs[4]:
    render_replay()

with tabs[5]:
    render_orders_positions(db, environment)

with tabs[6]:
    render_logs()
