"""
Streamlit MVP UI for QuantAgent

Focus: functionality over aesthetics. Reads from database if available.
Views: Dashboard, Configuration, Analyses, Backtesting, Replay, Orders & Positions, Logs, User Manual.

Run: streamlit run apps/streamlit/app.py
"""

from __future__ import annotations

import streamlit as st

from apps.streamlit.services.db import get_db_handle
from apps.streamlit.views.analyses import render as render_analyses
from apps.streamlit.views.backtesting import render as render_backtesting
from apps.streamlit.views.configuration import render as render_configuration
from apps.streamlit.views.dashboard import render as render_dashboard
from apps.streamlit.views.logs import render as render_logs
from apps.streamlit.views.manual import render as render_manual
from apps.streamlit.views.orders_positions import render as render_orders_positions
from apps.streamlit.views.paper_trading import render as render_paper_trading
from apps.streamlit.views.replay import render as render_replay
from quantagent.logging_config import setup_logging

# Initialize logging for Streamlit (DB only, no console clutter)
setup_logging(log_to_console=False, log_to_db=True)

# -----------------------------
# UI State & Defaults
# -----------------------------


ENVIRONMENTS = ["backtest", "paper"]  # prod out of MVP scope for UI
NAVIGATION_VIEWS = [
    "Dashboard",
    "Paper Trading",
    "Configuration",
    "Analyses",
    "Backtesting",
    "Replay",
    "Orders & Positions",
    "Logs",
    "User Manual",
]


def _get_current_view() -> str:
    current_view = st.query_params.get("view", NAVIGATION_VIEWS[0])
    if current_view not in NAVIGATION_VIEWS:
        return NAVIGATION_VIEWS[0]
    return current_view

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
        "default": {
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "temperature": 0.1,
        }
    }

if "backtest_runs" not in st.session_state:
    st.session_state.backtest_runs = []  # ephemeral placeholder until DB tables exist


col0, col1 = st.columns([1, 3])
with col0:
    environment = st.selectbox("Environment", ENVIRONMENTS, index=1)

db = get_db_handle()

with col1:
    if db.ok:
        st.caption("PostgreSQL connected.")
    else:
        st.caption(
            "Set DATABASE_URL and start PostgreSQL via docker-compose for full functionality."
        )
        st.warning(db.error)

selected_view = st.radio(
    "Navigation",
    NAVIGATION_VIEWS,
    index=NAVIGATION_VIEWS.index(_get_current_view()),
    horizontal=True,
    label_visibility="collapsed",
)
if st.query_params.get("view") != selected_view:
    st.query_params["view"] = selected_view

if selected_view == "Dashboard":
    render_dashboard(db, environment)
elif selected_view == "Paper Trading":
    render_paper_trading(db, environment)
elif selected_view == "Configuration":
    render_configuration(db, environment)
elif selected_view == "Analyses":
    render_analyses(db, environment)
elif selected_view == "Backtesting":
    render_backtesting(db, environment)
elif selected_view == "Replay":
    render_replay(db, environment)
elif selected_view == "Orders & Positions":
    render_orders_positions(db, environment)
elif selected_view == "Logs":
    render_logs(db)
else:
    render_manual()
