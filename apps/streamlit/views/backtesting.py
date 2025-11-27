from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st


def render() -> None:
    st.subheader("Backtesting – Create & Monitor Runs")

    # Ensure session key exists
    st.session_state.setdefault("backtest_runs", [])

    with st.form("create_backtest"):
        assets = st.text_input("Assets (comma)", value="BTC,SPX")
        timeframe = st.text_input("Timeframe", value="1h")
        start_date = st.date_input("Start date")
        end_date = st.date_input("End date")
        model_preset = st.selectbox("Model preset", ["default"])  # placeholder until DB exists
        profile_name = st.text_input("Profile name for initial exec (optional)")
        mode = st.selectbox("Mode", ["Generate analyses only", "Generate + Execute"], index=1)
        artifacts_policy = st.selectbox("Artifacts saving", ["none", "path-only", "path+thumbnail"], index=1)
        submitted = st.form_submit_button("Create run")

    if submitted:
        run_id = len(st.session_state.backtest_runs) + 1
        st.session_state.backtest_runs.append(
            {
                "id": run_id,
                "created_at": datetime.utcnow().isoformat(),
                "status": "pending",
                "progress": 0,
                "assets": [a.strip() for a in assets.split(",") if a.strip()],
                "timeframe": timeframe,
                "range": {"start": str(start_date), "end": str(end_date)},
                "model_preset": model_preset,
                "profile": profile_name or None,
                "mode": mode,
                "artifacts": artifacts_policy,
            }
        )
        st.success(f"Run {run_id} created (placeholder). Backend execution to be wired.")

    st.markdown("**Runs (session placeholder)**")
    st.dataframe(pd.DataFrame(st.session_state.backtest_runs), use_container_width=True)

