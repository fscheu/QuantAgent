from __future__ import annotations

import streamlit as st


def render() -> None:
    st.subheader("Replay – Scenario Sweeps (Sequential)")
    st.caption("Reuses stored analyses; executes different profiles sequentially.")

    run_id = st.number_input("Backtest run id", min_value=1, step=1)
    multi_profiles_raw = st.text_input("Profiles (comma)", value="moderate_equities,aggressive_all")
    if st.button("Start replay (sequential)"):
        profiles = [p.strip() for p in multi_profiles_raw.split(",") if p.strip()]
        st.success(
            f"Queued {len(profiles)} replay executions for run {int(run_id)} (placeholder). Backend to be wired."
        )

