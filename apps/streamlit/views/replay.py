from __future__ import annotations

import pandas as pd
import streamlit as st


def render(db, environment: str) -> None:
    st.subheader("Replay – Scenario Sweeps (Sequential)")
    st.caption("Reuses stored analyses; executes different profiles sequentially.")

    st.session_state.setdefault("replay_runs", [])
    st.session_state.setdefault("backtest_runs", [])
    st.session_state.setdefault("ui_profiles", {"portfolio": {}, "risk": {}, "combined": {}})

    available_runs = {r["id"] for r in st.session_state.backtest_runs if "id" in r}
    if db.ok:
        try:
            with db.SessionLocal() as s:
                for r in s.query(db.models.BacktestRun.id).all():
                    run_id = getattr(r, "id", None)
                    if run_id is None and isinstance(r, tuple) and r:
                        run_id = r[0]
                    if run_id is not None:
                        available_runs.add(run_id)
        except Exception:
            pass
    runs_list = sorted(list(available_runs))

    if not runs_list:
        st.info("Create a backtest run first.")
        return

    run_id = st.selectbox("Backtest run id", runs_list, index=0)
    profile_options = _collect_profile_options(db)
    profiles_selected = st.multiselect(
        "Profiles to replay sequentially", profile_options, default=profile_options[:1] if profile_options else []
    )
    if st.button("Start replay (sequential)"):
        queue = profiles_selected or ["default"]
        for profile in queue:
            st.session_state.replay_runs.append(
                {
                    "backtest_run_id": run_id,
                    "profile": profile,
                    "status": "queued",
                    "progress": 0,
                    "environment": environment,
                }
            )
        st.success(f"Queued {len(queue)} replay executions for run {run_id}. Backend wiring pending.")

    if st.session_state.replay_runs:
        st.markdown("**Replay queue (session)**")
        st.dataframe(pd.DataFrame(st.session_state.replay_runs), use_container_width=True)


def _collect_profile_options(db):
    names = []
    if db.ok:
        try:
            with db.SessionLocal() as s:
                names = [c.name for c in s.query(db.models.StrategyConfig).order_by(db.models.StrategyConfig.name)]
        except Exception:
            pass
    names.extend(list(st.session_state.ui_profiles.get("portfolio", {}).keys()))
    names.extend(list(st.session_state.ui_profiles.get("risk", {}).keys()))
    names.extend(list(st.session_state.ui_profiles.get("combined", {}).keys()))
    return sorted(list(dict.fromkeys(names)))

