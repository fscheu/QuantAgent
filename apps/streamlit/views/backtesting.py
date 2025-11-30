
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import pandas as pd
import streamlit as st


def _get_universe_from_profile(db, profile_name: Optional[str]) -> List[str]:
    if not profile_name:
        return []
    if db.ok:
        try:
            with db.SessionLocal() as s:
                cfg = (
                    s.query(db.models.StrategyConfig)
                    .filter_by(name=profile_name, kind="portfolio")
                    .one_or_none()
                )
                if cfg and isinstance(cfg.json_config, dict):
                    universe = cfg.json_config.get("universe", [])
                    if isinstance(universe, list):
                        return [str(u) for u in universe]
        except Exception:
            pass
    profile = st.session_state.ui_profiles.get("portfolio", {}).get(profile_name)
    if profile and isinstance(profile, dict):
        universe = profile.get("universe", [])
        if isinstance(universe, list):
            return [str(u) for u in universe]
    return []


def render(db, environment: str) -> None:
    st.subheader("Backtesting – Create & Monitor Runs")
    st.caption("Create backtest runs; backend execution wiring is pending. Uses portfolio Universe when assets are not provided.")

    st.session_state.setdefault("backtest_runs", [])
    st.session_state.setdefault("ui_profiles", {"portfolio": {}, "risk": {}, "combined": {}})
    st.session_state.setdefault(
        "model_presets", {"default": {"provider": "openai", "model_name": "gpt-4o-mini", "temperature": 0.1}}
    )

    autorefresh_fn = getattr(st, "autorefresh", None)
    if callable(autorefresh_fn):
        autorefresh_fn(interval=5000, key="backtests_refresh")

    portfolio_options = _collect_portfolio_options(db)
    model_presets = list(st.session_state.model_presets.keys())

    with st.form("create_backtest"):
        assets_raw = st.text_input("Assets (comma separated; empty → use Universe)", value="")
        timeframe = st.text_input("Timeframe", value="1h")
        start_date = st.date_input("Start date")
        end_date = st.date_input("End date")
        model_preset = st.selectbox("Model preset", model_presets, index=model_presets.index("default"))
        profile_name = st.selectbox(
            "Portfolio profile (for sizing/universe)", [""] + portfolio_options, index=0
        )
        mode = st.selectbox("Mode", ["Generate analyses only", "Generate + Execute"], index=1)
        artifacts_policy = st.selectbox("Artifacts saving", ["none", "path-only", "path+thumbnail"], index=1)
        submitted = st.form_submit_button("Create run")

    if submitted:
        assets_list = [a.strip() for a in assets_raw.split(",") if a.strip()]
        if not assets_list:
            assets_list = _get_universe_from_profile(db, profile_name)

        if not assets_list:
            st.error("Provide assets or select a portfolio profile with a Universe.")
        else:
            snapshot = {
                "environment": environment,
                "profile": profile_name or None,
                "model_preset": model_preset,
                "mode": mode,
                "artifacts": artifacts_policy,
            }
            run_id = None
            if db.ok:
                try:
                    run = db.models.BacktestRun(
                        name=f"run-{datetime.utcnow().isoformat()}",
                        timeframe=timeframe,
                        assets=assets_list,
                        start_date=datetime.combine(start_date, datetime.min.time()),
                        end_date=datetime.combine(end_date, datetime.min.time()),
                        config_snapshot=snapshot,
                    )
                    with db.SessionLocal() as s:
                        s.add(run)
                        s.commit()
                        s.refresh(run)
                        run_id = run.id
                except Exception as e:
                    st.warning(f"DB save failed, keeping run only in session: {e}")

            if run_id is None:
                run_id = len(st.session_state.backtest_runs) + 1

            st.session_state.backtest_runs.append(
                {
                    "id": run_id,
                    "environment": environment,
                    "created_at": datetime.utcnow().isoformat(),
                    "status": "pending",
                    "progress": 0,
                    "assets": assets_list,
                    "timeframe": timeframe,
                    "range_start": str(start_date),
                    "range_end": str(end_date),
                    "model_preset": model_preset,
                    "profile": profile_name or None,
                    "mode": mode,
                    "artifacts": artifacts_policy,
                }
            )
            st.success(f"Run {run_id} created. Backend execution wiring pending.")

    st.markdown("**Runs**")
    rows = list(st.session_state.backtest_runs)
    if db.ok:
        try:
            with db.SessionLocal() as s:
                for r in s.query(db.models.BacktestRun).order_by(db.models.BacktestRun.created_at.desc()).limit(50):
                    rows.append(
                        {
                            "id": r.id,
                            "created_at": r.created_at,
                            "status": "completed" if r.total_trades is not None else "pending",
                            "progress": 100 if r.total_trades is not None else 0,
                            "assets": r.assets,
                            "timeframe": r.timeframe,
                            "range_start": r.start_date,
                            "range_end": r.end_date,
                            "model_preset": (r.config_snapshot or {}).get("model_preset"),
                            "profile": (r.config_snapshot or {}).get("profile"),
                            "mode": (r.config_snapshot or {}).get("mode"),
                            "artifacts": (r.config_snapshot or {}).get("artifacts"),
                            "win_rate": r.win_rate,
                            "profit_factor": r.profit_factor,
                            "sharpe_ratio": r.sharpe_ratio,
                            "max_drawdown": r.max_drawdown,
                            "total_pnl": r.total_pnl,
                        }
                    )
        except Exception as e:
            st.info(f"Could not load DB runs: {e}")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No runs yet.")


def _collect_portfolio_options(db) -> List[str]:
    db_names = _collect_portfolio_from_db(db)
    session_names = list(st.session_state.ui_profiles.get("portfolio", {}).keys())
    merged = list(dict.fromkeys(db_names + session_names))
    return merged


def _collect_portfolio_from_db(db) -> List[str]:
    if not db.ok:
        return []
    try:
        with db.SessionLocal() as s:
            return [
                c.name
                for c in s.query(db.models.StrategyConfig)
                .filter_by(kind="portfolio")
                .order_by(db.models.StrategyConfig.name)
                .all()
            ]
    except Exception:
        return []
