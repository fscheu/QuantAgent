from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st


def render(db, environment: str) -> None:
    st.subheader("Replay – Scenario Sweeps (Sequential)")
    st.caption("Reuses stored analyses; executes different profiles sequentially.")

    st.session_state.setdefault("replay_results", [])

    runs = _load_completed_runs(db)
    if not runs:
        st.info("No completed backtest runs found. Run a backtest first.")
        return

    run_labels = {f"#{r['id']} — {r['name']} ({r['total_trades']} trades)": r["id"] for r in runs}
    selected_label = st.selectbox("Source backtest run", list(run_labels.keys()))
    source_run_id = run_labels[selected_label]

    profile_options = _collect_profile_options(db)
    profiles_selected = st.multiselect(
        "Profiles to replay sequentially",
        profile_options,
        default=profile_options[:1] if profile_options else [],
        help="Leave empty to replay with the source run's own config.",
    )
    if not profiles_selected:
        profiles_selected = ["(source config)"]

    if st.button("Start replay (sequential)"):
        st.session_state.replay_results = []
        progress_bar = st.progress(0, text="Starting replay…")

        for idx, profile in enumerate(profiles_selected):
            progress_bar.progress(
                idx / len(profiles_selected),
                text=f"Replaying profile '{profile}'…",
            )
            result = _execute_replay(db, source_run_id, profile, environment)
            st.session_state.replay_results.append(result)

        progress_bar.progress(1.0, text="Replay complete.")

    if st.session_state.replay_results:
        _render_comparison(st.session_state.replay_results)


def _execute_replay(db, source_run_id: int, profile: str, environment: str) -> dict:
    """Run a single replay and return a result dict."""
    started_at = datetime.utcnow()
    try:
        target_config = _resolve_profile_config(db, profile)

        from quantagent.backtesting.backtest import Backtest
        from quantagent.database import SessionLocal
        from quantagent.models import BacktestRun

        with SessionLocal() as session:
            source_run = session.query(BacktestRun).filter(BacktestRun.id == source_run_id).first()
            if source_run is None:
                return _error_result(profile, f"Source run {source_run_id} not found")

            initial_capital = (
                target_config.get("initial_cash")
                or source_run.config_snapshot.get("initial_cash", 100_000.0)
            )
            merged_config = dict(source_run.config_snapshot or {})
            merged_config.update(target_config)

            backtest = Backtest(
                start_date=source_run.start_date,
                end_date=source_run.end_date,
                assets=source_run.assets,
                timeframe=source_run.timeframe,
                initial_capital=float(initial_capital),
                config=merged_config,
                db_session=session,
            )
            metrics = backtest.run_replay(
                source_run_id=source_run_id,
                name=f"Replay #{source_run_id} — {profile}",
            )

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        return {
            "profile": profile,
            "status": "ok",
            "replay_run_id": backtest.backtest_run_id,
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "total_pnl": metrics.total_pnl,
            "total_return_pct": metrics.total_return_pct,
            "elapsed_s": round(elapsed, 1),
        }

    except Exception as exc:
        return _error_result(profile, str(exc))


def _error_result(profile: str, error: str) -> dict:
    return {"profile": profile, "status": "error", "error": error}


def _resolve_profile_config(db, profile_name: str) -> dict:
    """Load profile config from DB or return empty dict for source config."""
    if profile_name == "(source config)":
        return {}
    if not db.ok:
        return {}
    try:
        with db.SessionLocal() as s:
            cfg = (
                s.query(db.models.StrategyConfig)
                .filter(db.models.StrategyConfig.name == profile_name)
                .first()
            )
            if cfg:
                return cfg.json_config or {}
    except Exception:
        pass
    return {}


def _render_comparison(results: list) -> None:
    st.markdown("### Replay results")

    ok_results = [r for r in results if r.get("status") == "ok"]
    err_results = [r for r in results if r.get("status") == "error"]

    if ok_results:
        df = pd.DataFrame(
            [
                {
                    "Profile": r["profile"],
                    "Run ID": r.get("replay_run_id", "—"),
                    "Trades": r["total_trades"],
                    "Win Rate": f"{r['win_rate']:.1%}",
                    "Profit Factor": f"{r['profit_factor']:.2f}",
                    "Sharpe": f"{r['sharpe_ratio']:.2f}",
                    "Max DD": f"{r['max_drawdown']:.1%}",
                    "Total P&L": f"${r['total_pnl']:,.2f}",
                    "Return": f"{r['total_return_pct']:.2f}%",
                    "Elapsed (s)": r.get("elapsed_s", "—"),
                }
                for r in ok_results
            ]
        )
        st.dataframe(df, use_container_width=True)

    for r in err_results:
        st.error(f"Profile '{r['profile']}': {r.get('error', 'unknown error')}")


def _load_completed_runs(db) -> list:
    """Return list of recent BacktestRun dicts available for replay."""
    if not db.ok:
        return []
    try:
        with db.SessionLocal() as s:
            runs = (
                s.query(db.models.BacktestRun)
                .order_by(db.models.BacktestRun.created_at.desc())
                .limit(50)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "name": r.name or f"Run #{r.id}",
                    "total_trades": r.total_trades or 0,
                }
                for r in runs
            ]
    except Exception:
        return []


def _collect_profile_options(db) -> list:
    names = []
    if db.ok:
        try:
            with db.SessionLocal() as s:
                names = [
                    c.name
                    for c in s.query(db.models.StrategyConfig).order_by(
                        db.models.StrategyConfig.name
                    )
                ]
        except Exception:
            pass
    names.extend(list(st.session_state.get("ui_profiles", {}).get("portfolio", {}).keys()))
    names.extend(list(st.session_state.get("ui_profiles", {}).get("risk", {}).keys()))
    names.extend(list(st.session_state.get("ui_profiles", {}).get("combined", {}).keys()))
    return sorted(list(dict.fromkeys(names)))
