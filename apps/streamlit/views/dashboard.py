from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import streamlit as st


def _calculate_status(heartbeat: Optional[dict]) -> tuple[str, str]:
    """
    Calculate scheduler status based on last heartbeat.
    Uses same logic as paper_trading.py for consistency.
    """
    if not heartbeat:
        return ("🔴", "Stopped")

    last_run = heartbeat.get("timestamp")
    if not last_run:
        return ("🔴", "Stopped")

    # Calculate time since last run
    now = datetime.utcnow()
    if isinstance(last_run, str):
        last_run = datetime.fromisoformat(last_run.replace("Z", "+00:00"))

    time_since = now - last_run

    heartbeat_status = heartbeat.get("status")
    if heartbeat_status == "error":
        return ("❌", "Error")
    if heartbeat_status == "running":
        if time_since < timedelta(hours=2):
            return ("⏳", "Running")
        return ("🟠", "Stuck")

    # Default fallback: Active/Stale/Stopped
    if time_since < timedelta(hours=2):
        return ("🟢", "Active")
    elif time_since < timedelta(hours=24):
        return ("🟡", "Stale")
    return ("🔴", "Stopped")


def _humanize_time(dt) -> str:
    if not dt:
        return "Never"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except Exception:
            return "Unknown"
    now = datetime.utcnow()
    delta = now - dt
    if delta < timedelta(minutes=1):
        return "Just now"
    elif delta < timedelta(hours=1):
        m = int(delta.total_seconds() / 60)
        return f"{m} minute{'s' if m != 1 else ''} ago"
    elif delta < timedelta(days=1):
        h = int(delta.total_seconds() / 3600)
        return f"{h} hour{'s' if h != 1 else ''} ago"
    d = int(delta.days)
    return f"{d} day{'s' if d != 1 else ''} ago"


def _to_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _render_paper_mode(db) -> None:
    """Render dashboard in paper trading mode."""
    # 1. Scheduler status indicator
    try:
        hb = db.get_latest_heartbeat("paper") if db.ok else None
        emoji, status_text = _calculate_status(hb)
        st.write(f"**Scheduler Status:** {emoji} {status_text}")
        last_run_str = _humanize_time(hb.get("timestamp") if hb else None)
        errors = (hb.get("stats") or {}).get("errors", "-") if hb else "-"
        st.write(f"Last run: {last_run_str}  |  Errors: {errors}")
    except Exception:
        st.write("**Scheduler Status:** unknown")

    st.divider()

    # 2. Grilla de heartbeats recientes
    st.markdown("**Recent Runs**")
    if db.ok:
        try:
            heartbeats = db.get_recent_heartbeats("paper", limit=20)
            if heartbeats:
                rows = []
                for hb in heartbeats:
                    rows.append({
                        "Time": _humanize_time(hb.get("timestamp")),
                        "Status": _calculate_status(hb)[1],
                        "Processed": (hb.get("stats") or {}).get("processed", 0),
                        "Errors": (hb.get("stats") or {}).get("errors", 0),
                        "Duration": f"{(hb.get('stats') or {}).get('duration', 0):.2f}s"
                    })
                import pandas as pd
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                # 3. Run selector
                options = [f"{_humanize_time(hb.get('timestamp'))} | {_calculate_status(hb)[1]}" for hb in heartbeats]
                selected_run_str = st.selectbox("Select a run to view details:", options)
                if selected_run_str:
                    idx = options.index(selected_run_str)
                    selected_hb = heartbeats[idx]
                    st.markdown("**Run Details**")
                    st.write(f"Timestamp: {selected_hb.get('timestamp')}")
                    st.write(f"Status: {_calculate_status(selected_hb)[1]}")
                    st.write(f"Stats: {selected_hb.get('stats')}")
                    st.write(f"Assets: {selected_hb.get('assets', [])}")
            else:
                st.info("No runs found.")
        except Exception as e:
            st.warning(f"Error loading runs: {e}")
    else:
        st.info("Connect DB to view runs.")


def _render_backtest_mode(db) -> None:
    """Render dashboard in backtesting mode."""
    # 1. Active backtest indicator
    if db.ok:
        with db.SessionLocal() as s:
            try:
                pending = s.query(db.models.BacktestRun).filter(
                    db.models.BacktestRun.total_trades.is_(None)
                ).count()
                if pending > 0:
                    st.warning(f"{pending} backtest run(s) pending/running")
            except Exception:
                pass

    st.divider()

    # 2. Grilla de backtest runs
    st.markdown("**Backtest Runs**")
    if db.ok:
        with db.SessionLocal() as s:
            try:
                runs = s.query(db.models.BacktestRun).order_by(
                    db.models.BacktestRun.created_at.desc()
                ).limit(50).all()

                if runs:
                    rows = []
                    for run in runs:
                        rows.append({
                            "id": run.id,
                            "created": _humanize_time(run.created_at),
                            "assets": ", ".join(run.assets or []),
                            "timeframe": run.timeframe,
                            "win_rate": f"{run.win_rate:.2%}" if run.win_rate is not None else "-",
                            "profit_factor": f"{run.profit_factor:.2f}" if run.profit_factor is not None else "-",
                            "sharpe_ratio": f"{run.sharpe_ratio:.2f}" if run.sharpe_ratio is not None else "-",
                            "max_drawdown": f"{run.max_drawdown:.2%}" if run.max_drawdown is not None else "-",
                            "total_pnl": f"${run.total_pnl:,.2f}" if run.total_pnl is not None else "-",
                        })
                    import pandas as pd
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                    # 3. Run selector
                    run_ids = [str(r.id) for r in runs]
                    selected_run_id = st.selectbox("Select a run to view details:", run_ids)
                    if selected_run_id:
                        selected_run = next((r for r in runs if str(r.id) == selected_run_id), None)
                        if selected_run:
                            st.markdown("**Run Details**")
                            st.write(f"ID: {selected_run.id}")
                            st.write(f"Created: {selected_run.created_at}")
                            st.write(f"Assets: {', '.join(selected_run.assets or [])}")
                            st.write(f"Timeframe: {selected_run.timeframe}")
                            st.write(f"Start date: {selected_run.start_date}")
                            st.write(f"End date: {selected_run.end_date}")
                            st.write(f"Win rate: {selected_run.win_rate:.2%}" if selected_run.win_rate is not None else "Win rate: -")
                            st.write(f"Profit factor: {selected_run.profit_factor:.2f}" if selected_run.profit_factor is not None else "Profit factor: -")
                            st.write(f"Sharpe ratio: {selected_run.sharpe_ratio:.2f}" if selected_run.sharpe_ratio is not None else "Sharpe ratio: -")
                            st.write(f"Max drawdown: {selected_run.max_drawdown:.2%}" if selected_run.max_drawdown is not None else "Max drawdown: -")
                            st.write(f"Total P&L: ${selected_run.total_pnl:,.2f}" if selected_run.total_pnl is not None else "Total P&L: -")
                else:
                    st.info("No backtest runs found.")
            except Exception as e:
                st.warning(f"Error loading runs: {e}")
    else:
        st.info("Connect DB to view runs.")


def render(db, environment: str) -> None:
    st.subheader("Dashboard")
    if environment == "paper":
        _render_paper_mode(db)
    else:
        _render_backtest_mode(db)
