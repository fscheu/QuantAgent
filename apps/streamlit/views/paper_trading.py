"""Paper Trading Scheduler monitoring view for Streamlit dashboard."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import streamlit as st

from quantagent.strategy.registry import STRATEGY_REGISTRY, get_strategy_names

# PID file shared between Streamlit reruns (survives brief server restarts)
_PID_FILE = Path("/tmp/quantagent_scheduler.pid")
# Repo root: views/ → streamlit/ → apps/ → repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]


# ── PID helpers ──────────────────────────────────────────────────────────────


def _read_pid() -> Optional[int]:
    try:
        return int(_PID_FILE.read_text().strip())
    except Exception:
        return None


def _pid_is_alive(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _write_pid(pid: int) -> None:
    _PID_FILE.write_text(str(pid))


def _clear_pid() -> None:
    try:
        _PID_FILE.unlink()
    except FileNotFoundError:
        pass


def _launch_subprocess(
    assets_str: str,
    mode: str,
    interval_hours: float,
    environment: str,
    strategy: str,
    strategy_params: dict,
) -> None:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "apps" / "paper_trading.py"),
        "--environment", environment,
        "--assets", assets_str,
        "--strategy", strategy,
        "--strategy-params", json.dumps(strategy_params),
        "--enable",
    ]
    if mode == "Single cycle":
        cmd.append("--run-once")
    else:
        cmd += ["--interval-hours", str(interval_hours)]

    proc = subprocess.Popen(
        cmd,
        cwd=str(_REPO_ROOT),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _write_pid(proc.pid)


def _stop_subprocess(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        if _pid_is_alive(pid):
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    _clear_pid()


# ── Controls section ─────────────────────────────────────────────────────────


def _render_scheduler_controls(db, environment: str) -> None:
    pid = _read_pid()
    is_alive = _pid_is_alive(pid)
    hb = db.get_latest_heartbeat(environment)
    hb_running = bool(hb and hb.get("status") == "running")
    scheduler_running = is_alive and hb_running
    st.session_state.setdefault("default_strategy", {"paper": None, "backtest": None})

    strategy_names = get_strategy_names()
    default_strategy = st.session_state.default_strategy.get("paper")
    if (
        "sc_strategy_key" not in st.session_state
        or st.session_state.sc_strategy_key not in strategy_names
    ):
        st.session_state.sc_strategy_key = (
            default_strategy if default_strategy in strategy_names else "LLMAgentStrategy"
        )

    st.subheader("Scheduler Controls")

    with st.expander("▶ Start Scheduler", expanded=not scheduler_running):
        assets_input = st.text_input(
            "Assets (comma-separated)", value="BTC,SPX", key="sc_assets"
        )
        mode = st.radio("Mode", ["Single cycle", "Continuous"], key="sc_mode")
        interval_hours = 1.0
        if mode == "Continuous":
            interval_hours = st.number_input(
                "Interval (hours)", min_value=0.25, value=1.0, step=0.25, key="sc_interval"
            )

        strategy_key = st.selectbox(
            "Estrategia",
            strategy_names,
            index=(
                strategy_names.index(st.session_state.sc_strategy_key)
                if st.session_state.sc_strategy_key in strategy_names
                else 0
            ),
            key="sc_strategy_selector",
        )
        st.session_state.sc_strategy_key = strategy_key

        strategy_info = STRATEGY_REGISTRY[strategy_key]
        strategy_params = {}
        for param_key, param_spec in strategy_info.get("params", {}).items():
            default_value = param_spec.get("default")
            value = st.number_input(
                param_key.replace("_", " ").title(),
                value=default_value,
                key=f"sc_param_{param_key}",
                help=param_spec.get("description", ""),
            )
            strategy_params[param_key] = (
                int(value) if param_spec.get("type") is int else value
            )

        if strategy_info["type"] == "llm":
            st.info("Estrategia LLM — requiere modelo LLM y consume tokens.")

        if scheduler_running:
            st.warning("Scheduler is already running. Stop it before starting a new cycle.")

        if st.button("▶ Start", disabled=scheduler_running, key="sc_start"):
            st.session_state["sc_active_strategy"] = strategy_key
            _launch_subprocess(
                assets_input,
                mode,
                interval_hours,
                environment,
                strategy_key,
                strategy_params,
            )
            time.sleep(2)
            st.rerun()

    col_stop, _ = st.columns([1, 3])
    with col_stop:
        if st.button("■ Stop", disabled=not is_alive, key="sc_stop"):
            if pid is not None:
                _stop_subprocess(pid)
            time.sleep(1)
            st.rerun()


def _calculate_status(heartbeat: Optional[dict]) -> tuple[str, str]:
    """
    Calculate scheduler status based on last heartbeat.

    Args:
        heartbeat: Latest heartbeat dict or None

    Returns:
        Tuple of (status_emoji, status_text)
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

    # Status thresholds from design doc
    if time_since < timedelta(hours=2):
        return ("🟢", "Active")
    elif time_since < timedelta(hours=24):
        return ("🟡", "Stale")
    else:
        return ("🔴", "Stopped")


def _humanize_time(dt: Optional[datetime | str]) -> str:
    """
    Convert datetime to human-readable relative time.

    Args:
        dt: Datetime object or ISO string

    Returns:
        Human-readable string like "2 minutes ago"
    """
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
        minutes = int(delta.total_seconds() / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif delta < timedelta(days=1):
        hours = int(delta.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = int(delta.days)
        return f"{days} day{'s' if days != 1 else ''} ago"


def _calculate_duration(heartbeat: dict) -> str:
    """
    Calculate cycle duration from heartbeat.

    Args:
        heartbeat: Heartbeat dict

    Returns:
        Duration string like "3.5s" or from stats if available
    """
    stats = heartbeat.get("stats")
    if stats and "duration_seconds" in stats:
        duration_s = stats["duration_seconds"]
        if duration_s < 60:
            return f"{duration_s:.1f}s"
        else:
            return f"{duration_s / 60:.1f}m"

    # Fallback: calculate from timestamp and completed_at
    timestamp = heartbeat.get("timestamp")
    completed_at = heartbeat.get("completed_at")

    if timestamp and completed_at:
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))

        duration = (completed_at - timestamp).total_seconds()
        if duration < 60:
            return f"{duration:.1f}s"
        else:
            return f"{duration / 60:.1f}m"

    return "Unknown"


def _render_status_card(heartbeat: dict) -> None:
    """
    Render status card with key metrics.

    Args:
        heartbeat: Latest heartbeat dict
    """
    status_emoji, status_text = _calculate_status(heartbeat)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Status", f"{status_emoji} {status_text}")

    with col2:
        last_run = heartbeat.get("timestamp")
        st.metric("Last Run", _humanize_time(last_run))

    with col3:
        stats = heartbeat.get("stats", {})
        processed = stats.get("processed", 0)
        total = stats.get("total", 0)
        st.metric("Assets Processed", f"{processed}/{total}")

    with col4:
        duration = _calculate_duration(heartbeat)
        st.metric("Last Duration", duration)


def _render_recent_runs(recent: list[dict]) -> None:
    """
    Render table of recent scheduler runs.

    Args:
        recent: List of recent heartbeat dicts
    """
    st.subheader("Recent Runs")

    if not recent:
        st.info("No recent runs found")
        return

    # Build table data
    table_data = []
    for hb in recent:
        timestamp = hb.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_str = str(timestamp)
        else:
            time_str = str(timestamp)

        stats = hb.get("stats", {})
        status = hb.get("status", "unknown")
        processed = stats.get("processed", 0)
        errors = stats.get("errors", 0)
        duration = _calculate_duration(hb)

        # Status emoji
        if status == "completed":
            status_icon = "✅"
        elif status == "running":
            status_icon = "⏳"
        elif status == "error":
            status_icon = "❌"
        else:
            status_icon = "❓"

        table_data.append(
            {
                "Time": time_str,
                "Status": f"{status_icon} {status}",
                "Processed": f"{processed}/{stats.get('total', 0)}",
                "Errors": errors,
                "Duration": duration,
            }
        )

    st.dataframe(table_data, use_container_width=True)


def render(db, environment: str) -> None:
    """
    Main render function for Paper Trading tab.

    Args:
        db: Database service
        environment: Current environment (paper/prod/backtest)
    """
    st.header("📊 Paper Trading Scheduler")

    if not db.ok:
        st.error("❌ Database not available")
        st.info("Please check database connection and try again.")
        return

    # Lifecycle controls — always visible regardless of heartbeat state
    _render_scheduler_controls(db, environment)

    st.divider()

    # Get latest heartbeat
    heartbeat = db.get_latest_heartbeat(environment)

    if not heartbeat:
        active_strategy = st.session_state.get("sc_active_strategy")
        if active_strategy:
            st.caption(f"Running strategy: **{active_strategy}**")
        st.warning("⚠️ No scheduler heartbeat found")
        st.info(
            f"The scheduler may not be running, or no cycles have completed yet for environment: **{environment}**"
        )
        return

    # Render status card
    _render_status_card(heartbeat)
    active_strategy = st.session_state.get("sc_active_strategy")
    if active_strategy:
        st.caption(f"Running strategy: **{active_strategy}**")
    if heartbeat.get("error_message"):
        st.warning(f"Last runtime issue: {heartbeat['error_message']}")

    st.divider()

    # Render recent runs
    recent = db.get_recent_heartbeats(environment, limit=10)
    _render_recent_runs(recent)

    # Optional: Show last trade link
    last_trade_id = heartbeat.get("last_trade_id")
    if last_trade_id:
        st.divider()
        st.markdown(f"**Last Trade ID:** {last_trade_id}")

    # ── Section A: Positions, Orders & PnL ──────────────────────────────────
    st.divider()
    st.subheader("Positions & Orders")
    st.caption("Positions are not environment-scoped; showing all open positions.")

    col_pos, col_ord = st.columns(2)

    with col_pos:
        st.markdown("**Open Positions**")
        try:
            with db.SessionLocal() as s:
                positions = s.query(db.models.Position).all()
            if not positions:
                st.info("No open positions.")
            else:
                pos_data = [
                    {
                        "Symbol": p.symbol,
                        "Side": p.side.value if hasattr(p.side, "value") else str(p.side),
                        "Qty": float(p.quantity),
                        "Avg Entry": float(p.average_entry_price),
                        "Current": float(p.current_price),
                        "Unreal. PnL": float(p.unrealized_pnl),
                        "PnL %": f"{float(p.unrealized_pnl_pct):.2f}%",
                    }
                    for p in positions
                ]
                st.dataframe(pos_data, use_container_width=True)
        except Exception as e:
            st.info(f"Could not load positions: {e}")

    with col_ord:
        st.markdown("**Recent Orders**")
        try:
            with db.SessionLocal() as s:
                orders = (
                    s.query(db.models.Order)
                    .filter(db.models.Order.environment == environment)
                    .order_by(db.models.Order.created_at.desc())
                    .limit(20)
                    .all()
                )
            if not orders:
                st.info("No recent orders.")
            else:
                ord_data = [
                    {
                        "Symbol": o.symbol,
                        "Side": o.side.value if hasattr(o.side, "value") else str(o.side),
                        "Type": o.order_type.value if hasattr(o.order_type, "value") else str(o.order_type),
                        "Qty": float(o.quantity),
                        "Status": o.status.value if hasattr(o.status, "value") else str(o.status),
                        "Created": str(o.created_at)[:19] if o.created_at else "-",
                    }
                    for o in orders
                ]
                st.dataframe(ord_data, use_container_width=True)
        except Exception as e:
            st.info(f"Could not load orders: {e}")

    st.divider()
    st.subheader("PnL Summary")

    try:
        with db.SessionLocal() as s:
            positions_pnl = s.query(db.models.Position).all()
            today_start = datetime.combine(datetime.utcnow().date(), datetime.min.time())
            daily_trades = (
                s.query(db.models.Trade)
                .filter(
                    db.models.Trade.environment == environment,
                    db.models.Trade.closed_at.isnot(None),
                    db.models.Trade.closed_at >= today_start,
                )
                .all()
            )

        unrealized = sum(float(p.unrealized_pnl) for p in positions_pnl)
        realized_today = sum(
            float(t.pnl) for t in daily_trades if t.pnl is not None
        )

        pcol1, pcol2 = st.columns(2)
        pcol1.metric("Unrealized PnL (all positions)", f"${unrealized:,.2f}")
        pcol2.metric("Realized PnL today", f"${realized_today:,.2f}")
    except Exception as e:
        st.info(f"PnL data unavailable: {e}")

    # ── Section B: LLM Cost & Latency ────────────────────────────────────────
    st.divider()
    st.subheader("LLM Cost & Latency (last 24h)")

    try:
        metrics = db.get_paper_llm_metrics(environment, hours_back=24)
        if not metrics or metrics.get("calls", 0) == 0:
            st.info("No LLM telemetry data found for this environment.")
        else:
            calls = metrics["calls"]
            total_tokens = metrics.get("total_tokens_sum", 0)
            avg_latency = metrics.get("duration_ms_avg", 0.0)
            cost = (
                f"${total_tokens / 1_000_000 * 0.60:.4f}"
                if total_tokens
                else "-"
            )
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("LLM Calls", calls)
            mc2.metric("Total Tokens", f"{total_tokens:,}")
            mc3.metric("Avg Latency (ms)", f"{avg_latency:,.0f}")
            mc4.metric("Approx Cost (USD)", cost)
    except Exception as e:
        st.info(f"LLM telemetry unavailable: {e}")
