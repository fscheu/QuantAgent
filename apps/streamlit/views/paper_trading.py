"""Paper Trading Scheduler monitoring view for Streamlit dashboard."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import streamlit as st


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

    # Get latest heartbeat
    heartbeat = db.get_latest_heartbeat(environment)

    if not heartbeat:
        st.warning("⚠️ No scheduler heartbeat found")
        st.info(
            f"The scheduler may not be running, or no cycles have completed yet for environment: **{environment}**"
        )
        st.markdown("---")
        st.markdown("**How to start the scheduler:**")
        st.code("python apps/paper_trading.py", language="bash")
        return

    # Render status card
    _render_status_card(heartbeat)

    st.divider()

    # Render recent runs
    recent = db.get_recent_heartbeats(environment, limit=10)
    _render_recent_runs(recent)

    # Optional: Show last trade link
    last_trade_id = heartbeat.get("last_trade_id")
    if last_trade_id:
        st.divider()
        st.markdown(f"**Last Trade ID:** {last_trade_id}")
