"""
Tests for QuantAgent-vje: apps/streamlit/views/paper_trading.py

Covers pure-function logic for scheduler status display:
- _calculate_status (AC-6, AC-7, AC-8)
- _humanize_time
- _calculate_duration (including long-cycle edge case)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from apps.streamlit.views.paper_trading import (
    _calculate_duration,
    _calculate_status,
    _humanize_time,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _heartbeat(minutes_ago: float = 0, hours_ago: float = 0) -> dict:
    """Return a minimal heartbeat dict with timestamp offset from now."""
    ts = datetime.utcnow() - timedelta(minutes=minutes_ago, hours=hours_ago)
    return {"timestamp": ts, "stats": {}, "completed_at": None}


# ---------------------------------------------------------------------------
# _calculate_status — AC-6, AC-7, AC-8
# ---------------------------------------------------------------------------


def test_calculate_status_active_recent():
    """AC-6: Heartbeat < 2h ago → Active."""
    hb = _heartbeat(minutes_ago=30)
    emoji, text = _calculate_status(hb)
    assert emoji == "🟢"
    assert text == "Active"


def test_calculate_status_active_just_now():
    """AC-6: Heartbeat seconds ago is still Active."""
    hb = _heartbeat(minutes_ago=0)
    emoji, text = _calculate_status(hb)
    assert emoji == "🟢"
    assert text == "Active"


def test_calculate_status_stale_5h():
    """AC-7: Heartbeat 5h ago → Stale."""
    hb = _heartbeat(hours_ago=5)
    emoji, text = _calculate_status(hb)
    assert emoji == "🟡"
    assert text == "Stale"


def test_calculate_status_stale_at_boundary():
    """AC-7: Heartbeat exactly 2h ago is Stale (boundary, just past active)."""
    hb = _heartbeat(hours_ago=2, minutes_ago=1)
    emoji, text = _calculate_status(hb)
    assert emoji == "🟡"
    assert text == "Stale"


def test_calculate_status_stopped_over_24h():
    """AC-8: Heartbeat > 24h ago → Stopped."""
    hb = _heartbeat(hours_ago=25)
    emoji, text = _calculate_status(hb)
    assert emoji == "🔴"
    assert text == "Stopped"


def test_calculate_status_stopped_none_heartbeat():
    """AC-8: No heartbeat at all → Stopped."""
    emoji, text = _calculate_status(None)
    assert emoji == "🔴"
    assert text == "Stopped"


def test_calculate_status_stopped_missing_timestamp():
    """AC-8: Heartbeat with no timestamp field → Stopped."""
    emoji, text = _calculate_status({"timestamp": None, "stats": {}})
    assert emoji == "🔴"
    assert text == "Stopped"


def test_calculate_status_iso_string_timestamp():
    """_calculate_status handles ISO string timestamps (from DB serialisation)."""
    ts = datetime.utcnow() - timedelta(minutes=10)
    hb = {"timestamp": ts.isoformat(), "stats": {}}
    emoji, text = _calculate_status(hb)
    assert emoji == "🟢"
    assert text == "Active"


# ---------------------------------------------------------------------------
# _humanize_time
# ---------------------------------------------------------------------------


def test_humanize_time_just_now():
    """Under 1 minute → 'Just now'."""
    dt = datetime.utcnow() - timedelta(seconds=30)
    assert _humanize_time(dt) == "Just now"


def test_humanize_time_one_minute():
    """1 minute ago."""
    dt = datetime.utcnow() - timedelta(minutes=1)
    result = _humanize_time(dt)
    assert "minute" in result
    assert "1" in result


def test_humanize_time_plural_minutes():
    """Multiple minutes — plural form."""
    dt = datetime.utcnow() - timedelta(minutes=5)
    result = _humanize_time(dt)
    assert "5 minutes ago" == result


def test_humanize_time_one_hour():
    """1 hour ago — singular."""
    dt = datetime.utcnow() - timedelta(hours=1, minutes=5)
    result = _humanize_time(dt)
    assert "hour" in result
    assert "1 " in result


def test_humanize_time_plural_hours():
    """Multiple hours."""
    dt = datetime.utcnow() - timedelta(hours=3)
    result = _humanize_time(dt)
    assert "3 hours ago" == result


def test_humanize_time_days():
    """Multiple days."""
    dt = datetime.utcnow() - timedelta(days=3)
    result = _humanize_time(dt)
    assert "3 days ago" == result


def test_humanize_time_none():
    """None input → 'Never'."""
    assert _humanize_time(None) == "Never"


def test_humanize_time_iso_string():
    """ISO string input is parsed correctly."""
    dt = datetime.utcnow() - timedelta(minutes=15)
    result = _humanize_time(dt.isoformat())
    assert "minute" in result


def test_humanize_time_invalid_string():
    """Unrecognised string → 'Unknown'."""
    result = _humanize_time("not-a-date")
    assert result == "Unknown"


# ---------------------------------------------------------------------------
# _calculate_duration
# ---------------------------------------------------------------------------


def test_calculate_duration_from_stats():
    """Reads duration_seconds from stats when present."""
    hb = {
        "stats": {"duration_seconds": 3.5},
        "timestamp": datetime.utcnow(),
        "completed_at": None,
    }
    assert _calculate_duration(hb) == "3.5s"


def test_calculate_duration_from_timestamps():
    """Falls back to completed_at - timestamp when stats absent."""
    start = datetime.utcnow() - timedelta(seconds=8)
    end = datetime.utcnow()
    hb = {
        "stats": {},
        "timestamp": start,
        "completed_at": end,
    }
    result = _calculate_duration(hb)
    # Should be roughly 8s; just assert format
    assert result.endswith("s")
    seconds = float(result[:-1])
    assert 7.0 <= seconds <= 10.0


def test_calculate_duration_unknown_when_no_data():
    """No stats, no completed_at → 'Unknown'."""
    hb = {"stats": {}, "timestamp": datetime.utcnow(), "completed_at": None}
    assert _calculate_duration(hb) == "Unknown"


def test_calculate_duration_long_cycle_shows_minutes():
    """Edge Case 4: Very long duration (>60s) → shows minutes notation."""
    hb = {
        "stats": {"duration_seconds": 600.0},
        "timestamp": datetime.utcnow(),
        "completed_at": None,
    }
    result = _calculate_duration(hb)
    assert result.endswith("m")
    assert "10" in result


def test_calculate_duration_iso_string_timestamps():
    """Handles ISO string timestamps in completed_at / timestamp fields."""
    start = datetime.utcnow() - timedelta(seconds=5)
    end = datetime.utcnow()
    hb = {
        "stats": {},
        "timestamp": start.isoformat(),
        "completed_at": end.isoformat(),
    }
    result = _calculate_duration(hb)
    assert result.endswith("s")
