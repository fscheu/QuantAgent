"""
Tests for QuantAgent-s62: Extend Minimal Operational Observability.

Covers:
  AC3  Paper Trading tab: LLM telemetry section (get_environment_metrics,
        get_paper_llm_metrics)
  AC4  Logs view: environment filter (paper / backtest / all)
  AC5  Graceful degradation: empty/null data, DB not available

Functions under test:
  - quantagent.llm_telemetry.get_environment_metrics
  - apps.streamlit.services.db.DbHandle.get_paper_llm_metrics
  - Log environment-filter query logic (inlined from apps/streamlit/views/logs.py)
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantagent.llm_telemetry import get_environment_metrics
from quantagent.models import Base, Log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log(
    db,
    environment: str | None = "paper",
    event_type: str = "llm_call",
    input_tokens: int | None = 10,
    output_tokens: int | None = 5,
    total_tokens: int | None = 15,
    duration_ms: float = 100.0,
    minutes_ago: float = 5,
) -> Log:
    row = Log(
        timestamp=datetime.utcnow() - timedelta(minutes=minutes_ago),
        level="INFO",
        module="quantagent.llm_telemetry",
        message=f"llm_call test {environment}",
        event_type=event_type,
        environment=environment,
        extra_data={
            "operation": "test",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "duration_ms": duration_ms,
        },
    )
    db.add(row)
    db.commit()
    return row


def _make_session_factory(tables=None):
    """Create an in-memory SQLite sessionmaker with all (or selected) tables."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    if tables is None:
        Base.metadata.create_all(engine)
    else:
        for t in tables:
            t.create(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


# ---------------------------------------------------------------------------
# get_environment_metrics — AC3 / AC4 / AC5
# ---------------------------------------------------------------------------


class TestGetEnvironmentMetrics:
    """Unit tests for llm_telemetry.get_environment_metrics."""

    def test_empty_returns_zero_aggregate(self, db_session):
        """AC5: No matching rows → zero-call aggregate, no crash."""
        result = get_environment_metrics(db_session, environment="paper")
        assert result["calls"] == 0
        assert result["input_tokens_sum"] == 0
        assert result["output_tokens_sum"] == 0
        assert result["total_tokens_sum"] == 0
        assert result["duration_ms_sum"] == 0.0
        assert result["duration_ms_avg"] == 0.0
        assert result["by_operation"] == {}

    def test_returns_required_keys(self, db_session):
        """AC3: Aggregate always contains all decision-useful keys."""
        result = get_environment_metrics(db_session, environment="paper")
        required = {
            "calls",
            "input_tokens_sum",
            "output_tokens_sum",
            "total_tokens_sum",
            "duration_ms_sum",
            "duration_ms_avg",
            "by_operation",
        }
        assert required.issubset(result.keys())

    def test_filters_by_environment_paper(self, db_session):
        """AC4: Only 'paper' rows are counted when environment='paper'."""
        _log(db_session, environment="paper", input_tokens=100)
        _log(db_session, environment="paper", input_tokens=200)
        _log(db_session, environment="backtest", input_tokens=9999)

        result = get_environment_metrics(db_session, environment="paper")
        assert result["calls"] == 2
        assert result["input_tokens_sum"] == 300

    def test_filters_by_environment_backtest(self, db_session):
        """AC4: Only 'backtest' rows returned when environment='backtest'."""
        _log(db_session, environment="paper", input_tokens=9999)
        _log(db_session, environment="backtest", input_tokens=50)

        result = get_environment_metrics(db_session, environment="backtest")
        assert result["calls"] == 1
        assert result["input_tokens_sum"] == 50

    def test_excludes_rows_outside_time_window(self, db_session):
        """AC4: Rows older than hours_back are excluded from the aggregate."""
        # Recent row — should be included
        _log(db_session, environment="paper", input_tokens=10, minutes_ago=30)
        # Old row (49h ago) — must be excluded by hours_back=48
        _log(db_session, environment="paper", input_tokens=9999, minutes_ago=49 * 60)

        result = get_environment_metrics(db_session, environment="paper", hours_back=48)
        assert result["calls"] == 1
        assert result["input_tokens_sum"] == 10

    def test_excludes_non_llm_call_events(self, db_session):
        """Only event_type='llm_call' rows are counted."""
        _log(db_session, environment="paper", event_type="heartbeat", input_tokens=9999)
        _log(db_session, environment="paper", event_type="llm_call", input_tokens=10)

        result = get_environment_metrics(db_session, environment="paper")
        assert result["calls"] == 1
        assert result["input_tokens_sum"] == 10

    def test_handles_null_tokens_gracefully(self, db_session):
        """AC5: Rows with null token fields do not crash; treat nulls as 0."""
        _log(db_session, environment="paper", input_tokens=None, output_tokens=None, total_tokens=None)
        _log(db_session, environment="paper", input_tokens=20, output_tokens=10, total_tokens=30)

        result = get_environment_metrics(db_session, environment="paper")
        assert result["calls"] == 2
        assert result["input_tokens_sum"] == 20
        assert result["output_tokens_sum"] == 10

    def test_aggregates_duration_and_avg(self, db_session):
        """AC3: duration_ms_sum and duration_ms_avg computed correctly."""
        _log(db_session, environment="paper", duration_ms=100.0)
        _log(db_session, environment="paper", duration_ms=300.0)

        result = get_environment_metrics(db_session, environment="paper")
        assert result["duration_ms_sum"] == 400.0
        assert result["duration_ms_avg"] == 200.0

    def test_by_operation_breakdown_present(self, db_session):
        """AC3: by_operation breakdown is populated."""
        _log(db_session, environment="paper", input_tokens=100)
        _log(db_session, environment="paper", input_tokens=200)

        result = get_environment_metrics(db_session, environment="paper")
        assert "by_operation" in result
        assert "test" in result["by_operation"]
        assert result["by_operation"]["test"]["calls"] == 2


# ---------------------------------------------------------------------------
# DbHandle.get_paper_llm_metrics — AC3 / AC5
# ---------------------------------------------------------------------------


class TestDbHandleGetPaperLlmMetrics:
    """Unit tests for DbHandle.get_paper_llm_metrics."""

    def test_returns_empty_dict_when_db_not_ok(self):
        """AC5: No DB → returns {} without crashing."""
        from apps.streamlit.services.db import DbHandle

        db = DbHandle(ok=False, error="DATABASE_URL not set")
        result = db.get_paper_llm_metrics("paper")
        assert result == {}

    def test_returns_aggregate_dict_with_real_data(self):
        """AC3: Returns populated aggregate when DB has matching rows."""
        from apps.streamlit.services.db import DbHandle

        SessionLocal = _make_session_factory()

        # Seed one paper llm_call row
        with SessionLocal() as session:
            session.add(Log(
                timestamp=datetime.utcnow() - timedelta(minutes=10),
                level="INFO",
                module="quantagent.llm_telemetry",
                message="llm_call indicator success",
                event_type="llm_call",
                environment="paper",
                extra_data={
                    "operation": "indicator",
                    "input_tokens": 80,
                    "output_tokens": 40,
                    "total_tokens": 120,
                    "duration_ms": 250.0,
                },
            ))
            session.commit()

        db = DbHandle(ok=True, error=None, SessionLocal=SessionLocal)
        result = db.get_paper_llm_metrics("paper")

        assert result["calls"] == 1
        assert result["input_tokens_sum"] == 80
        assert result["output_tokens_sum"] == 40
        assert result["total_tokens_sum"] == 120

    def test_returns_empty_dict_when_no_matching_rows(self):
        """AC5: DB available but no rows for the environment → {} (zero-call dict)."""
        from apps.streamlit.services.db import DbHandle

        SessionLocal = _make_session_factory()
        db = DbHandle(ok=True, error=None, SessionLocal=SessionLocal)
        result = db.get_paper_llm_metrics("paper")

        # Should return a zero-call aggregate, not an exception
        assert isinstance(result, dict)
        assert result.get("calls", 0) == 0

    def test_returns_empty_dict_on_exception(self):
        """AC5: DB error during query → returns {} without propagating."""
        from apps.streamlit.services.db import DbHandle

        def bad_session_factory():
            raise RuntimeError("connection refused")

        # Make it behave as context manager factory
        class _BadFactory:
            def __call__(self):
                raise RuntimeError("connection refused")

        db = DbHandle(ok=True, error=None, SessionLocal=_BadFactory())
        result = db.get_paper_llm_metrics("paper")
        assert result == {}

    def test_respects_hours_back_parameter(self):
        """AC3: hours_back parameter is forwarded to get_environment_metrics."""
        from apps.streamlit.services.db import DbHandle

        SessionLocal = _make_session_factory()

        with SessionLocal() as session:
            # Recent row (1h ago) — should be included with hours_back=2
            session.add(Log(
                timestamp=datetime.utcnow() - timedelta(hours=1),
                level="INFO",
                module="quantagent.llm_telemetry",
                message="llm_call test",
                event_type="llm_call",
                environment="paper",
                extra_data={"operation": "test", "input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "duration_ms": 50.0},
            ))
            # Old row (3h ago) — excluded with hours_back=2
            session.add(Log(
                timestamp=datetime.utcnow() - timedelta(hours=3),
                level="INFO",
                module="quantagent.llm_telemetry",
                message="llm_call test old",
                event_type="llm_call",
                environment="paper",
                extra_data={"operation": "test", "input_tokens": 9999, "output_tokens": 9999, "total_tokens": 9999, "duration_ms": 50.0},
            ))
            session.commit()

        db = DbHandle(ok=True, error=None, SessionLocal=SessionLocal)
        result = db.get_paper_llm_metrics("paper", hours_back=2)

        assert result["calls"] == 1
        assert result["input_tokens_sum"] == 10


# ---------------------------------------------------------------------------
# Log environment filter query logic — AC4
# ---------------------------------------------------------------------------
#
# The filter logic in logs.py is:
#   if log_env != "all":
#       query = query.filter(db.models.Log.environment == log_env)
#
# We test this predicate directly against an in-memory DB to prove
# correctness without invoking Streamlit.


def _query_logs_with_env_filter(session, log_env: str) -> list:
    """Replicate the environment filter from apps/streamlit/views/logs.py."""
    query = session.query(Log)
    if log_env != "all":
        query = query.filter(Log.environment == log_env)
    return query.all()


class TestLogsEnvironmentFilter:
    """AC4: Logs view environment filter."""

    def test_paper_filter_returns_only_paper_logs(self, db_session):
        """AC4: Selecting 'paper' excludes 'backtest' and 'prod' rows."""
        _log(db_session, environment="paper", event_type="heartbeat")
        _log(db_session, environment="backtest", event_type="heartbeat")
        _log(db_session, environment="prod", event_type="heartbeat")

        rows = _query_logs_with_env_filter(db_session, "paper")
        assert len(rows) == 1
        assert rows[0].environment == "paper"

    def test_backtest_filter_returns_only_backtest_logs(self, db_session):
        """AC4: 'backtest' filter excludes 'paper'."""
        _log(db_session, environment="paper", event_type="heartbeat")
        _log(db_session, environment="backtest", event_type="heartbeat")

        rows = _query_logs_with_env_filter(db_session, "backtest")
        assert len(rows) == 1
        assert rows[0].environment == "backtest"

    def test_all_filter_returns_all_environments(self, db_session):
        """AC4: 'all' applies no environment constraint — all rows returned."""
        _log(db_session, environment="paper", event_type="heartbeat")
        _log(db_session, environment="backtest", event_type="heartbeat")
        _log(db_session, environment="prod", event_type="heartbeat")

        rows = _query_logs_with_env_filter(db_session, "all")
        assert len(rows) == 3

    def test_paper_filter_returns_empty_when_no_paper_logs(self, db_session):
        """AC5: No paper logs → empty list, no crash."""
        _log(db_session, environment="backtest", event_type="heartbeat")

        rows = _query_logs_with_env_filter(db_session, "paper")
        assert rows == []

    def test_all_filter_with_no_logs_returns_empty(self, db_session):
        """AC5: No logs at all → empty list for 'all'."""
        rows = _query_logs_with_env_filter(db_session, "all")
        assert rows == []

    def test_filter_preserves_llm_call_rows(self, db_session):
        """AC4: Environment filter works on llm_call rows (mixed event_types)."""
        _log(db_session, environment="paper", event_type="llm_call", input_tokens=10)
        _log(db_session, environment="paper", event_type="heartbeat")
        _log(db_session, environment="backtest", event_type="llm_call", input_tokens=9999)

        paper_rows = _query_logs_with_env_filter(db_session, "paper")
        assert len(paper_rows) == 2
        all_rows = _query_logs_with_env_filter(db_session, "all")
        assert len(all_rows) == 3
