"""
Tests for LLM telemetry helpers (QuantAgent-69d).

Covers:
  AC-1  Successful call persists telemetry row with duration_ms > 0
  AC-2  Token fields nullable when provider response lacks usage data
  AC-3  Failed calls still produce a log row marked status=error
  AC-4  Backtest aggregation excludes rows from other backtest runs
  AC-5  Session aggregation excludes rows from other thread_ids
  AC-6  Aggregate output contains all required decision-useful fields
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from quantagent.llm_telemetry import (
    TelemetryCtx,
    _aggregate_rows,
    extract_usage,
    get_backtest_metrics,
    get_session_metrics,
    persist_llm_call,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_log_row(
    thread_id: str | None = None,
    backtest_run_id: int | None = None,
    operation: str = "test_op",
    input_tokens: int | None = 10,
    output_tokens: int | None = 5,
    total_tokens: int | None = 15,
    duration_ms: float = 100.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        event_type="llm_call",
        thread_id=thread_id,
        extra_data={
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "duration_ms": duration_ms,
            "backtest_run_id": backtest_run_id,
            "status": "success",
        },
    )


# ---------------------------------------------------------------------------
# TelemetryCtx defaults
# ---------------------------------------------------------------------------


def test_telemetry_ctx_defaults():
    ctx = TelemetryCtx()
    assert ctx.operation == ""
    assert ctx.provider == ""
    assert ctx.model == ""
    assert ctx.environment is None
    assert ctx.symbol is None
    assert ctx.thread_id is None
    assert ctx.backtest_run_id is None
    assert ctx.extra == {}


def test_telemetry_ctx_set_fields():
    ctx = TelemetryCtx(
        operation="indicator",
        provider="openai",
        model="gpt-4o-mini",
        environment="paper",
        symbol="BTC",
        thread_id="thread-abc",
        backtest_run_id=42,
        extra={"custom": True},
    )
    assert ctx.operation == "indicator"
    assert ctx.provider == "openai"
    assert ctx.model == "gpt-4o-mini"
    assert ctx.backtest_run_id == 42


# ---------------------------------------------------------------------------
# extract_usage
# ---------------------------------------------------------------------------


class _MockResponseLangchain:
    def __init__(self, usage_metadata):
        self.usage_metadata = usage_metadata


class _MockResponseOpenAILegacy:
    def __init__(self, token_usage):
        self.response_metadata = {"token_usage": token_usage}


class _MockResponseOpenAIAltKey:
    def __init__(self, usage):
        self.response_metadata = {"usage": usage}


class _MockResponseNoUsage:
    pass


def test_extract_usage_langchain_standard():
    resp = _MockResponseLangchain(
        {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    )
    usage = extract_usage(resp)
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50
    assert usage["total_tokens"] == 150


def test_extract_usage_openai_legacy_prompt_completion():
    resp = _MockResponseOpenAILegacy(
        {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120}
    )
    usage = extract_usage(resp)
    assert usage["input_tokens"] == 80
    assert usage["output_tokens"] == 40
    assert usage["total_tokens"] == 120


def test_extract_usage_openai_alt_key_input_output():
    resp = _MockResponseOpenAIAltKey(
        {"input_tokens": 60, "output_tokens": 30, "total_tokens": 90}
    )
    usage = extract_usage(resp)
    assert usage["input_tokens"] == 60
    assert usage["output_tokens"] == 30
    assert usage["total_tokens"] == 90


def test_extract_usage_no_usage_data_returns_none():
    """AC-2: when provider exposes no usage data all token fields are None."""
    resp = _MockResponseNoUsage()
    usage = extract_usage(resp)
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] is None
    assert usage["total_tokens"] is None


def test_extract_usage_langchain_partial_tokens():
    resp = _MockResponseLangchain({"input_tokens": 10})
    usage = extract_usage(resp)
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] is None
    assert usage["total_tokens"] is None


def test_extract_usage_empty_response_metadata():
    class Resp:
        response_metadata = {}

    usage = extract_usage(Resp())
    assert usage == {"input_tokens": None, "output_tokens": None, "total_tokens": None}


# ---------------------------------------------------------------------------
# _aggregate_rows (unit)
# ---------------------------------------------------------------------------


def test_aggregate_rows_empty():
    result = _aggregate_rows([])
    assert result["calls"] == 0
    assert result["input_tokens_sum"] == 0
    assert result["output_tokens_sum"] == 0
    assert result["total_tokens_sum"] == 0
    assert result["duration_ms_sum"] == 0.0
    assert result["duration_ms_avg"] == 0.0
    assert result["by_operation"] == {}


def test_aggregate_rows_single():
    row = _make_log_row(input_tokens=10, output_tokens=5, total_tokens=15, duration_ms=200.0)
    result = _aggregate_rows([row])
    assert result["calls"] == 1
    assert result["input_tokens_sum"] == 10
    assert result["output_tokens_sum"] == 5
    assert result["total_tokens_sum"] == 15
    assert result["duration_ms_sum"] == 200.0
    assert result["duration_ms_avg"] == 200.0


def test_aggregate_rows_multiple():
    rows = [
        _make_log_row(operation="indicator", input_tokens=100, output_tokens=50, total_tokens=150, duration_ms=300.0),
        _make_log_row(operation="indicator", input_tokens=200, output_tokens=100, total_tokens=300, duration_ms=100.0),
        _make_log_row(operation="decision", input_tokens=50, output_tokens=25, total_tokens=75, duration_ms=200.0),
    ]
    result = _aggregate_rows(rows)
    assert result["calls"] == 3
    assert result["input_tokens_sum"] == 350
    assert result["output_tokens_sum"] == 175
    assert result["total_tokens_sum"] == 525
    assert result["duration_ms_sum"] == 600.0
    assert result["duration_ms_avg"] == 200.0


def test_aggregate_rows_by_operation_breakdown():
    """AC-6: by_operation breakdown present with per-operation stats."""
    rows = [
        _make_log_row(operation="indicator", input_tokens=100, output_tokens=50, total_tokens=150, duration_ms=300.0),
        _make_log_row(operation="indicator", input_tokens=200, output_tokens=100, total_tokens=300, duration_ms=100.0),
        _make_log_row(operation="decision", input_tokens=50, output_tokens=25, total_tokens=75, duration_ms=200.0),
    ]
    result = _aggregate_rows(rows)
    by_op = result["by_operation"]
    assert "indicator" in by_op
    assert "decision" in by_op
    assert by_op["indicator"]["calls"] == 2
    assert by_op["indicator"]["input_tokens"] == 300
    assert by_op["decision"]["calls"] == 1
    assert by_op["decision"]["output_tokens"] == 25


def test_aggregate_rows_none_tokens_treated_as_zero():
    """AC-2: rows with null token fields still aggregate without error."""
    rows = [
        _make_log_row(input_tokens=None, output_tokens=None, total_tokens=None, duration_ms=100.0),
        _make_log_row(input_tokens=10, output_tokens=5, total_tokens=15, duration_ms=50.0),
    ]
    result = _aggregate_rows(rows)
    assert result["calls"] == 2
    assert result["input_tokens_sum"] == 10
    assert result["duration_ms_sum"] == 150.0


def test_aggregate_rows_returns_required_keys():
    """AC-6: Aggregate returns all required keys for downstream reporting."""
    result = _aggregate_rows([_make_log_row()])
    required_keys = {
        "calls", "input_tokens_sum", "output_tokens_sum", "total_tokens_sum",
        "duration_ms_sum", "duration_ms_avg", "by_operation",
    }
    assert required_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# persist_llm_call (using real in-memory DB via db_session fixture)
# ---------------------------------------------------------------------------


def test_persist_llm_call_success_writes_log_row(db_session):
    """AC-1: Successful call produces one llm_call Log row with required fields."""
    from quantagent.models import Log

    ctx = TelemetryCtx(
        operation="indicator",
        provider="openai",
        model="gpt-4o-mini",
        environment="paper",
        symbol="BTC",
        thread_id="thread-001",
    )
    response = _MockResponseLangchain(
        {"input_tokens": 120, "output_tokens": 60, "total_tokens": 180}
    )

    with patch("quantagent.database._get_session_factory", return_value=lambda: db_session):
        with patch.object(db_session, "close", return_value=None):
            persist_llm_call(ctx=ctx, status="success", duration_ms=250.0, response=response)

    rows = db_session.query(Log).filter(Log.event_type == "llm_call").all()
    assert len(rows) == 1
    row = rows[0]
    assert row.level == "INFO"
    assert row.module == "quantagent.llm_telemetry"
    assert row.symbol == "BTC"
    assert row.thread_id == "thread-001"
    ed = row.extra_data
    assert ed["status"] == "success"
    assert ed["input_tokens"] == 120
    assert ed["output_tokens"] == 60
    assert ed["total_tokens"] == 180
    assert ed["duration_ms"] > 0
    assert ed["provider"] == "openai"
    assert ed["model"] == "gpt-4o-mini"
    assert ed["operation"] == "indicator"


def test_persist_llm_call_no_usage_data_nullable_tokens(db_session):
    """AC-2: When response has no usage metadata, token fields are None."""
    from quantagent.models import Log

    ctx = TelemetryCtx(operation="decision", provider="anthropic")
    response = _MockResponseNoUsage()

    with patch("quantagent.database._get_session_factory", return_value=lambda: db_session):
        with patch.object(db_session, "close", return_value=None):
            persist_llm_call(ctx=ctx, status="success", duration_ms=100.0, response=response)

    row = db_session.query(Log).filter(Log.event_type == "llm_call").one()
    ed = row.extra_data
    assert ed["input_tokens"] is None
    assert ed["output_tokens"] is None
    assert ed["total_tokens"] is None
    assert ed["duration_ms"] > 0


def test_persist_llm_call_error_status_writes_error_row(db_session):
    """AC-3: Failed call persists row with level=ERROR and status=error."""
    from quantagent.models import Log

    ctx = TelemetryCtx(operation="trend", provider="qwen")

    with patch("quantagent.database._get_session_factory", return_value=lambda: db_session):
        with patch.object(db_session, "close", return_value=None):
            persist_llm_call(
                ctx=ctx,
                status="error",
                duration_ms=50.0,
                error_message="APITimeoutError: request timed out",
            )

    row = db_session.query(Log).filter(Log.event_type == "llm_call").one()
    assert row.level == "ERROR"
    assert "llm_call trend error" in row.message
    ed = row.extra_data
    assert ed["status"] == "error"
    assert ed["error_message"] == "APITimeoutError: request timed out"
    assert ed["duration_ms"] > 0


def test_persist_llm_call_swallows_db_exception():
    """persist_llm_call is best-effort: a DB failure must NOT propagate."""
    ctx = TelemetryCtx(operation="indicator")

    with patch("quantagent.database._get_session_factory", side_effect=Exception("DB down")):
        # Should not raise
        persist_llm_call(ctx=ctx, status="success", duration_ms=10.0)


def test_persist_llm_call_backtest_id_stored(db_session):
    from quantagent.models import Log

    ctx = TelemetryCtx(operation="decision", backtest_run_id=99)

    with patch("quantagent.database._get_session_factory", return_value=lambda: db_session):
        with patch.object(db_session, "close", return_value=None):
            persist_llm_call(ctx=ctx, status="success", duration_ms=80.0)

    row = db_session.query(Log).filter(Log.event_type == "llm_call").one()
    assert row.extra_data["backtest_run_id"] == 99


# ---------------------------------------------------------------------------
# get_session_metrics — AC-5: session isolation
# ---------------------------------------------------------------------------


def test_get_session_metrics_empty_returns_zero_calls(db_session):
    result = get_session_metrics(db_session, thread_id="thread-xyz")
    assert result["calls"] == 0


def test_get_session_metrics_filters_by_thread_id(db_session):
    """AC-5: Only rows belonging to the queried thread_id are aggregated."""
    from quantagent.models import Log

    def _log(thread_id, extra_data):
        return Log(
            timestamp=datetime.utcnow(),
            level="INFO",
            module="quantagent.llm_telemetry",
            message="llm_call indicator success",
            event_type="llm_call",
            thread_id=thread_id,
            extra_data=extra_data,
        )

    db_session.add(_log("thread-A", {"operation": "indicator", "input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "duration_ms": 200.0}))
    db_session.add(_log("thread-A", {"operation": "decision", "input_tokens": 200, "output_tokens": 100, "total_tokens": 300, "duration_ms": 300.0}))
    db_session.add(_log("thread-B", {"operation": "indicator", "input_tokens": 999, "output_tokens": 999, "total_tokens": 999, "duration_ms": 999.0}))
    db_session.commit()

    result = get_session_metrics(db_session, thread_id="thread-A")
    assert result["calls"] == 2
    assert result["input_tokens_sum"] == 300
    assert result["output_tokens_sum"] == 150
    assert result["total_tokens_sum"] == 450
    assert result["duration_ms_sum"] == 500.0


def test_get_session_metrics_excludes_non_llm_call_rows(db_session):
    """Only event_type='llm_call' rows count."""
    from quantagent.models import Log

    db_session.add(Log(
        timestamp=datetime.utcnow(),
        level="INFO",
        module="quantagent.scheduler",
        message="heartbeat",
        event_type="heartbeat",
        thread_id="thread-A",
        extra_data={"duration_ms": 5000.0},
    ))
    db_session.add(Log(
        timestamp=datetime.utcnow(),
        level="INFO",
        module="quantagent.llm_telemetry",
        message="llm_call indicator success",
        event_type="llm_call",
        thread_id="thread-A",
        extra_data={"operation": "indicator", "input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "duration_ms": 100.0},
    ))
    db_session.commit()

    result = get_session_metrics(db_session, thread_id="thread-A")
    assert result["calls"] == 1
    assert result["input_tokens_sum"] == 10


# ---------------------------------------------------------------------------
# get_backtest_metrics — AC-4: backtest isolation
# ---------------------------------------------------------------------------


def test_get_backtest_metrics_empty_returns_zero_calls(db_session):
    result = get_backtest_metrics(db_session, backtest_run_id=1)
    assert result["calls"] == 0


def test_get_backtest_metrics_filters_by_backtest_run_id(db_session):
    """AC-4: Metrics for backtest A exclude rows from backtest B."""
    from quantagent.models import Log

    def _log(backtest_run_id, input_tokens):
        return Log(
            timestamp=datetime.utcnow(),
            level="INFO",
            module="quantagent.llm_telemetry",
            message="llm_call decision success",
            event_type="llm_call",
            extra_data={
                "operation": "decision",
                "input_tokens": input_tokens,
                "output_tokens": input_tokens // 2,
                "total_tokens": input_tokens + input_tokens // 2,
                "duration_ms": 100.0,
                "backtest_run_id": backtest_run_id,
            },
        )

    db_session.add(_log(backtest_run_id=1, input_tokens=100))
    db_session.add(_log(backtest_run_id=1, input_tokens=200))
    db_session.add(_log(backtest_run_id=2, input_tokens=9999))
    db_session.commit()

    result_a = get_backtest_metrics(db_session, backtest_run_id=1)
    assert result_a["calls"] == 2
    assert result_a["input_tokens_sum"] == 300

    result_b = get_backtest_metrics(db_session, backtest_run_id=2)
    assert result_b["calls"] == 1
    assert result_b["input_tokens_sum"] == 9999


def test_get_backtest_metrics_by_operation_present(db_session):
    """AC-6: by_operation present in backtest aggregate."""
    from quantagent.models import Log

    for op in ("indicator", "pattern", "trend", "decision"):
        db_session.add(Log(
            timestamp=datetime.utcnow(),
            level="INFO",
            module="quantagent.llm_telemetry",
            message=f"llm_call {op} success",
            event_type="llm_call",
            extra_data={"operation": op, "input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "duration_ms": 50.0, "backtest_run_id": 7},
        ))
    db_session.commit()

    result = get_backtest_metrics(db_session, backtest_run_id=7)
    assert result["calls"] == 4
    assert set(result["by_operation"].keys()) == {"indicator", "pattern", "trend", "decision"}
    for op_stats in result["by_operation"].values():
        assert op_stats["calls"] == 1
        assert op_stats["input_tokens"] == 10


# ---------------------------------------------------------------------------
# invoke_with_retry + telemetry_ctx integration
# ---------------------------------------------------------------------------


def test_invoke_with_retry_persists_telemetry_on_success():
    """AC-1: invoke_with_retry calls persist_llm_call on success."""
    from quantagent.agent_utils import invoke_with_retry

    mock_fn = MagicMock(return_value="ok")
    ctx = TelemetryCtx(operation="test_op", provider="openai")

    with patch("quantagent.llm_telemetry.persist_llm_call") as mock_persist:
        result = invoke_with_retry(mock_fn, telemetry_ctx=ctx)

    assert result == "ok"
    mock_persist.assert_called_once()
    _, kwargs = mock_persist.call_args
    assert kwargs["ctx"] is ctx
    assert kwargs["status"] == "success"
    assert kwargs["duration_ms"] > 0
    assert kwargs["response"] == "ok"


def test_invoke_with_retry_persists_telemetry_on_non_retryable_error():
    """AC-3: Non-retryable error triggers persist with status=error."""
    from quantagent.agent_utils import invoke_with_retry

    class AuthErr(Exception):
        pass

    AuthErr.__module__ = "openai"
    AuthErr.__name__ = "AuthenticationError"

    mock_fn = MagicMock(side_effect=AuthErr("bad key"))
    ctx = TelemetryCtx(operation="indicator", provider="openai")

    with patch("quantagent.llm_telemetry.persist_llm_call") as mock_persist:
        with pytest.raises(AuthErr):
            invoke_with_retry(mock_fn, telemetry_ctx=ctx)

    mock_persist.assert_called_once()
    _, kwargs = mock_persist.call_args
    assert kwargs["status"] == "error"
    assert kwargs["duration_ms"] > 0
    assert "AuthenticationError" in kwargs["error_message"]


def test_invoke_with_retry_persists_telemetry_on_max_retries_exceeded():
    """AC-3: After retries exhausted, persist_llm_call called with status=error."""
    from quantagent.agent_utils import invoke_with_retry

    class RateErr(Exception):
        pass

    RateErr.__module__ = "openai"
    RateErr.__name__ = "RateLimitError"

    mock_fn = MagicMock(side_effect=RateErr("rate limited"))
    ctx = TelemetryCtx(operation="decision", provider="openai")

    with patch("time.sleep"), patch("quantagent.llm_telemetry.persist_llm_call") as mock_persist:
        with pytest.raises(RuntimeError, match="Max retries"):
            invoke_with_retry(mock_fn, retries=2, telemetry_ctx=ctx)

    mock_persist.assert_called_once()
    _, kwargs = mock_persist.call_args
    assert kwargs["status"] == "error"
    assert kwargs["duration_ms"] > 0


def test_invoke_with_retry_no_telemetry_ctx_does_not_call_persist():
    """When telemetry_ctx is None, persist_llm_call is never called."""
    from quantagent.agent_utils import invoke_with_retry

    mock_fn = MagicMock(return_value="ok")

    with patch("quantagent.llm_telemetry.persist_llm_call") as mock_persist:
        invoke_with_retry(mock_fn)

    mock_persist.assert_not_called()
