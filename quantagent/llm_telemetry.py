"""
LLM telemetry helpers for QuantAgent-69d.

Provides usage extraction, Log persistence, and query-time aggregation
for LLM call metrics (tokens, duration). Uses the existing `logs` table
as the persistence surface; no new tables or migrations required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ── telemetry context ────────────────────────────────────────────────────────


@dataclass
class TelemetryCtx:
    """Optional context passed to instrumented LLM calls."""

    operation: str = ""
    provider: str = ""
    model: str = ""
    environment: str | None = None
    symbol: str | None = None
    thread_id: str | None = None
    checkpoint_id: str | None = None
    backtest_run_id: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ── usage extraction ─────────────────────────────────────────────────────────


def extract_usage(response: Any) -> dict[str, int | None]:
    """
    Extract token counts from a LangChain (or raw provider) response.

    Priority order:
    1. response.usage_metadata  (LangChain standard, dict-like)
    2. response.response_metadata["token_usage"]  (OpenAI legacy shape)
    3. Fallback: all None (provider does not expose usage)

    Returns a dict with keys: input_tokens, output_tokens, total_tokens.
    Values are int or None.
    """
    usage: dict[str, int | None] = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }

    # 1. LangChain standard usage_metadata
    usage_meta = getattr(response, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        usage["input_tokens"] = usage_meta.get("input_tokens")
        usage["output_tokens"] = usage_meta.get("output_tokens")
        usage["total_tokens"] = usage_meta.get("total_tokens")
        return usage

    # 2. OpenAI legacy shape via response_metadata
    resp_meta = getattr(response, "response_metadata", None)
    if isinstance(resp_meta, dict):
        token_usage = resp_meta.get("token_usage") or resp_meta.get("usage")
        if isinstance(token_usage, dict):
            usage["input_tokens"] = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
            usage["output_tokens"] = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
            usage["total_tokens"] = token_usage.get("total_tokens")

    return usage


# ── persistence ───────────────────────────────────────────────────────────────


def persist_llm_call(
    *,
    ctx: TelemetryCtx,
    status: str,
    duration_ms: float,
    response: Any = None,
    error_message: str | None = None,
) -> None:
    """
    Write one ``Log(event_type='llm_call')`` row to the database.

    This is best-effort: any DB exception is logged and swallowed so
    that telemetry failures never propagate to the caller.
    """
    try:
        from quantagent.database import SessionLocal
        from quantagent.models import Log

        usage = extract_usage(response) if response is not None else {"input_tokens": None, "output_tokens": None, "total_tokens": None}

        extra_data: dict[str, Any] = {
            "provider": ctx.provider or None,
            "model": ctx.model or None,
            "operation": ctx.operation or None,
            "status": status,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "duration_ms": round(duration_ms, 2),
            "backtest_run_id": ctx.backtest_run_id,
        }
        if error_message:
            extra_data["error_message"] = error_message
        if ctx.extra:
            extra_data.update(ctx.extra)

        log_entry = Log(
            timestamp=datetime.utcnow(),
            level="INFO" if status == "success" else "ERROR",
            module="quantagent.llm_telemetry",
            message=f"llm_call {ctx.operation or 'unknown'} {status}",
            event_type="llm_call",
            environment=ctx.environment,
            symbol=ctx.symbol,
            thread_id=ctx.thread_id,
            checkpoint_id=ctx.checkpoint_id,
            extra_data=extra_data,
        )

        session = SessionLocal()
        try:
            session.add(log_entry)
            session.commit()
        finally:
            session.close()

    except Exception:
        logger.debug("Failed to persist LLM telemetry row", exc_info=True)


# ── aggregation ───────────────────────────────────────────────────────────────


def _aggregate_rows(rows: list[Any]) -> dict[str, Any]:
    """Compute aggregate stats over a list of Log rows with event_type='llm_call'."""
    calls = len(rows)
    input_sum = 0
    output_sum = 0
    total_sum = 0
    duration_sum = 0.0
    by_operation: dict[str, dict[str, Any]] = {}

    for row in rows:
        ed = row.extra_data or {}
        inp = ed.get("input_tokens") or 0
        out = ed.get("output_tokens") or 0
        tot = ed.get("total_tokens") or 0
        dur = ed.get("duration_ms") or 0.0
        op = ed.get("operation") or "unknown"

        input_sum += inp
        output_sum += out
        total_sum += tot
        duration_sum += dur

        if op not in by_operation:
            by_operation[op] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "duration_ms": 0.0}
        by_operation[op]["calls"] += 1
        by_operation[op]["input_tokens"] += inp
        by_operation[op]["output_tokens"] += out
        by_operation[op]["total_tokens"] += tot
        by_operation[op]["duration_ms"] += dur

    return {
        "calls": calls,
        "input_tokens_sum": input_sum,
        "output_tokens_sum": output_sum,
        "total_tokens_sum": total_sum,
        "duration_ms_sum": round(duration_sum, 2),
        "duration_ms_avg": round(duration_sum / calls, 2) if calls else 0.0,
        "by_operation": by_operation,
    }


def get_session_metrics(db: Any, thread_id: str) -> dict[str, Any]:
    """
    Return aggregated LLM call metrics for a given thread_id.

    Args:
        db: SQLAlchemy session.
        thread_id: The session/thread identifier to filter on.

    Returns:
        Aggregate dict with calls, token sums, duration stats, by_operation.
    """
    from quantagent.models import Log

    rows = (
        db.query(Log)
        .filter(Log.event_type == "llm_call", Log.thread_id == thread_id)
        .all()
    )
    return _aggregate_rows(rows)


def get_environment_metrics(
    db: Any,
    environment: str,
    hours_back: int = 24,
) -> dict[str, Any]:
    """
    Return aggregated LLM call metrics for a given environment and time window.

    Args:
        db: SQLAlchemy session.
        environment: Environment string (e.g. 'paper', 'backtest').
        hours_back: How many hours back to look (default 24).

    Returns:
        Aggregate dict with calls, token sums, duration stats, by_operation.
    """
    from datetime import timedelta

    from quantagent.models import Log

    cutoff = datetime.utcnow() - timedelta(hours=hours_back)
    rows = (
        db.query(Log)
        .filter(
            Log.event_type == "llm_call",
            Log.environment == environment,
            Log.timestamp >= cutoff,
        )
        .all()
    )
    return _aggregate_rows(rows)


def get_backtest_metrics(db: Any, backtest_run_id: int) -> dict[str, Any]:
    """
    Return aggregated LLM call metrics for a given backtest run.

    Filters by JSON field extra_data.backtest_run_id. This is acceptable
    for current scale; optimise separately if query becomes slow.

    Args:
        db: SQLAlchemy session.
        backtest_run_id: The BacktestRun.id to filter on.

    Returns:
        Aggregate dict with calls, token sums, duration stats, by_operation.
    """
    from quantagent.models import Log

    rows = (
        db.query(Log)
        .filter(Log.event_type == "llm_call")
        .all()
    )
    # Filter in Python to stay DB-agnostic (SQLite vs Postgres JSON operators differ)
    filtered = [r for r in rows if (r.extra_data or {}).get("backtest_run_id") == backtest_run_id]
    return _aggregate_rows(filtered)
