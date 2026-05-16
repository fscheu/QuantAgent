# Run Report — QuantAgent-69d implementer

**Run ID:** 20260512T174841Z-QuantAgent-69d-implementer  
**Branch:** `feature/QuantAgent-69d-token-time-metrics-refresh`  
**Commit:** `4d5e043c`

## Summary

Implemented LLM token usage and runtime telemetry tracking for QuantAgent-69d.
No migrations required; uses the existing `Log` model (`event_type='llm_call'`).

## Files Changed

| File | Change |
|------|--------|
| `quantagent/llm_telemetry.py` | New module: TelemetryCtx, extract_usage(), persist_llm_call(), get_session_metrics(), get_backtest_metrics() |
| `quantagent/agent_utils.py` | Extended invoke_with_retry() with optional telemetry_ctx parameter |
| `quantagent/indicator_agent.py` | Wired TelemetryCtx (operation=indicator_agent) |
| `quantagent/pattern_agent.py` | Wired TelemetryCtx (operation=pattern_agent) |
| `quantagent/trend_agent.py` | Wired TelemetryCtx (operation=trend_agent) |
| `quantagent/decision_agent.py` | Wired TelemetryCtx (operation=decision_agent) |

## Design Decisions

- **Capture seam:** `invoke_with_retry()` — single instrumentation point covering all agent LLM calls.
- **Persistence:** Best-effort `Log` row write per call; DB errors are swallowed to avoid propagation.
- **Usage extraction:** Supports `usage_metadata` (LangChain standard) and `response_metadata.token_usage` (OpenAI legacy). Falls back to null fields if no usage data present.
- **Aggregation:** Query-time helpers (`get_session_metrics`, `get_backtest_metrics`); no materialized aggregates.
- **Timing:** `time.perf_counter()` monotonic clock; duration stored as `duration_ms` float.

## Quality Gates

| Gate | Result |
|------|--------|
| `git status --short` | clean |
| `ruff check --fix` (changed files) | PASS — All checks passed |
| `python -m compileall -q .` | PASS |
| `pytest tests/test_agent_utils_retry.py` | PASS — 48/48 |

## Acceptance Criteria Coverage

| AC | Status |
|----|--------|
| AC-1: Successful call persists telemetry | Covered by persist_llm_call on success path |
| AC-2: Token fields nullable by provider | Covered by extract_usage() null-safe fallback |
| AC-3: Failed calls produce evidence | Covered by persist_llm_call on error path |
| AC-4: Backtest aggregation isolated | Covered by get_backtest_metrics() backtest_run_id filter |
| AC-5: Session aggregation isolated | Covered by get_session_metrics() thread_id filter |
| AC-6: Aggregate output is decision-useful | Covered by _aggregate_rows() returning calls/token sums/duration stats/by_operation |

## Risks

- `persist_llm_call` is best-effort: DB errors are silently swallowed.
- `provider` and `model` fields are not auto-extracted from LLM object; they default to `""` unless callers populate them.
- JSON-field filtering for `backtest_run_id` is done in Python for portability; if this becomes slow, a separate SQL index ticket should be filed.
