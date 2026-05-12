# Planning: Track LLM token usage and runtime metrics

**Issue:** QuantAgent-69d  
**Related:** [RQ](../01_requirements/QuantAgent-69d-RQ-token-time-metrics.md) | [DS](../03_design/QuantAgent-69d-DS-token-time-metrics.md) | [AC](../05_acceptance_tests/QuantAgent-69d-AC-token-time-metrics.md)
**Complexity:** STANDARD

## Tasks

### Task 1: Confirm the live invocation seam (~0.5h)
- Verify every in-scope model call still flows through `invoke_with_retry()`.
- Identify the minimal set of callers that must pass telemetry context.

### Task 2: Add telemetry capture to retry wrapper (~1h)
- Extend `invoke_with_retry()` with optional telemetry/context input.
- Measure duration with a monotonic clock.
- Persist one `Log(event_type="llm_call")` row on success and on final failure.

### Task 3: Wire caller context (~1h)
- Pass `provider`, `model`, `operation`, `environment`, `symbol`, `thread_id`, and `backtest_run_id` from the relevant agent/backtest paths.
- Keep unrelated call sites untouched.

### Task 4: Add read-side aggregation helpers (~1h)
- Implement query helpers that return:
  - raw rows for a session/backtest
  - aggregate totals and per-operation breakdown
- Reuse the existing `Log` model; no new ORM tables.

### Task 5: Validate with focused tests (~1h)
- Add tests for:
  - success telemetry
  - failure telemetry
  - provider-without-usage handling
  - thread/backtest aggregation isolation

## Risks
- Some response objects may hide usage metadata in different keys.
- Existing call sites may not all have immediate access to `backtest_run_id`; backtest-only paths should be prioritized first.

## Rollout
- Implement without migrations.
- Prefer backtest validation first because it provides deterministic provenance and easier aggregate verification.

## Validation commands
```bash
ruff check --fix .
python -m pytest tests/ -k "69d or token or usage" -v
python -m compileall -q .
```
