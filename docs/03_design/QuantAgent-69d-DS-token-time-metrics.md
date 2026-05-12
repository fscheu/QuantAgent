# Design: Track LLM token usage and runtime metrics

**Issue:** QuantAgent-69d  
**Related:** [RQ](../01_requirements/QuantAgent-69d-RQ-token-time-metrics.md) | [AC](../05_acceptance_tests/QuantAgent-69d-AC-token-time-metrics.md)

## Level of detail
**STANDARD** — bounded feature using existing persistence and logging primitives.

## Affected components
- `quantagent/agent_utils.py`
- Agent callers that already use `invoke_with_retry(...)`
- Query/service layer that reads `Log` rows for analysis
- Optional backtest-facing service/UI read path only if a lightweight existing hook already exists

## Design choice
Use the existing `logs` table as the persistence surface for per-call LLM telemetry.

Why this is the minimal fit:
- `Log` already stores `event_type`, `extra_data`, `thread_id`, `checkpoint_id`, `environment`, and `symbol`.
- The issue requires observability and aggregation, not a brand-new metrics subsystem.
- A P3 feature does not justify new tables/migrations unless the existing log shape proves insufficient.

## Capture seam
Instrument `invoke_with_retry()` as the primary seam because current indicator/pattern/trend/decision LLM calls already flow through it.

### Required extension
Allow callers to pass a small telemetry context, for example:
- `provider`
- `model`
- `operation`
- `environment`
- `symbol`
- `thread_id`
- `checkpoint_id`
- `backtest_run_id`

The wrapper should:
1. capture start time with a monotonic clock;
2. execute the existing retry behavior unchanged;
3. on success, extract usage metadata from the response if present;
4. on failure, log an error telemetry row before re-raising.

## Persistence shape
Persist one `Log` row per call with:
- `event_type = "llm_call"`
- top-level context fields where the schema already supports them (`environment`, `symbol`, `thread_id`, `checkpoint_id`)
- `extra_data` JSON containing:
  - `provider`
  - `model`
  - `operation`
  - `status` (`success` or `error`)
  - `input_tokens`
  - `output_tokens`
  - `total_tokens`
  - `duration_ms`
  - `backtest_run_id`
  - provider-specific raw usage metadata only if already returned

## Usage extraction
Support the common LangChain response shapes in priority order:
1. `result.usage_metadata`
2. `result.response_metadata["token_usage"]`
3. provider-specific usage payloads nested under response metadata

If no usage payload exists, persist null token fields instead of inventing values.

## Aggregation
Do not materialize aggregates in this ticket.

Instead, implement query-time aggregation over `Log` rows filtered by:
- `event_type = "llm_call"`
- `thread_id = ...` for session metrics, or
- `extra_data.backtest_run_id = ...` for backtest metrics

Returned aggregate shape should include:
- `calls`
- `input_tokens_sum`
- `output_tokens_sum`
- `total_tokens_sum`
- `duration_ms_sum`
- `duration_ms_avg`
- `by_operation`

## Risks / constraints
- Some provider wrappers may expose usage metadata inconsistently; null-safe extraction is required.
- Not every LLM call in the repo may currently pass enough context into `invoke_with_retry()`; the implementation must wire only the relevant callers touched by this ticket.
- JSON-field filtering by `backtest_run_id` is acceptable for this scope; if it becomes slow later, that is a separate optimization ticket.
