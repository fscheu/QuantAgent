# Requirements: Track LLM token usage and runtime metrics

**Issue:** QuantAgent-69d  
**Type:** Feature  
**Priority:** P3  
**Status:** Open

## Objective
Capture real per-call LLM usage metrics so QuantAgent can analyze token consumption and execution time by session and by backtest without inferring costs from chat history or rough estimates.

## Context
- QuantAgent already persists generic execution events via `Log` / `logs`.
- Most model calls already pass through `invoke_with_retry()` in `quantagent/agent_utils.py`.
- Backtest runs already carry stable provenance via `backtest_run_id`, and per-symbol executions already generate deterministic `thread_id` values.

## Scope
### In scope
1. **Per-call LLM metrics**
   - Record, when available:
     - `input_tokens`
     - `output_tokens`
     - `total_tokens`
     - `duration_ms`
   - Record context:
     - `provider`
     - `model`
     - `operation`
     - `environment`
     - `symbol`
     - `thread_id`
     - `checkpoint_id`
     - `backtest_run_id` (stored in structured metadata)

2. **Queryable aggregates**
   - Provide aggregation by:
     - `thread_id` for paper/prod-style sessions
     - `backtest_run_id` for backtests
   - Minimum aggregate fields:
     - call count
     - token sums
     - duration sum and average
     - breakdown by `operation`

3. **Evidence-backed persistence**
   - Metrics must come from the actual model response metadata and measured runtime.
   - If a provider does not expose token counts, persist null token fields and keep runtime/context.

4. **Minimal analysis surface**
   - Provide an internal query/service surface that can return:
     - raw call rows
     - aggregated session/backtest totals

### Out of scope
- USD cost calculation or model pricing tables.
- Infrastructure metrics (CPU/RAM/network).
- Instrumenting non-LLM tool calls.
- A new standalone dashboard or large UI expansion.

## Functional requirements
### FR-1: Persist one record per LLM call
When QuantAgent completes an instrumented LLM invocation,
- **Then** one structured log record exists for that call.

### FR-2: Preserve runtime even when token metadata is absent
When the provider response does not include usage metadata,
- **Then** token fields are null
- **And** `duration_ms` and contextual identifiers are still persisted.

### FR-3: Persist failed-call telemetry
When an LLM call fails after or during execution,
- **Then** the system still records runtime/context
- **And** marks the event as an error in structured metadata.

### FR-4: Support backtest-scoped analysis
When calls occur inside a backtest,
- **Then** they are queryable by the producing `backtest_run_id`.

### FR-5: Support session-scoped analysis
When calls occur in non-backtest execution with a stable `thread_id`,
- **Then** they are queryable and aggregable by that `thread_id` only.

## Constraints
- Reuse the existing `logs` table and logging pipeline; do not introduce dedicated metrics tables in this ticket.
- Keep implementation overhead low and bounded to the LLM invocation path.
- Do not require reading secrets or provider-specific billing APIs.

## Definition of done
- Instrumented LLM calls persist structured usage/runtime telemetry in `logs`.
- Aggregation by `thread_id` and `backtest_run_id` is available through code-level queries.
- Failed calls and provider-without-usage cases produce usable telemetry.
