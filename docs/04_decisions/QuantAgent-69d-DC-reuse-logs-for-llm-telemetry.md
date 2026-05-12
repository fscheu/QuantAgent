# Decision: Reuse `logs` for LLM telemetry

**Issue:** QuantAgent-69d

## Context
QuantAgent needs per-call LLM token/runtime telemetry plus session/backtest aggregation. The repo already has a `Log` model and structured `extra_data` field.

## Options considered
### Option A — New dedicated metrics tables
- Pros: stricter schema, easier SQL aggregation later
- Cons: more code, migration churn, larger blast radius for a P3 feature

### Option B — Reuse `logs` with structured `extra_data`
- Pros: minimal change, no migration, fits current observability primitives
- Cons: backtest aggregation relies on JSON metadata and disciplined payload shape

## Decision
Choose **Option B** for this ticket.

## Consequences
- Implementation stays bounded to the invocation and query layers.
- `backtest_run_id` lives in structured metadata instead of a first-class log column.
- If query volume or performance later becomes a real problem, a follow-up ticket can extract dedicated tables from proven telemetry needs.

## Status
Accepted
