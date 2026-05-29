# QuantAgent-um8 — RQ — Batch Processing for Backtesting LLM Calls

**Issue:** QuantAgent-um8  
**Type:** Enhancement / Optimization  
**Labels:** backtesting, enhancement, optimization  
**Priority:** 3 (ESTÁNDAR)

---

## Context

Backtesting currently invokes the LLM pipeline once per (asset, timestamp) pair in a sequential loop.
For a 3-month, 4h-timeframe, 3-asset backtest this produces roughly 1,600–2,000 LLM calls.
Provider batch APIs (Anthropic and OpenAI) offer a 50% price reduction for async batch submissions,
and LangGraph's built-in `.batch()` method enables concurrent execution without provider changes.

Neither capability is currently used. All LLM calls go through `trading_graph.graph.invoke()`, one at a time.

---

## Functional Requirements

### FR-1: Concurrent signal generation for same-timestamp assets

When a backtest involves multiple assets and no active position exists for any of them at a given
timestamp, the LLM analysis calls for all those assets at that timestamp MUST be dispatched
concurrently (not sequentially).

### FR-2: Provider Batch API submission (Anthropic / OpenAI)

When the configured LLM provider is Anthropic or OpenAI, the backtest MUST offer an opt-in
mode that collects all pending analysis requests and submits them via the provider's Batch API
before the trade-simulation loop begins.

Requirements:
- FR-2a: All requests in the batch must carry a unique `custom_id` that maps back to `(symbol, timestamp)`.
- FR-2b: The batch submission must be recorded (batch ID + provider) in the `BacktestRun` config snapshot.
- FR-2c: Results must be polled until complete (status = `ended` / `completed`) before simulation starts.
- FR-2d: Partial failures (some requests failed in the batch) must be logged per-request; remaining
  successful results must still be applied to the simulation.
- FR-2e: Timeout: if results are not available within a configurable window (`batch_timeout_hours`,
  default 24 h), the run must fail with a clear error.

### FR-3: Batching parameters configurable at run time

The following parameters must be configurable via the `Backtest` constructor's `config` dict:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_mode` | str | `"off"` | `"off"` / `"concurrent"` / `"provider_batch"` |
| `batch_size` | int | 100 | Max requests per provider batch submission |
| `batch_timeout_hours` | float | 24.0 | Max wait time for provider batch results |
| `batch_max_workers` | int | 4 | Thread pool size for concurrent mode |

### FR-4: Result traceability

Signals generated via batch path must be stored identically to signals from the sequential path:
same `Signal` ORM model, same `backtest_run_id` foreign key, same `model_provider/model_name` fields.
No new DB schema required.

### FR-5: Non-LLM strategies unaffected

Non-LLM strategies (e.g., `TripleScreenStrategy`, `52WeekHighMomentumStrategy`) must bypass all
batch logic. Batch processing applies only when the active strategy is `LLMAgentStrategy`.

---

## Out of Scope

- Streaming / real-time batch calls during paper or live trading
- Batch processing for vision sub-agents (pattern_agent, trend_agent) independently
- Caching or deduplication of identical LLM prompts across backtest runs
- New database tables or Alembic migrations
- UI controls for batch configuration (Streamlit)
