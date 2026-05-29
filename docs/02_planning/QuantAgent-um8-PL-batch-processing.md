# QuantAgent-um8 — PL — Batch Processing Implementation Plan

**Issue:** QuantAgent-um8  
**Requirements:** [QuantAgent-um8-RQ-batch-processing.md](../01_requirements/QuantAgent-um8-RQ-batch-processing.md)  
**Design:** [QuantAgent-um8-DS-batch-processing.md](../03_design/QuantAgent-um8-DS-batch-processing.md)  
**Acceptance:** [QuantAgent-um8-AC-batch-processing.md](../05_acceptance_tests/QuantAgent-um8-AC-batch-processing.md)

---

## Summary

Implement optional batch processing for LLM calls during backtesting in two modes:
- **`concurrent`**: parallel dispatch using LangGraph `.batch()` — no cost change, lower wall-clock time
- **`provider_batch`**: Anthropic/OpenAI Batch API — 50% cost reduction, async two-phase execution

Default mode (`"off"`) keeps existing sequential behavior unchanged.

---

## Implementation Steps

### Step 1 — Create `quantagent/backtesting/batch.py`

New module. Contains:
- `BatchRequest` dataclass: `custom_id`, `symbol`, `timestamp`, `kline_data`, `timeframe`, `current_price`
- `BatchSignalCollector`: accumulates requests; builds provider-format payloads
- `AnthropicBatchProvider`: submits and polls Anthropic Message Batches API
- `OpenAIBatchProvider`: submits and polls OpenAI Batch API
- `BacktestBatchTimeoutError` exception

**Files changed:** `quantagent/backtesting/batch.py` (new)

**Validation:** `python -m compileall quantagent/backtesting/batch.py`

---

### Step 2 — Modify `Backtest.__init__` for batch parameters

Read `batch_mode`, `batch_size`, `batch_timeout_hours`, `batch_max_workers` from `config` dict.
Validate: if `batch_mode == "provider_batch"` and provider not in `{"anthropic", "openai"}`, raise `ValueError`.

**Files changed:** `quantagent/backtesting/backtest.py`

---

### Step 3 — Implement `_run_sequential()` (extract current logic)

Extract the existing sequential loop body from `run()` into `_run_sequential()`.
`run()` now calls `_run_sequential()` when `batch_mode == "off"`. No behavioral change.

**Files changed:** `quantagent/backtesting/backtest.py`

**Validation:** `pytest tests/test_backtest.py -v` — all existing tests must pass.

---

### Step 4 — Implement `_run_concurrent()` (Mode A)

Replace per-asset inner loop body with `graph.batch()` across assets that have no active position
at each timestamp. Trade execution (OrderManager, PositionMonitor) remains sequential after signals
are collected.

Timeline structure: merge all assets into a single sorted timestamp list; at each timestamp,
dispatch concurrent analysis for assets without active position.

**Files changed:** `quantagent/backtesting/backtest.py`

**Validation:**
```bash
pytest tests/test_batch_processing.py::test_concurrent_mode -v
pytest tests/test_backtest.py -v
```

---

### Step 5 — Implement `_run_provider_batch()` (Mode B)

**Phase 1 — Collection:**
Run a dry-run loop: same logic as `_run_sequential()` but instead of calling `strategy.generate_signal()`,
call `collector.add(symbol, timestamp, kline_data, timeframe, price)`.

**Phase 2 — Submission:**
Call `batch_provider.submit(collector.requests, chunk_size=batch_size)` → returns list of batch IDs.
Store batch IDs in `self._batch_ids` for config snapshot.

**Phase 2 — Polling:**
Call `batch_provider.poll(batch_ids, timeout_h=batch_timeout_hours)` → returns `Dict[custom_id, TradingSignal]`.

**Phase 2 — Simulation:**
Re-run the time loop using `signal_cache.get(custom_id)` instead of LLM invocation.
Partial failures use HOLD.

**Files changed:** `quantagent/backtesting/backtest.py`

**Validation:**
```bash
pytest tests/test_batch_processing.py -v
```

---

### Step 6 — Extend `_build_config_snapshot()` for batch metadata

Add `batch_mode`, `batch_size`, `batch_provider_ids` (if applicable) to the config snapshot dict.

**Files changed:** `quantagent/backtesting/backtest.py`

---

### Step 7 — Write `tests/test_batch_processing.py`

Unit tests covering:
- `BatchSignalCollector.make_custom_id()` uniqueness and reversibility (AC-4)
- `BatchSignalCollector.build_anthropic_batch()` format validation
- `AnthropicBatchProvider`/`OpenAIBatchProvider` with mocked API client
- Partial failure handling (AC-5)
- Timeout error (AC-6)
- Unsupported provider validation (AC-7)
- Chunk size splitting (AC-10)

**Files changed:** `tests/test_batch_processing.py` (new)

---

### Step 8 — Update `docs/03_design/backtesting_engine.md`

Add a "Batch Processing" section describing the two modes, parameters, and provider notes.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Provider batch API rate limits | Low | `batch_size` config; retry with backoff in `submit()` |
| Anthropic/OpenAI API schema changes | Low | Pin SDK versions; test against real API in integration test |
| State drift in two-phase simulation | Medium | Verify signal_cache covers all collected requests; HOLD for misses |
| Sequential mode regression from refactor | Low | Step 3 extraction validated by existing test suite |
| Long poll wait (multi-hour batches) | Low | Configurable `batch_timeout_hours`; default 24h generous |

---

## Estimated Complexity

| Step | Effort |
|---|---|
| Step 1: batch.py module | Medium (~150 lines) |
| Steps 2–3: init + extract | Low |
| Step 4: concurrent mode | Low–Medium |
| Step 5: provider_batch mode | Medium–High (async polling) |
| Steps 6–7: snapshot + tests | Low–Medium |
| Step 8: doc update | Low |

**Total:** ~350–450 lines of new/modified code. ESTÁNDAR scope.

---

## Dependencies

- `anthropic` SDK (already in requirements for Anthropic provider support)
- `openai` SDK (already in requirements for OpenAI provider support)
- No new dependencies required

---

## Blocking Issues

None identified. `QuantAgent-kkj.11` (multi-provider routing) is complementary but not a hard dependency:
batch mode reads `agent_llm_provider` from existing config, same as current code.
