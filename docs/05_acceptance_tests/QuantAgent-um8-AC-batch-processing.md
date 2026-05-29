# QuantAgent-um8 — AC — Batch Processing Acceptance Criteria

**Issue:** QuantAgent-um8  
**Requirements:** [QuantAgent-um8-RQ-batch-processing.md](../01_requirements/QuantAgent-um8-RQ-batch-processing.md)  
**Design:** [QuantAgent-um8-DS-batch-processing.md](../03_design/QuantAgent-um8-DS-batch-processing.md)

---

## AC-1: Default mode is sequential (no regression)

**Given** a `Backtest` initialized with default config (no `batch_mode` key)  
**When** `run()` is called  
**Then** execution follows the existing sequential path with no behavioral change  
**And** all existing tests in `tests/test_backtest.py` continue to pass

---

## AC-2: Concurrent mode dispatches asset calls in parallel

**Given** a `Backtest` initialized with `config={"batch_mode": "concurrent", "batch_max_workers": 2}`  
**And** the backtest has 2+ assets and uses `LLMAgentStrategy`  
**When** `run()` is called with a mock `trading_graph` that tracks call order  
**Then** signals for multiple assets at the same timestamp are requested concurrently (not strictly sequentially)  
**And** the final `BacktestMetrics` is equivalent to the sequential mode result

---

## AC-3: Non-LLM strategies bypass batch logic

**Given** a `Backtest` initialized with `config={"batch_mode": "concurrent"}` and a non-LLM strategy  
**When** `run()` is called  
**Then** no concurrent dispatch is attempted  
**And** the strategy's `generate_signal()` is called sequentially per period

---

## AC-4: `custom_id` maps uniquely to (symbol, timestamp)

**Given** `BatchSignalCollector.make_custom_id(symbol, timestamp)`  
**When** called with any combination of valid symbols and datetimes  
**Then** the returned string is unique per (symbol, timestamp) pair  
**And** the pair can be recovered from the string (reversible)

---

## AC-5: Partial batch failure is handled gracefully

**Given** a `provider_batch` run where some requests in the batch fail (mocked)  
**When** `poll()` returns a mix of successes and failures  
**Then** each failed request is logged as WARNING with its `custom_id`  
**And** the failed (symbol, timestamp) pair produces a HOLD signal (no trade)  
**And** the remaining successful signals are applied normally  
**And** the backtest completes (does not raise an exception)

---

## AC-6: Batch timeout raises a clear error

**Given** a `provider_batch` run with `config={"batch_timeout_hours": 0.001}` (near-zero)  
**And** the mock provider never returns `status=completed`  
**When** `run()` is called  
**Then** a `BacktestBatchTimeoutError` is raised  
**And** the error message contains the batch ID(s) and elapsed time

---

## AC-7: `provider_batch` rejected for unsupported providers

**Given** a `Backtest` with `config={"batch_mode": "provider_batch"}` and `agent_llm_provider="qwen"`  
**When** `Backtest.__init__` is called  
**Then** a `ValueError` is raised immediately  
**And** the error message names the unsupported provider and suggests `concurrent` mode

---

## AC-8: Batch config captured in BacktestRun snapshot

**Given** a successful `provider_batch` run  
**When** the `BacktestRun` record is inspected after `run()` completes  
**Then** `config_snapshot` contains `batch_mode`, `batch_size`, and `batch_provider_ids`  
**And** `batch_provider_ids` is a non-empty list of provider batch IDs

---

## AC-9: Signal traceability preserved in batch mode

**Given** a successful `provider_batch` or `concurrent` run  
**When** `Signal` records are queried for this `backtest_run_id`  
**Then** each signal has a non-null `backtest_run_id`, `model_provider`, `model_name`  
**And** the `symbol` and `generated_at` fields correctly match the (symbol, timestamp) pair  
**And** the signal is identical in structure to what the sequential mode would produce

---

## AC-10: Batch chunking respects `batch_size`

**Given** a `provider_batch` run with 250 pending requests and `config={"batch_size": 100}`  
**When** the `BatchProvider.submit()` is called  
**Then** exactly 3 separate batch API calls are made (chunks of 100, 100, 50)

---

## Test Commands

```bash
# Unit tests for batch module
pytest tests/test_batch_processing.py -v

# Regression: ensure sequential path unaffected
pytest tests/test_backtest.py -v

# Full suite
pytest tests/ -v
```
