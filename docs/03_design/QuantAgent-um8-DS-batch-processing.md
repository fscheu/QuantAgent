# QuantAgent-um8 — DS — Batch Processing Design

**Issue:** QuantAgent-um8  
**Requirements:** [QuantAgent-um8-RQ-batch-processing.md](../01_requirements/QuantAgent-um8-RQ-batch-processing.md)

---

## Architectural Context

The current backtesting loop in `Backtest.run()` is:

```
for asset in assets:
    for timestamp in asset_dates:
        _analyze_and_trade(asset, timestamp)   # 1 LLM call each
```

LLM calls happen inside `LLMAgentStrategy.generate_signal()` → `graph.invoke()`.
These calls are independent **across assets at the same timestamp** but sequential
**across timestamps for the same asset** because portfolio state evolves with each trade.

---

## Key Constraint: Why Full Batching Is Not Possible

Trade execution at timestamp T affects portfolio state at T+1 (position sizing, daily loss
limits). Therefore the outer time loop CANNOT be fully pre-batched and replayed.

However, signal generation (the LLM analysis) is stateless: it only reads market data and
produces a `TradingSignal`. The actual state mutation happens in `execute_decision()` after.
This means signal generation requests CAN be batched while trade execution remains sequential.

---

## Design: Two Batch Modes

### Mode A — `concurrent` (LangGraph `.batch()`)

Dispatch all asset analysis calls at the same timestamp in a thread pool.

```
for timestamp in merged_timeline:
    assets_without_position = [a for a in assets if not active_pos(a)]
    signals = graph.batch([state_for(a) for a in assets_without_position],
                          config={"max_concurrency": batch_max_workers})
    for asset, signal in zip(assets_without_position, signals):
        _apply_signal_and_trade(asset, timestamp, signal)
    for asset in assets_with_position:
        _check_exit(asset, timestamp)
```

**Pros:** No provider API changes, immediate results, no polling.  
**Cons:** No cost reduction (full API pricing per call).

### Mode B — `provider_batch` (Anthropic / OpenAI Batch APIs)

Two-phase execution:

**Phase 1 — Collection:** Run the full backtest in "dry-run" mode: compute market data,
check active positions, but instead of invoking the graph, collect `(custom_id, messages)`
tuples for all calls that would be made.

**Phase 2 — Submission + Simulation:** Submit the collected requests to the provider batch
API in chunks of `batch_size`. Poll until all chunks complete. Map results back by `custom_id`
into a `signal_cache: Dict[(symbol, timestamp), TradingSignal]`. Re-run the time loop,
this time reading signals from `signal_cache` instead of invoking the LLM.

```
# Phase 1
pending: List[BatchRequest] = collect_batch_requests()

# Phase 2 — submit
batch_ids = submit_in_chunks(pending, chunk_size=batch_size)

# Phase 2 — poll
results: Dict[custom_id, TradingSignal] = poll_until_complete(batch_ids, timeout_h=batch_timeout_hours)

# Phase 2 — simulate
for timestamp in merged_timeline:
    signal = results.get(make_custom_id(asset, timestamp))
    _apply_signal_and_trade(asset, timestamp, signal)
```

**Pros:** 50% cost reduction on provider-batch-eligible requests.  
**Cons:** Asynchronous — results not immediate; requires polling loop; Anthropic/OpenAI only.

---

## Component Design

### 1. `BatchSignalCollector` (new, `quantagent/backtesting/batch.py`)

Responsibility: collect LLM invocation requests during the dry-run phase of `provider_batch` mode.

```python
@dataclass
class BatchRequest:
    custom_id: str          # f"{symbol}__{timestamp.isoformat()}"
    symbol: str
    timestamp: datetime
    kline_data: List[Dict]
    timeframe: str
    current_price: float

class BatchSignalCollector:
    requests: List[BatchRequest]
    def add(self, symbol, timestamp, kline_data, timeframe, price): ...
    def make_custom_id(self, symbol, timestamp) -> str: ...
    def build_anthropic_batch(self) -> List[dict]: ...    # Anthropic request format
    def build_openai_jsonl(self) -> str: ...              # OpenAI JSONL format
```

### 2. `BatchProvider` (new, `quantagent/backtesting/batch.py`)

Responsibility: submit and poll provider batch APIs.

```python
class BatchProvider(Protocol):
    def submit(self, requests: List[dict], chunk_size: int) -> List[str]: ...
    def poll(self, batch_ids: List[str], timeout_h: float) -> Dict[str, TradingSignal]: ...

class AnthropicBatchProvider:    # implements BatchProvider
    ...

class OpenAIBatchProvider:       # implements BatchProvider
    ...
```

### 3. `Backtest` modifications (`quantagent/backtesting/backtest.py`)

- Add `batch_mode`, `batch_size`, `batch_timeout_hours`, `batch_max_workers` to `__init__` (read from `config`).
- Add `_run_concurrent()` (Mode A): replaces inner loop body with `graph.batch()`.
- Add `_run_provider_batch()` (Mode B): two-phase orchestration.
- `run()` dispatches to the right path based on `batch_mode`.

```python
def run(self, name=None) -> BacktestMetrics:
    ...
    if self.batch_mode == "concurrent":
        self._run_concurrent()
    elif self.batch_mode == "provider_batch":
        self._run_provider_batch()
    else:
        self._run_sequential()   # existing code, unchanged
    ...
```

### 4. `custom_id` traceability

Format: `"{symbol}__{timestamp.isoformat()}"` (URL-safe, reversible).
The batch results dict is keyed by this same string. After Phase 2 poll, each matched result
is written as a `Signal` ORM record with the same fields as the sequential path (FR-4).

---

## Provider API Notes

### Anthropic Message Batches

- SDK: `anthropic.beta.messages.batches.create(requests=[...])`
- Each request: `{"custom_id": str, "params": {"model": ..., "max_tokens": ..., "messages": [...]}}`
- Polling: `client.beta.messages.batches.retrieve(batch_id)` → check `.processing_status == "ended"`
- Results: `client.beta.messages.batches.results(batch_id)`
- Limits: 10,000 requests / batch; results valid for 29 days.
- Cost: 50% off standard pricing.

### OpenAI Batch API

- Input: JSONL file upload via `files.create()`; then `batches.create(input_file_id=...)`
- Each line: `{"custom_id": str, "method": "POST", "url": "/v1/chat/completions", "body": {...}}`
- Polling: `client.batches.retrieve(batch_id)` → check `.status == "completed"`
- Limits: 50,000 requests or 100 MB per batch.
- Cost: 50% off standard pricing.

---

## Error Handling

| Failure case | Behavior |
|---|---|
| Provider batch API unavailable | Fall back to sequential mode; log warning |
| Individual request failed in batch | Log `WARNING` with `custom_id`; treat as HOLD signal for that (symbol, timestamp) |
| Batch timeout exceeded | Raise `BacktestBatchTimeoutError`; run aborts cleanly |
| Provider not Anthropic/OpenAI in `provider_batch` mode | Raise `ValueError` at `Backtest.__init__`; suggest `concurrent` mode |

---

## Config Snapshot Extension

Add batch config to `BacktestRun.config_snapshot`:

```json
{
  "batch_mode": "provider_batch",
  "batch_size": 100,
  "batch_provider_ids": ["batch_abc123", "batch_def456"],
  "batch_timeout_hours": 24.0
}
```

---

## Files to Create / Modify

| Action | File | Purpose |
|---|---|---|
| Create | `quantagent/backtesting/batch.py` | `BatchRequest`, `BatchSignalCollector`, `AnthropicBatchProvider`, `OpenAIBatchProvider` |
| Modify | `quantagent/backtesting/backtest.py` | Dispatch logic, `_run_concurrent()`, `_run_provider_batch()` |
| Create | `tests/test_batch_processing.py` | Unit tests for batch collector, ID mapping, partial failure |
| Modify | `docs/03_design/backtesting_engine.md` | Add batch processing section |
