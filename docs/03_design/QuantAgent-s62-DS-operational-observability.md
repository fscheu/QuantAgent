# Design: Extend Minimal Operational Observability

**Issue:** QuantAgent-s62  
**Related:** [RQ](../01_requirements/QuantAgent-s62-RQ-operational-observability.md) | [PL](../02_planning/QuantAgent-s62-PL-operational-observability.md) | [AC](../05_acceptance_tests/QuantAgent-s62-AC-operational-observability.md)

---

## Level of Detail

**STANDARD** — UI-layer wiring of existing primitives; no new persistence or models required.

---

## Affected Components

- `quantagent/llm_telemetry.py` — new aggregation function
- `apps/streamlit/services/db.py` — new DbHandle method
- `apps/streamlit/views/dashboard.py` — wire placeholder
- `apps/streamlit/views/paper_trading.py` — two new sections
- `apps/streamlit/views/logs.py` — environment filter

---

## Design Decisions

### DD1 — Reuse existing primitives, no new tables

`QuantAgent-69d` already persists `llm_call` log rows with `environment`, `extra_data` (tokens, duration), and `event_type`. The new `get_environment_metrics()` function filters on these fields. This requires no migrations.

`QuantAgent-vje` already defines `SchedulerHeartbeat` model and `DbHandle.get_latest_heartbeat()`. The dashboard wiring is pure UI: replace placeholder text with a call to the existing method.

### DD2 — `get_environment_metrics()` in `llm_telemetry.py`

```python
def get_environment_metrics(
    db: Any,
    environment: str,
    hours_back: int = 24,
) -> dict[str, Any]:
```

- Queries `Log` rows: `event_type='llm_call'`, `environment=environment`, `timestamp >= now - hours_back hours`.
- Delegates to `_aggregate_rows()` (reuse existing).
- Returns same shape as `get_session_metrics()`: `{calls, input_tokens_sum, output_tokens_sum, total_tokens_sum, duration_ms_sum, duration_ms_avg, by_operation}`.

### DD3 — `DbHandle.get_paper_llm_metrics()`

Thin wrapper in `db.py` to keep the view layer clean:

```python
def get_paper_llm_metrics(self, environment: str, hours_back: int = 24) -> dict:
    if not self.ok:
        return {}
    try:
        from quantagent.llm_telemetry import get_environment_metrics
        with self.SessionLocal() as session:
            return get_environment_metrics(session, environment, hours_back)
    except Exception:
        return {}
```

### DD4 — Dashboard scheduler status wiring

The Dashboard's "Scheduler Status" placeholder gets replaced with an inline snippet that calls `db.get_latest_heartbeat(environment)` and renders status/last-run. Helper functions `_calculate_status()` and `_humanize_time()` are duplicated inline (3 lines each) in dashboard.py to avoid creating a shared utils file for this single usage — consistent with AGENTS.md "minimum needed for the current task."

### DD5 — Paper Trading tab extensions

Two new `st.divider()` + section blocks are appended after the existing `_render_recent_runs()` call in `render()`:

**Block A — Positions, Orders & PnL:**
```
[Divider]
st.subheader("Positions & Orders")
  col: open positions table (Position model — all, no env filter)
  col: recent orders table (Order filtered by environment, limit 20)
[Divider]
st.subheader("PnL Summary")
  unrealized PnL from positions
  daily realized PnL from trades (today, filtered by environment)
```

**Block B — LLM Cost & Latency:**
```
[Divider]
st.subheader("LLM Cost & Latency (last 24h)")
  metrics = db.get_paper_llm_metrics(environment)
  if metrics.get("calls", 0) == 0:
      st.info("No LLM telemetry data found for this environment.")
  else:
      4 columns: calls | total_tokens | avg_latency_ms | approx_cost_usd
```

Approximate cost is computed from total tokens using a conservative constant (`~$0.60/1M tokens` as a placeholder; operator can override). If `total_tokens_sum == 0`, cost shows as `-`.

### DD6 — Logs environment filter

Add a selectbox before existing filters:
```python
env_options = ["all", "paper", "backtest"]
log_env = st.selectbox("Environment", env_options, index=0)
```

If `log_env != "all"`, apply `.filter(Log.environment == log_env)` to the query.

---

## Position Model Note

`Position` has no `environment` column. The open positions section shows all positions with a caption: _"Positions are not environment-scoped; showing all open positions."_ Adding `environment` to `Position` is out of scope for this ticket.

---

## Degradation Contract

Every new DB call is wrapped in try/except and returns empty dict/None/[]. Every UI block checks the returned value and shows an explicit `st.info()` message when no data is present. No section may propagate an exception to the Streamlit renderer.
