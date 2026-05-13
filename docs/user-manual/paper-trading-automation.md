# Paper Trading Automation

The paper trading scheduler keeps QuantAgent running even when you are away. It replays the same analysis agents you use for backtesting, tags every decision as **paper**, and executes the resulting orders on a fixed interval so you can observe live behavior without touching real capital.

---

## What the Trading Scheduler Does

- **Runs unattended analysis cycles** on a timer (default: every 1 hour) using APScheduler
- **Fetches fresh market data**, runs the TradingGraph, and decides whether to go LONG, SHORT, or HOLD for each configured asset
- **Executes simulated orders** through the OrderManager with `environment="paper"` so they appear in dashboards/logs without polluting live data
- **Logs every step** (start, per-asset processing, decisions, warnings, shutdown) in structured JSON, making it easy to audit in the Logs tab or external observability tools
- **Recovers from transient failures** (API timeouts, analysis exceptions) by skipping the problem asset and moving on to the next one

> **Source:** [QuantAgent-3o4 requirements](../01_requirements/QuantAgent-3o4-RQ-trading-scheduler.md) · [Design](../03_design/QuantAgent-3o4-DS-trading-scheduler.md) · [Test suite](../05_acceptance_tests/QuantAgent-3o4-AC-trading-scheduler.md / `tests/trading/test_scheduler.py`)

---

## Prerequisites

1. **Core app installed** (database running, `.env` configured, migrations applied). See [Getting Started](getting-started.md).
2. **Strategy profiles saved** so the scheduler has risk parameters to reuse when it opens positions later through the dashboard.
3. **Scheduler settings configured** in `quantagent/settings.py` or via environment variables:

```python
scheduler = SchedulerSettings(
    enabled=True,
    interval_hours=1.0,      # float hours; must be > 0
    assets=["BTC", "SPX"],   # at least one supported symbol
    environment="paper",     # paper vs (future) live
)
```

- Set `enabled=False` if you want the CLI available but dont want the process to auto-start.
- The `assets` list is validated; unsupported tickers raise an error at startup.

---

## Starting the Scheduler

Run the dedicated entry point once your virtual environment is active:

```bash
python apps/paper_trading.py
```

Optional CLI overrides:

```bash
python apps/paper_trading.py --interval 2 --assets BTC,SPX,CL
```

- `--interval` overrides `interval_hours` on the fly (value is in hours; `0.5` = every 30 minutes).
- `--assets` accepts a comma-separated list. Whitespace is ignored and duplicates are removed before the run begins.

When the process starts youll see logs similar to:

```
INFO Scheduler started interval_hours=1.0 assets=['BTC', 'SPX'] environment='paper'
INFO Starting analysis cycle for 2 assets
```

Leave the terminal open; the APScheduler background thread will keep firing until you stop it.

---

## Monitoring Live Runs

1. **Dashboard → Dashboard tab → Scheduler Status** now reflects the real process:
   - `Active` (green) means the last heartbeat is newer than 2 hours.
   - `Stale` (yellow) means the scheduler stopped checking in recently and may need attention.
   - `Stopped` (red) means there is no usable heartbeat.
2. **Dashboard → Paper Trading tab** shows the operational detail you need after the heartbeat check:
   - recent runs,
   - open positions,
   - recent orders filtered to `environment="paper"`,
   - realized/unrealized P&L,
   - LLM cost and latency for the last 24 hours.
3. **Logs tab**: set `Environment = paper` and optionally `module = quantagent.trading.scheduler` to isolate scheduler-specific events.
4. **CLI output**: the terminal running `apps/paper_trading.py` streams the same structured logs if you need low-level debugging.

For a deeper checklist, follow the updated [Monitoring Guide](monitoring.md#scheduler-status).

---

## Stopping or Restarting Safely

- Press **Ctrl+C** (SIGINT) or send SIGTERM; the entry point registers handlers that call `TradingScheduler.stop()` and wait for the in-flight cycle to finish before exiting.
- Youll see `Scheduler stopped gracefully` when its safe to close the terminal.
- To restart, simply run the command again. The scheduler is idempotent: calling `start()` twice logs a warning and keeps the original job alive.

---

## Customizing Assets and Intervals

- **Change defaults** in `settings.scheduler` for persistent behavior, or rely on CLI overrides for one-off experiments.
- **Universe vs Scheduler Assets**: profile universes drive backtests, while the scheduler uses its own `assets` list. Keep them aligned so automated trades reflect the strategies you validated.
- **Intervals** shorter than 15 minutes increase API usage dramatically. Start with 1 hour, validate stability, then lower the interval if needed.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ValueError: interval_hours must be > 0` | Config typo | Update `settings.scheduler.interval_hours` or CLI value to a positive float |
| `ValueError: assets list cannot be empty` | Assets not set after CLI override | Provide at least one supported symbol (BTC, SPX, CL, DAX, ES, NQ, QQQ, GC, VIX, DXY) |
| Scheduler indicator shows **Stopped** | Process crashed or was never started | Check terminal logs, restart `python apps/paper_trading.py` |
| Orders missing from dashboard | Environment filtering | Confirm the Orders tab filter includes `paper`, and that the scheduler log shows successful executions |
| Repeated API warnings | Rate limits or network issues | Increase interval, reduce asset count, or investigate API credentials |

---

## Related Documentation

- Requirements: [QuantAgent-3o4-RQ-trading-scheduler.md](../01_requirements/QuantAgent-3o4-RQ-trading-scheduler.md)
- Design: [QuantAgent-3o4-DS-trading-scheduler.md](../03_design/QuantAgent-3o4-DS-trading-scheduler.md)
- Acceptance Tests: [QuantAgent-3o4-AC-trading-scheduler.md](../05_acceptance_tests/QuantAgent-3o4-AC-trading-scheduler.md)
- Test Suite: [`tests/trading/test_scheduler.py`](../../tests/trading/test_scheduler.py)

Use this guide as the operational companion to keep your paper trading environment running continuously.
