# QuantAgent-aki — Pilot Runbook: Controlled Paper Trading Pilot

**Issue:** QuantAgent-aki  
**Date:** 2026-05-14  
**Author:** autodev-implementer  
**Related:** [RQ](../01_requirements/QuantAgent-aki-RQ-paper-pilot.md) | [PL](../02_planning/QuantAgent-aki-PL-paper-pilot.md) | [AC](../05_acceptance_tests/QuantAgent-aki-AC-paper-pilot.md)

---

## Pilot Configuration

| Parameter       | Value                                         |
|----------------|-----------------------------------------------|
| Strategies     | `RSIMeanReversionStrategy`, `FiftyTwoWeekHighStrategy` |
| Universe       | SPY, AAPL, MSFT                              |
| Cycles         | 3 (default, overridable via `--cycles`)       |
| Environment    | `paper`                                       |
| Timeframe      | `1h`                                          |
| Lookback       | `168h` (7 days of hourly bars)               |
| Capital        | Paper (no real money)                         |
| LLM calls      | None (deterministic strategies only)         |

---

## Pre-run Checklist

Before running the pilot, verify all of the following:

1. **DATABASE_URL set**: `echo $DATABASE_URL` must return a non-empty string, OR the repo's `.env` file must contain it.
2. **DB reachable and migrations current**: The pilot script performs a DB ping on startup. Alembic schema must include `scheduler_heartbeats`, `signals`, `orders`, `trades`, `active_positions`.
3. **yfinance reachable**: Internet access is required to fetch SPY, AAPL, MSFT 1h bars.
4. **Shared venv available**: `/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python` must be accessible.
5. **Blocking issues closed**: QuantAgent-sft, QuantAgent-s62, QuantAgent-339 must be in stable state (confirmed before this pilot was scheduled).

---

## How to Run

Run from the **main repo directory** (so `.env` is resolved correctly):

```bash
cd /home/azureuser/repos/projects/QuantAgent

/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python \
  scripts/run_paper_pilot.py \
  --cycles 3 \
  --tickers SPY AAPL MSFT \
  --output-dir docs/envelopes/QuantAgent-aki/20260514T074352Z-QuantAgent-aki-implementer
```

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--cycles N` | 3 | Number of trading cycles to execute |
| `--tickers A B C` | SPY AAPL MSFT | Assets to include in pilot |
| `--output-dir PATH` | `.` | Directory for `pilot_evidence.json` and `readiness_report.md` |

---

## Expected Output Files

| File | Location | Description |
|------|----------|-------------|
| `pilot_evidence.json` | `--output-dir` | Machine-readable evidence (all cycle data) |
| `readiness_report.md` | `--output-dir` | Human-readable go/no-go report |
| Stdout | Terminal | Per-cycle progress log |

---

## Success / Failure Exit Criteria

### Success (exit code 0)
- At least 1 cycle completed without a Python exception crashing the cycle
- `pilot_evidence.json` is written and parseable
- `readiness_report.md` is written with an explicit GO/NO-GO/CONDITIONAL GO verdict

### Blocked (exit code 1 — expected blockers)
- DATABASE_URL not set or DB unreachable
- Alembic schema missing required tables
- yfinance returns empty data for all tickers across all cycles

### Failure (unexpected, exit code 1)
- Python import error in quantagent package
- Unhandled exception during cycle execution not captured by error handling

### Valid thin-evidence outcome
- 3 cycles completed, 0 signals generated → still a SUCCESS exit (runtime health confirmed, signal thresholds not met under current market conditions). The readiness report will note this explicitly.

---

## Notes for Interpreting Results

- The 52-week high strategy requires ~302 hourly bars (252 lookback + SMA period). With 168h lookback, it will return HOLD on every cycle. This is expected.
- RSI signals fire only when RSI < 30 or RSI > 70. In normal market conditions, signals may not appear in 3 cycles.
- Zero trades does not mean the pilot failed — it means the runtime is healthy and deterministic strategies did not trigger.
