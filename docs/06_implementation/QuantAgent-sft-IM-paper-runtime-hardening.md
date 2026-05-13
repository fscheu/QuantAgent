# QuantAgent-sft — Implementation: paper runtime hardening

## Overview
- Scoped scheduler-managed `ActivePosition` rows to the paper environment instead of relying on the model default.
- Persisted `trade_id` when the paper scheduler opens an active position so the `signal -> order -> trade -> position` chain remains reconstructible.
- Hardened scheduler heartbeat state transitions so recovered stale `running` rows and fatal cycle failures become observable.
- Updated the Streamlit paper runtime view to distinguish `running`, `stuck`, and `error` states instead of inferring health only from recency.

## Approach
- `quantagent/trading/position_monitor.py`
  - Added optional environment context to `PositionMonitor`.
  - Filtered active-position lookups by environment when that context is present.
  - Persisted the resolved environment when opening a position.
- `quantagent/trading/scheduler.py`
  - Instantiated `PositionMonitor` with `Environment.PAPER`.
  - Resolved `trade_id` from the filled order before creating the `ActivePosition`.
  - Reset heartbeat completion fields on new cycle start, flag stale recovered `running` rows, and added explicit heartbeat `error` updates for fatal cycle failures.
  - Scoped `last_trade_id` lookup to the scheduler environment.
- `apps/streamlit/views/paper_trading.py`
  - Surfaced heartbeat `error_message`.
  - Mapped recent `running` heartbeats to `Running`, stale `running` heartbeats to `Stuck`, and explicit heartbeat failures to `Error`.

## Deviations
- No test files were modified in this run because the executor contract for this phase explicitly disabled `write_tests`.

## Validation
- `/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python -m pytest tests/test_vje_scheduler_heartbeat_backend.py tests/test_vje_paper_trading_view.py -q`
- `/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python -m ruff check --fix apps/streamlit/views/paper_trading.py quantagent/trading/position_monitor.py quantagent/trading/scheduler.py docs/06_implementation/QuantAgent-sft-IM-paper-runtime-hardening.md`
- `/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python -m compileall -q quantagent apps`
