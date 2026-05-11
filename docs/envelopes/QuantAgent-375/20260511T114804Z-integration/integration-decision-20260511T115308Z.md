# QuantAgent-375 integration decision

- Timestamp: 2026-05-11T11:53:08Z
- Decision: MERGE
- Issue: QuantAgent-375
- Tester run id: 20260511T114804Z-QuantAgent-375-tester
- Merge strategy: `--no-ff`
- Conflict status: none
- Merge commit: `3c7467b4`
- Feature branch: `feature/QuantAgent-375-scope-replay-signal-lookup-to-selected-s`
- Feature head integrated: `ec704bad`

## Evidence reviewed

- Implementer salvage commit: `3a6a7869`
- Tester commit: `c754d4a8`
- Cleanup commit: `ec704bad`
- Targeted regression file: `tests/test_replay_signal_scoping.py`

## Verification

- `pytest tests/test_backtest.py tests/test_backtest_run_isolation.py tests/test_replay_signal_scoping.py -q` → PASS (`37 passed, 84 warnings in 4.82s`)
- `python -m compileall -q quantagent/backtesting/backtest.py quantagent/models.py tests/test_replay_signal_scoping.py` → PASS

## Scope check

Merged diff stayed within issue scope after dropping unrelated autofix noise from:
- `alembic/versions/f7d3bad02cae_add_active_positions_table.py`
- `tests/test_2mu_error_logging.py`
- `tests/test_vje_paper_trading_view.py`

Final in-scope files:
- `alembic/versions/d1e2f3a4b5c6_add_backtest_run_id_to_signals_and_replay_source_to_runs.py`
- `quantagent/backtesting/backtest.py`
- `quantagent/models.py`
- `tests/test_replay_signal_scoping.py`

## User manual

Skipped: internal backtesting/replay provenance fix, no end-user documentation surface change.
