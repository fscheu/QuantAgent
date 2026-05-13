# QuantAgent-sft — Integration decision

- Issue: QuantAgent-sft
- Run ID: 20260513T180303Z-techlead-integration
- Decision: MERGE
- Merge strategy: local `--no-ff` merge into `origin/main` baseline, then push `HEAD:main`
- Conflict status: none
- Feature branch: feature/QuantAgent-sft-paper-runtime-hardening-refresh-20260513T175506Z
- Merge commit: 2f94af25432ec8828dbd3f311aa48e6419e2bca1
- Main baseline: ef913a5fcd044b3d949b057118b7aefdd3fee8c9
- Feature tip: 0d5dfb60b4c776d42e7d7e032370d9c752099e89
- User manual: skipped — `docs/user-manual/` no existe en este repo

## Evidence reviewed
- Planner docs presentes para RQ / PL / DS / AC.
- Implementación en rama fresca desde `origin/main` para evitar contaminación con `QuantAgent-s62`.
- Verificación en integration worktree:
  - `/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python -m pytest tests/test_vje_paper_trading_view.py tests/test_position_monitor.py tests/test_vje_scheduler_heartbeat_backend.py -q`
  - `/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python -m compileall -q quantagent apps tests`
- Resultado: PASS (47 tests + compileall clean)

## Scope accepted
- Heartbeat del scheduler paper distingue `running`, `error`, `stuck` y resetea estado stale.
- `ActivePosition` queda scopiado al environment paper y registra `trade_id` cuando el scheduler abre la posición.
- La UI Streamlit expone el estado operacional degradado y el último `error_message`.
- Tests cubren estados nuevos de heartbeat y scoping del `PositionMonitor`.

## Notes
- El branch histórico `feature/QuantAgent-sft-paper-runtime-hardening` estaba stale respecto de `main`; se rehabilitó en rama fresca `feature/QuantAgent-sft-paper-runtime-hardening-refresh-20260513T175506Z`.
- No se observaron conflictos de merge ni drift adicional al integrar sobre `origin/main`.
