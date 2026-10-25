# Bloqueo — QuantAgent-e4jc.4a

Intenté cubrir el PnL no realizado y el valor de una celda corta sintética. No dejé el cambio porque no pude cumplir las verificaciones obligatorias en este worktree.

- `timeout 5m python -m pytest -q tests/test_r78_trade_pnl_calculation.py`: 14 errores de setup; SQLite devuelve `attempt to write a readonly database` al limpiar `scheduler_heartbeats`.
- `scripts/loop/smoke.sh`: termina con `RuntimeError: cannot cache function 'set_seed_nb': no locator available` desde `vectorbt` instalado en `.venv` de solo lectura.

Decisión que destraba: habilitar una base de tests escribible para el worktree y un cache escribible compatible con Numba; luego repetir ambos comandos antes de implementar.
