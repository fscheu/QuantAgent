# QuantAgent-lrf8.3.2: convivencia de fuentes offline

Las pruebas integrales verifican que `backtest run` y `verify` conservan filas `alpaca` junto con datos
de fixture o snapshot. Usan ambos loaders y el motor reales; la inserción ajena solo prepara la sesión.
Cada pasada de `verify` carga de nuevo y conserva las dos etiquetas.

## Verificación

- `NUMBA_CACHE_DIR=/tmp/numba-lrf8 MPLCONFIGDIR=/tmp/mpl-lrf8 python -m pytest -q tests/test_backtest_cli.py tests/test_backtest_cli_snapshot.py` — 79 passed; último ajuste snapshot: 2 passed.
- `NUMBA_CACHE_DIR=/tmp/numba-lrf8 MPLCONFIGDIR=/tmp/mpl-lrf8 scripts/loop/smoke.sh` — exit 0; RSI spy-smoke: 9 trades, PnL 561.18.
- `NUMBA_CACHE_DIR=/tmp/numba-lrf8 MPLCONFIGDIR=/tmp/mpl-lrf8 python -m pytest -q -m oraculo` — 35 passed, 1635 deselected.
- El oráculo `tests/test_data_provider.py::test_m1_references_unchanged` conserva las tres referencias M1: 340/4571.27, 6/-249.35, 6/174.34.
- Corrida RSI, SPY, snapshot `etf-1d-2026-10`, 2007-01-03–2018-12-31, costos etf, tamaño 1.0, cierre: 187 trades, PnL -9141.21.
