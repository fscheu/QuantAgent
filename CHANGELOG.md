# Changelog

## 2026-09-26 — Cierre del ciclo PLAN-30-DIAS: `quantagent backtest run`

Ciclo: 2026-09-14 → 2026-09-17 (ver `PLAN-30-DIAS.md`). Rama: `feature/plan30`.
Estado verificado el 2026-09-26 sobre `feature/plan30` (`1321db6b`).

### Qué hace hoy el sistema

Corre un backtest determinista, sin red ni API keys, sobre datos OHLCV versionados en el repo,
e imprime 5 métricas. Opcionalmente escribe el trade log completo a CSV.

- Estrategias seleccionables: `rsi`, `fifty-two-week-high`, `triple-screen`.
- Fixtures versionados: `tests/fixtures/spy-90d.csv` (1h, 2160 velas), `tests/fixtures/spy-smoke.csv` (1h, 120 velas).
- Cada `Trade` queda asociado a su corrida (`trades.backtest_run_id`, migración Alembic).
- `entry_time` / `exit_time` del CSV son la fecha de la vela simulada, no la hora de reloj.

### Cómo se corre

```bash
source .venv/bin/activate
python -m quantagent.cli backtest run --strategy rsi --fixture spy-90d --out run.csv
```

Salida verificada el 2026-09-26:

```
Trades: 227
Win rate: 49.78%
Profit factor: 0.61
Sharpe ratio: 0.40
Total PnL: -4717.33
Trade log written to run.csv
```

`run.csv` tiene 228 líneas (header + 227 trades) con las columnas
`entry_time,exit_time,symbol,side,qty,entry_price,exit_price,stop_loss,pnl,exit_reason`.
Referencia completa del comando: sección "Run a Deterministic Backtest (CLI)" del `README.md`.

Tests: `pytest -q -m "not slow and not api"` → 781 passed, 0 failed, 22 skipped (2026-09-26).

### Tickets cerrados en el ciclo

`QuantAgent-iip` (P0, data bleed entre timeframes), `QuantAgent-gg6` (trade log a CSV, entregado por CLI),
`QuantAgent-83e` (timestamps de vela en el trade log).

### Limitaciones conocidas

- `fifty-two-week-high` y `triple-screen` terminan con exit 0 pero **0 trades** sobre `spy-90d`:
  el fixture no cumple sus condiciones de entrada. Solo RSI está demostrado punta a punta.
- Antes de las 5 líneas de métricas se imprimen mensajes "Insufficient data for SPY at ..." y dos
  `SAWarning` de SQLAlchemy. El README dice que la salida son solo esas 5 líneas.
- Las fórmulas de las métricas no están auditadas contra un cálculo independiente.

### Qué quedó fuera

- Botón de descarga en Streamlit (D19). La vista de backtesting de Streamlit sigue sin ejecutar backtests.
- Auditoría de métricas (`QuantAgent-hx0`), reproducibilidad (`QuantAgent-y8z`), test golden (`QuantAgent-piv`),
  cross-validation (`QuantAgent-832`).
- Ejecución background de backtests (`QuantAgent-u0w`), broker Alpaca y todo M2.
- El plan de mes 2 (`docs/02_planning/2026-09-17_plan_mes_2_loop_ai.md`) no se ejecutó.
  Lo reemplaza `PLAN-CONTINUACION.md`.
