# QuantAgent-kkj.3 — RQ: Dashboard environment-aware con selector de corridas

**Issue:** QuantAgent-kkj.3  
**Parent:** QuantAgent-kkj (M2 Milestone — Paper Trading Operativo)  
**Type:** Feature / UX  
**Status:** open

---

## Contexto

El Dashboard actual de Streamlit recibe el parámetro `environment` pero su layout es idéntico en ambos modos (paper/backtest). El selector de environment en el tope no produce cambios visuales evidentes. Esto rompe el flujo esperado:

> seleccionar environment → ver corridas de ese tipo → seleccionar corrida → ver estadísticas

Observado en revisión funcional M2 (2026-05-25).

---

## Requerimientos funcionales

### RF-1: Layout diferenciado por environment

El Dashboard debe mostrar contenido distinto según el valor de `environment`:
- `"paper"` → modo paper (ver RF-2)
- `"backtest"` → modo backtest (ver RF-3)

El layout general (título, selector de environment, navegación) no cambia.

### RF-2: Modo paper

- Mostrar un **indicador de estado del scheduler** siempre visible:
  - Estados: Running / Active / Stale / Stuck / Error / Stopped
  - Fuente de datos: `SchedulerHeartbeat` más reciente via `db.get_latest_heartbeat("paper")`
- Mostrar una **grilla de corridas** (heartbeat records recientes):
  - Columnas mínimas: timestamp, status, assets procesados, errores, duración
  - Fuente: `db.get_recent_heartbeats("paper", limit=20)`
- Mostrar un **selector de corrida** (selectbox sobre los heartbeats):
  - Al seleccionar, mostrar las estadísticas de esa corrida debajo de la grilla
- Las métricas/estadísticas de la corrida seleccionada se muestran solo al seleccionar; no hay pantalla fija de métricas completa

### RF-3: Modo backtest

- Mostrar un **indicador de backtest activo** cuando haya algún `BacktestRun` con `total_trades IS NULL` (pending/running):
  - Mensaje visible si hay runs pendientes
  - Vacío si no hay runs activos
- Mostrar una **grilla de backtest runs**:
  - Columnas: id, name, created_at, status, assets, timeframe, win_rate, profit_factor, sharpe_ratio, max_drawdown, total_pnl
  - Fuente: `BacktestRun` table (últimos 50), con fallback a `st.session_state.backtest_runs`
- Mostrar un **selector de run** (selectbox sobre los IDs disponibles):
  - Al seleccionar, mostrar las métricas de ese run (win_rate, profit_factor, sharpe_ratio, max_drawdown, total_pnl)

---

## Requerimientos no funcionales

- **RNF-1:** El cambio está confinado a `apps/streamlit/views/dashboard.py`.
  No se toca `app.py`, `paper_trading.py`, ni modelos/servicios de DB.
- **RNF-2:** No se implementan métricas que no existen en el modelo de datos actual.
- **RNF-3:** Las pestañas dedicadas de Paper Trading y Backtesting no se eliminan ni modifican.

---

## Fuera de scope

- Cambios en `app.py`, `paper_trading.py`, `backtesting.py`, modelos, o servicios de DB
- Nuevas métricas no presentes en `BacktestRun` o `SchedulerHeartbeat`
- Rediseño de otras vistas (Configuration, Analyses, Replay, etc.)
- Real-time refresh automático del dashboard
