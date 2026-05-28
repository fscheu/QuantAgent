# QuantAgent-kkj.3 — PL: Dashboard environment-aware con selector de corridas

**Issue:** QuantAgent-kkj.3  
**Phase:** planner  
**Run ID:** 20260528T213653Z-QuantAgent-kkj.3-planner

---

## Objetivo

Reestructurar `apps/streamlit/views/dashboard.py` para que el layout cambie según `environment` (paper/backtest), mostrando una grilla de corridas y un selector para ver estadísticas de una corrida específica.

---

## Estado actual (pre-cambio)

- `dashboard.py` tiene una función `render(db, environment)` con layout único:
  - 5 KPIs fijos (Portfolio Value, Daily P&L, Win Rate, Open Positions, Open Orders)
  - Recent Trades table
  - Scheduler Status simplificado (columna derecha)
- El `environment` se usa solo para filtrar trades y orders; el layout no cambia
- Tiene funciones helper duplicadas (`_calculate_status`, `_humanize_time`) respecto a `paper_trading.py`, pero con lógica menos completa (no distingue Running/Stuck/Error)
- `app.py` pasa `environment` correctamente; no requiere cambio

---

## Diseño de implementación

### Archivo único a modificar

`apps/streamlit/views/dashboard.py` — reestructura completa de la función `render()` y helpers.

### Estructura propuesta

```python
def render(db, environment: str) -> None:
    st.subheader("Dashboard")
    if environment == "paper":
        _render_paper_mode(db)
    else:
        _render_backtest_mode(db)
```

### Modo paper: `_render_paper_mode(db)`

```
1. Scheduler status indicator (siempre visible al tope)
   - Llama db.get_latest_heartbeat("paper")
   - Muestra emoji + texto: Running / Active / Stale / Stuck / Error / Stopped
   - Reutiliza la lógica de _calculate_status de paper_trading.py (más completa)
   
2. st.divider()

3. Grilla de corridas (heartbeats recientes)
   - db.get_recent_heartbeats("paper", limit=20)
   - Columnas: Time, Status, Processed, Errors, Duration
   - Si vacío: st.info("No runs found")

4. Run selector
   - selectbox con opciones: "<timestamp> | <status>" para cada heartbeat
   - Si hay runs: muestra stats de la corrida seleccionada
     - Stats, assets, duration, errors
```

### Modo backtest: `_render_backtest_mode(db)`

```
1. Active backtest indicator
   - Query BacktestRun WHERE total_trades IS NULL
   - Si hay runs pendientes: st.warning("N backtest run(s) pending/running")
   - Si no: nothing rendered

2. st.divider()

3. Grilla de backtest runs
   - Query BacktestRun ORDER BY created_at DESC LIMIT 50
   - Fallback: st.session_state.backtest_runs
   - Columnas: id, name/created_at, status, assets, timeframe, win_rate, profit_factor, sharpe_ratio, max_drawdown, total_pnl

4. Run selector
   - selectbox con opciones por run ID
   - Si run seleccionado: muestra métricas de ese run
     - win_rate, profit_factor, sharpe_ratio, max_drawdown, total_pnl
     - assets, timeframe, start/end date
```

### Helpers a actualizar

- Reemplazar `_calculate_status` actual (simplificado) con la versión completa de `paper_trading.py` (que maneja Running/Stuck/Error además de Active/Stale/Stopped)
- Mantener `_humanize_time` y `_to_float` (sin cambio)
- Eliminar los 5 KPIs del layout paper (no son environment-aware suficientes para ser el contenido principal del dashboard)

---

## Decisiones clave

| Decisión | Elegida | Alternativa descartada | Razón |
|---|---|---|---|
| Fuente de "corridas paper" | `SchedulerHeartbeat` records | Agregar tabla nueva | No requiere cambios de modelo |
| Fuente de "corridas backtest" | `BacktestRun` tabla existente | Solo session_state | Persistencia real en DB |
| Indicador "backtest activo" | `total_trades IS NULL` | Campo `status` nuevo | No requiere migración |
| Scope de cambio | Solo `dashboard.py` | Refactor shared utils | Mínimo impacto, máxima claridad |
| KPIs | Eliminar del layout principal | Mantener en ambos modos | No son environment-aware; generan confusión |

---

## Archivos afectados

| Archivo | Tipo de cambio |
|---|---|
| `apps/streamlit/views/dashboard.py` | Reestructura completa de `render()` |

## Archivos NO afectados

| Archivo | Razón |
|---|---|
| `apps/streamlit/app.py` | El environment selector y routing ya son correctos |
| `apps/streamlit/views/paper_trading.py` | Solo referencia; no se modifica |
| `apps/streamlit/views/backtesting.py` | Vista dedicada sigue existiendo sin cambios |
| `apps/streamlit/services/db.py` | `get_recent_heartbeats` y query de `BacktestRun` son suficientes |
| `quantagent/models.py` | Sin cambios de schema |
| Migraciones Alembic | No requeridas |

---

## Criterio de éxito del implementer

- `dashboard.py` compila sin errores (`python -m compileall`)
- En modo paper: se muestra scheduler status + heartbeat grid + selector
- En modo backtest: se muestra BacktestRun grid + selector + métricas por run
- No hay cambios en otros archivos del repo

---

## Riesgos

- **R1 (bajo):** Si no hay heartbeats en DB, el modo paper muestra estado vacío — aceptable, se maneja con `st.info()`
- **R2 (bajo):** Si no hay `BacktestRun` registrados, el modo backtest muestra grilla vacía — aceptable, mismo patrón que `backtesting.py`
- **R3 (medio):** El `_calculate_status` actual en `dashboard.py` difiere del de `paper_trading.py`; el implementer debe usar la versión más completa para consistencia visual
