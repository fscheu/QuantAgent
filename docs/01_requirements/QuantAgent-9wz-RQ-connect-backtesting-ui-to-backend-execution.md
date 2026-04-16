# Requirements: Connect Backtesting UI to Backend Execution

**Issue ID**: QuantAgent-9wz  
**Type**: Feature - MVP Blocker  
**Priority**: 1  
**Labels**: backtesting, mvp-blocker, streamlit

---

## Objetivo

Conectar el formulario de creación de backtests en Streamlit con el motor de ejecución backend para que los runs se ejecuten realmente en lugar de quedarse en estado "pending", utilizando **APScheduler** como mecanismo estándar de ejecución en background (alineado con decisión arquitectural de QuantAgent-3o4 TradingScheduler).

---

## Contexto

### Estado Actual
- El tab Backtesting permite crear registros `BacktestRun` en la base de datos
- No hay mecanismo de ejecución; runs permanecen en status `pending` indefinidamente
- El usuario ve mensaje: "Run X created. Backend execution wiring pending."

### Arquitectura Objetivo
- **APScheduler** como mecanismo estándar de background execution (per QuantAgent-3o4)
- TradingScheduler ya establece APScheduler como infraestructura base para scheduled/background tasks
- QuantAgent-9wz extiende este mecanismo para backtest execution

---

## Alcance

### Incluye
- Uso de APScheduler para ejecutar backtest runs en background
- Job polling: scheduler detecta runs pendientes y ejecuta automáticamente
- Actualización de status en tiempo real: `pending → running → completed/failed`
- Progress tracking: candles procesadas, total, ETA
- Población de métricas finales (win_rate, profit_factor, sharpe, max_dd, total_pnl)
- Cancelación via APScheduler job removal + flag cooperativo
- Polling UI con st.autorefresh para mostrar progreso en tiempo real

### No Incluye
- Replay execution (es feature separada, issue futuro)
- Replay scenario sweeps (feature del Tab 5, fuera de scope)
- Advanced scheduling patterns (cron triggers; MVP usa on-demand + polling)
- Distributed execution (single-machine por ahora)
- Artifact management changes (path-only policy ya implementado)

---

## Constraints

- **APScheduler infraestructura**: Asume que QuantAgent-3o4 provee scheduler base (o se implementa mínimamente aquí)
- **Single-threaded execution**: Un backtest a la vez por simplicidad MVP
- **Streamlit constraints**: No long-running requests; backend ejecuta async, UI polling
- **Database**: BacktestRun model ya existe; agregar campos de progreso si faltan
- **Job store**: In-memory job store para MVP (SQLAlchemy job store opcional futuro)
- **Cancellation**: Debe ser graceful (no dejar state corrupto)

---

## Casos de Uso

### UC1: Usuario crea backtest y ve ejecución
1. Usuario llena form (assets, timeframe, dates, model preset, profile)
2. Click "Create run"
3. Sistema crea BacktestRun en DB con status=pending
4. APScheduler poller job detecta run pendiente (polling cada 10s)
5. Scheduler dispara job de ejecución para ese run_id
6. Status → running; UI muestra progreso (X/Y candles, ETA)
7. Ejecución completa → status=completed, métricas pobladas
8. UI muestra métricas finales en tabla

### UC2: Usuario cancela backtest en ejecución
1. Backtest está en status=running
2. Usuario ve botón "Cancel" en run details
3. Click → sistema marca run para cancelación + remueve job de scheduler
4. Worker detecta cancelación flag y termina gracefully
5. Status → cancelled, progreso queda en punto actual
6. UI refleja status cancelled

### UC3: Backtest falla por error
1. Durante ejecución, ocurre excepción (e.g., data fetch error)
2. Job captura error, registra en logs
3. Status → failed, error_message poblado
4. Job termina, APScheduler no reintenta
5. UI muestra error en run details

---

## Decisiones de Alto Nivel

### Mecanismo de Ejecución
**Opción 1**: Thread pool + queue (approach anterior)  
**Opción 2**: APScheduler con polling + on-demand jobs ✅  
**Opción 3**: Celery + Redis (overkill para MVP)

**Decisión**: APScheduler (Opción 2) — alineado con QuantAgent-3o4 TradingScheduler

**Razón**:
- Consistencia arquitectural: TradingScheduler ya usa APScheduler
- Unified execution model: backtests + paper trading + scheduled tasks comparten infraestructura
- Persistence & recovery: APScheduler con SQLAlchemy job store permite recovery post-crash (futuro)
- Simplifica codebase: no múltiples execution mechanisms

### APScheduler Pattern
**Pattern elegido**: Poller job + on-demand execution jobs

**Implementación**:
- **Poller job**: Interval job (cada 10s) que query DB por runs con status=pending
- **Execution job**: Date job (run once) triggered por poller para cada run_id pendiente
- **Job ID format**: `backtest_run_{run_id}` para identificación/cancelación

### Progress Tracking
- `BacktestRun.status`: ENUM(pending, running, completed, failed, cancelled)
- `BacktestRun.progress_pct`: FLOAT (0-100)
- `BacktestRun.processed_candles`: INT
- `BacktestRun.total_candles`: INT (estimado)
- `BacktestRun.eta_seconds`: INT (nullable, estimado)
- `BacktestRun.error_message`: TEXT (nullable)
- `BacktestRun.apscheduler_job_id`: TEXT (nullable, para tracking)

### Cancelación
- `BacktestRun.cancel_requested`: BOOLEAN (default False)
- User click Cancel → set flag + `scheduler.remove_job(job_id)`
- Execution job chequea flag cada N candles y termina gracefully si está seteado

---

## Dependencia: QuantAgent-3o4 TradingScheduler

### Asumido por 3o4
- APScheduler instance inicializado y accesible globalmente
- Scheduler running en background (daemon o managed process)
- Basic job management functions (add_job, remove_job, get_jobs)

### Necesario agregar en 9wz (si no existe en 3o4)
- Poller job para detectar pending backtest runs
- Execution job function para correr backtests
- Job store configuration (in-memory para MVP, SQLAlchemy opcional)

**Nota**: Si 3o4 no existe aún, QuantAgent-9wz implementa APScheduler setup mínimo como prerequisito.

---

## Invariantes

- Solo un backtest corriendo a la vez (MVP constraint)
- Status transitions: `pending → running → {completed|failed|cancelled}`
- Métricas solo se populan en status=completed
- Cancel solo disponible en status=running
- Progress updates cada ~10 candles o 5 segundos (lo que ocurra primero)
- APScheduler job_id es único por run (formato: `backtest_run_{run_id}`)

---

## Edge Cases

### Backtest sin assets ni profile universe
- Validación en UI (ya existe): error antes de crear run

### Backtest con date range sin datos
- Execution job maneja gracefully: status=failed, error_message="No data for range"

### Streamlit reload durante ejecución
- APScheduler persiste (no se pierde); UI continúa polling

### Usuario crea múltiples runs rápidamente
- Poller job detecta todos los pending runs
- Ejecuta secuencialmente (max_instances=1 en execution job)

### Scheduler crash durante ejecución
- Next app start: detectar runs con status=running (stale) y marcar como failed
- Si SQLAlchemy job store: APScheduler puede recuperar jobs (opcional futuro)

### APScheduler no iniciado
- Validación en app startup: error claro si scheduler no está disponible
- Fallback: UI muestra "Background execution unavailable" (MVP puede omitir)

---

## Definition of Done

- [ ] APScheduler configurado y corriendo al app startup
- [ ] Poller job detecta pending runs cada 10s
- [ ] Usuario crea backtest → ejecución se triggerea automáticamente via APScheduler
- [ ] Status transitions correctamente: pending → running → completed/failed
- [ ] Progress mostrado en UI (processed/total candles, ETA)
- [ ] Métricas pobladas al completar (win_rate, profit_factor, sharpe, max_dd, total_pnl)
- [ ] Botón Cancel funciona: remueve job + flag cooperativo termina ejecución gracefully
- [ ] UI usa st.autorefresh para polling en tiempo real
- [ ] Stale run recovery al startup (detectar runs stuck en running)
- [ ] No regresión en funcionalidad existente (crear runs, listar en tabla)
- [ ] Logs apropiados (start, progress, completion, errors)
- [ ] Compatible con future TradingScheduler usage (no conflictos de jobs)
