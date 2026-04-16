# Planning: Connect Backtesting UI to Backend Execution

**Issue ID**: QuantAgent-9wz  
**Type**: Feature - MVP Blocker  
**Priority**: 1  
**Estimated Effort**: 8–12 hours

---

## Tareas

### 1. Crear migration para campos de progreso y scheduler tracking en BacktestRun
**Estimado**: 1 hora  
**Dependencias**: Ninguna

- Crear migration Alembic con campos:
  - `status` ENUM (pending, running, completed, failed, cancelled)
  - `progress_pct` FLOAT
  - `processed_candles`, `total_candles` INT (nullable)
  - `eta_seconds` INT (nullable)
  - `cancel_requested` BOOLEAN
  - `error_message` TEXT (nullable)
  - `apscheduler_job_id` TEXT (nullable, para tracking)
- Migrar runs existentes: status='completed' si tienen metrics, else 'pending'
- Ejecutar migration en dev DB
- Validar campos están disponibles

**Entregables**:
- `alembic/versions/xxx_add_backtest_progress_and_scheduler_fields.py`

---

### 2. Actualizar BacktestRun model
**Estimado**: 30 min  
**Dependencias**: Tarea 1

- Agregar campos a `quantagent/models.py` (o schema.py)
- Definir ENUM para status
- Agregar defaults apropiados
- Agregar campo `apscheduler_job_id` para tracking
- Actualizar docstrings

**Entregables**:
- `quantagent/models.py` (o archivo de models actualizado)

---

### 3. Implementar APScheduler setup y singleton
**Estimado**: 1.5 horas  
**Dependencias**: Ninguna (puede ir en paralelo con Tarea 2)

- Crear `quantagent/scheduler.py`
- Implementar:
  - `get_scheduler()`: singleton que retorna APScheduler instance
  - `init_scheduler()`: configura y arranca scheduler
  - `shutdown_scheduler()`: graceful shutdown
  - `recover_stale_runs()`: detecta runs stuck en running, marca como failed
- Configuración:
  - BackgroundScheduler con ThreadPoolExecutor (max_workers=1)
  - In-memory job store (MVP)
  - Timezone UTC
- Agregar logging apropiado

**Entregables**:
- `quantagent/scheduler.py`

---

### 4. Implementar APScheduler jobs (poller + execution)
**Estimado**: 3 horas  
**Dependencias**: Tarea 2, 3

- Crear `quantagent/backtest_jobs.py`
- Implementar:
  - `poll_pending_backtest_runs()`: IntervalTrigger job (cada 10s)
    - Query DB: `BacktestRun.filter_by(status='pending')`
    - Para cada run: trigger execution job si no existe
    - Store job_id en DB
  - `execute_backtest_run(run_id)`: Date trigger job (run once)
    - Update status=running
    - Call `_execute_backtest_logic(run_id)`
    - Handle exceptions → status=failed
    - Update status=completed y métricas al finalizar
  - `_execute_backtest_logic(run_id)`: core execution wrapper
    - Load run config
    - Setup progress_callback y cancel_check
    - Call backtesting engine
    - Populate metrics
- Exception handling + logging
- Job ID format: `f"backtest_run_{run_id}"`

**Entregables**:
- `quantagent/backtest_jobs.py`

---

### 5. Integrar con Backtesting Engine existente
**Estimado**: 2 horas  
**Dependencias**: Tarea 4

- Identificar módulo de backtesting engine actual (si existe)
- Agregar callback interface:
  - `progress_callback(processed, total)` → update DB
  - `cancel_check()` → return `run.cancel_requested`
- Implementar wrapper en `backtest_jobs._execute_backtest_logic()`:
  - Cargar run de DB
  - Preparar config (assets, timeframe, dates)
  - Llamar engine con callbacks
  - Capturar resultado (metrics)
  - Actualizar run con metrics
- Error handling: catch + log + status=failed
- Validar que cancel_check se llama cada N candles

**Entregables**:
- Modificaciones a engine (si necesario)
- Integración completa en `backtest_jobs.py`

---

### 6. Inicializar APScheduler en Streamlit app startup
**Estimado**: 45 min  
**Dependencias**: Tarea 3, 4

- En `apps/streamlit/app.py`:
  - Agregar función init con `@st.cache_resource`:
    ```python
    @st.cache_resource
    def initialize_app():
        from quantagent.scheduler import init_scheduler
        init_scheduler()
        return True
    ```
  - Llamar `initialize_app()` al principio de main
- Registrar poller job:
  - En `init_scheduler()`, agregar poller job:
    ```python
    scheduler.add_job(
        poll_pending_backtest_runs,
        trigger='interval',
        seconds=10,
        id='backtest_poller',
        replace_existing=True
    )
    ```
- Verificar que scheduler persiste entre Streamlit hot-reloads

**Entregables**:
- Scheduler iniciado en app startup con poller job

---

### 7. Actualizar Streamlit view para trigger ejecución (via poller)
**Estimado**: 30 min  
**Dependencias**: Tarea 6

- En `apps/streamlit/views/backtesting.py`:
  - Después de crear BacktestRun en DB, **no** llamar trigger manual
  - Confiar en poller job (cada 10s) para detectar y ejecutar
  - Actualizar mensaje success: "Run X created. Execution will start shortly (polled every 10s)."
- Remover caption "backend execution wiring pending"
- Opcional: agregar botón "Force Trigger" que llame directamente a `scheduler.add_job(execute_backtest_run, args=[run.id])` (debug only)

**Entregables**:
- `apps/streamlit/views/backtesting.py` (actualizado)

---

### 8. Agregar UI de progreso en run details
**Estimado**: 2 horas  
**Dependencias**: Tarea 7

- En `backtesting.py`, agregar sección "Run Details" (expandable):
  - Progress bar (st.progress con progress_pct / 100)
  - Texto: "Processed X / Y candles"
  - ETA: "Estimated time remaining: Xs" o "Calculating..."
  - Status badge (color-coded: running=🔵, completed=✅, failed=❌, cancelled=⚪)
  - APScheduler job ID (si disponible, para debugging)
  - Error message (solo si status=failed)
  - Cancel button (solo visible si status=running):
    - Click → set `cancel_requested=True` en DB
    - Llamar `scheduler.remove_job(job_id)` (best effort)
    - Deshabilitar botón después de click
- Polling: ya existe `st.autorefresh(5000)` → verificar que funciona

**Entregables**:
- UI de progreso en `apps/streamlit/views/backtesting.py`

---

### 9. Implementar lógica de cancelación
**Estimado**: 1 hora  
**Dependencias**: Tareas 5, 8

- En `backtest_jobs._execute_backtest_logic()`:
  - `cancel_check()` verifica `run.cancel_requested` cada N candles
  - Si True, raise `CancelledException()`
  - Catch en `execute_backtest_run()` → update status=cancelled
- En UI, botón Cancel:
  - Set flag en DB
  - Remover job: `scheduler.remove_job(f"backtest_run_{run.id}")`
  - Mostrar "Cancellation requested..." mientras status=running
- Testing manual: crear run, cancelar mid-execution
- Verificar que poller no re-triggerea el run (porque status ya no es pending)

**Entregables**:
- Cancelación funcionando E2E

---

### 10. Agregar logging apropiado
**Estimado**: 1 hora  
**Dependencias**: Tareas 4, 5

- En `backtest_jobs.py`:
  - Log poller: "Poller detected X pending runs"
  - Log execution start: "Starting backtest run X (assets=Y, timeframe=Z)"
  - Log progress: cada 25% avance
  - Log completion: "Backtest X completed in Ys (metrics: ...)"
  - Log errors: full traceback
  - Log cancellation: "Backtest X cancelled by user"
- En `scheduler.py`:
  - Log scheduler init: "APScheduler started"
  - Log recovery: "Recovered X stale runs"
  - Log shutdown: "APScheduler stopped"
- Usar structured logging (si ya está configurado en proyecto)

**Entregables**:
- Logging comprehensivo en todos los componentes

---

### 11. Testing E2E y validación
**Estimado**: 2 horas  
**Dependencias**: Todas las anteriores

- Tests manuales:
  - Crear backtest simple (1 asset, 1 día) → verify poller detecta y ejecuta
  - Crear backtest largo → verify progress updates en UI
  - Cancelar run mid-execution → verify job removal + status=cancelled
  - Simular error (config inválido) → verify status=failed + error_message
  - Restart app con run en ejecución → verify recovery (status=failed)
  - Crear múltiples runs rápidamente → verify ejecución secuencial
- Verificar no-regresión:
  - Listar runs existentes funciona
  - Crear run sin Universe (usando profile) funciona
- Opcional: Unit tests básicos para poller y execution jobs

**Entregables**:
- Validación manual completa
- Fix de bugs encontrados
- Documentación de edge cases encontrados

---

## Dependencias Externas

- **APScheduler**: >= 3.10.0 (agregar a requirements.txt si no existe)
- **Backtesting Engine**: Debe existir o implementarse (fuera de scope si no existe → simplificar a mock para MVP)
- **Database**: Postgres con SQLAlchemy ya configurado
- **Streamlit**: >= 1.28 (para st.autorefresh y st.cache_resource)

---

## Orden de Ejecución Recomendado

**Path 1 (Core)**: 1 → 2 → 3 → 4 → 5 → 6 (APScheduler setup + jobs + engine integration)  
**Path 2 (UI)**: 7 → 8 → 9 (UI trigger + progress + cancelación)  
**Path 3 (Polish)**: 10 → 11 (logging + testing)

**Parallel work possible**:
- Tareas 1-2 (DB) pueden ir en paralelo con Tarea 3 (scheduler setup)
- Tarea 10 (logging) puede ir en paralelo con Tareas 7-9 (UI)

---

## Riesgos

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| APScheduler no familiar al equipo | Media | Medio | Docs claras; config simple MVP |
| Backtesting engine no existe aún | Alta | Alto | Implementar mock engine sintético (Plan B) |
| Poller delay (10s) percibido como lento | Baja | Bajo | Documentar; ajustar a 5s si necesario |
| Job ID colisiones con 3o4 futuro | Media | Alto | Prefijos claros (`backtest_run_*`) |
| APScheduler lifecycle en Streamlit | Media | Alto | `@st.cache_resource` + testing exhaustivo |
| Scheduler crash deja jobs huérfanos | Media | Medio | Recovery logic al startup |

---

## Estrategia de Testing

### Unit Testing
- Mock de BacktestRun model + APScheduler
- Test poller job detects pending runs
- Test execution job status transitions
- Test progress calculation
- Test cancel logic (flag + job removal)

### Integration Testing
- E2E: crear run → poller detecta → ejecuta → completa
- E2E: cancelación mid-execution (job removal + flag)
- E2E: error handling (status=failed)
- E2E: scheduler restart → stale runs recovery

### Manual Testing (obligatorio)
- Flujo completo en UI local
- Verificar autorefresh funciona
- Probar edge cases (sin datos, config inválido, múltiples runs)
- Verificar poller no consume recursos excesivos

---

## Rollout

1. **Branch**: `feature/QuantAgent-9wz-apscheduler-backtest-execution`
2. **Commits incrementales**:
   - Migration + model update
   - APScheduler setup
   - Jobs implementation (poller + execution)
   - Engine integration
   - UI updates (trigger + progress + cancel)
   - Logging
   - Testing + fixes
3. **Testing local**: Validación manual completa
4. **PR Review**: Code review + demo en PR description (mostrar progreso real-time)
5. **Merge**: Después de aprobación
6. **Documentación**: Actualizar `docs/03_design/streamlit_app_architecture.md` con APScheduler details

---

## Checkpoints

### Checkpoint 1 (3 horas)
- [ ] Migration creada y ejecutada
- [ ] Model actualizado con campos de progreso + apscheduler_job_id
- [ ] APScheduler setup implementado (`scheduler.py`)
- [ ] Poller job skeleton implementado (sin engine integration)

### Checkpoint 2 (6 horas)
- [ ] Execution job implementado (mock backtest si engine no existe)
- [ ] Status transitions funcionan (pending → running → completed)
- [ ] Poller detecta pending runs y triggerea execution jobs
- [ ] Scheduler inicializado en Streamlit app

### Checkpoint 3 (9 horas)
- [ ] Engine integration completa (o mock si no existe)
- [ ] Progress tracking funcionando
- [ ] UI muestra progreso en tiempo real
- [ ] Cancelación implementada (flag + job removal)
- [ ] Stale run recovery implementado

### Checkpoint 4 (12 horas)
- [ ] Logging completo
- [ ] Testing E2E pasando
- [ ] No regresión en features existentes
- [ ] Ready for PR

---

## Notas de Implementación

### Si backtesting engine no existe
**Plan B**: Implementar mock engine para MVP:
```python
# quantagent/backtesting/mock_engine.py
def run_backtest_mock(
    assets, timeframe, start_date, end_date, config_snapshot,
    progress_callback=None, cancel_check=None
) -> BacktestResult:
    """Mock engine que genera métricas sintéticas."""
    import time
    import random
    
    total_candles = 100  # Simulado
    
    for i in range(total_candles):
        time.sleep(0.1)  # simular trabajo
        
        if progress_callback:
            progress_callback(i+1, total_candles)
        
        if cancel_check and cancel_check():
            raise CancelledException("Backtest cancelled")
    
    return BacktestResult(
        win_rate=random.uniform(0.45, 0.65),
        profit_factor=random.uniform(1.1, 2.5),
        sharpe_ratio=random.uniform(0.5, 2.0),
        max_drawdown=random.uniform(-0.15, -0.05),
        total_pnl=random.uniform(-1000, 5000),
        total_trades=random.randint(50, 200),
    )
```

### Configuración APScheduler en requirements.txt
```
APScheduler>=3.10.0
```

### Si QuantAgent-3o4 ya existe
- Importar scheduler existente: `from quantagent.trading_scheduler import get_scheduler`
- Agregar backtest jobs al scheduler existente
- Asegurar job ID conventions (prefijos distintos)

### Priorización si tiempo limitado
1. **Must-have**: Tasks 1-7 (ejecución básica via APScheduler)
2. **Should-have**: Tasks 8-9 (progreso + cancelación)
3. **Nice-to-have**: Tasks 10-11 (logging + tests comprehensivos)

Si tiempo es crítico: implementar Must-have primero, iterar Should-have después.

---

## APScheduler Reference (Quick)

```python
# Agregar job interval
scheduler.add_job(
    func=poll_pending_backtest_runs,
    trigger='interval',
    seconds=10,
    id='backtest_poller',
    replace_existing=True
)

# Agregar job date (run once)
scheduler.add_job(
    func=execute_backtest_run,
    trigger='date',  # Immediate
    args=[run_id],
    id=f'backtest_run_{run_id}',
    max_instances=1
)

# Remover job
scheduler.remove_job(job_id)

# Check if job exists
if scheduler.get_job(job_id):
    pass
```

---

## Definition of Done (Recap)

- [ ] APScheduler configurado y corriendo
- [ ] Poller job detecta pending runs cada 10s
- [ ] Execution jobs corriendo via APScheduler
- [ ] Status transitions correctas
- [ ] Progress tracking funcionando
- [ ] Métricas pobladas al completar
- [ ] Cancelación funciona (flag + job removal)
- [ ] UI polling con autorefresh
- [ ] Stale run recovery al startup
- [ ] No regresión en features existentes
- [ ] Logs apropiados
- [ ] Ready for merge con 3o4 (si existe)
