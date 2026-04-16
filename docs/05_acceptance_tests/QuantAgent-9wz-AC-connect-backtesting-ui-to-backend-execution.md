# Acceptance Criteria: Connect Backtesting UI to Backend Execution

**Issue ID**: QuantAgent-9wz  
**Type**: Feature - MVP Blocker  
**Priority**: 1

---

## Criterios de Aceptación

### AC1: APScheduler iniciado al app startup
```
Given la aplicación Streamlit se inicia
When llega a la función init
Then APScheduler se inicializa exitosamente
  And scheduler está en estado running
  And poller job está registrado con ID 'backtest_poller'
  And poller job tiene trigger interval de 10 segundos
  And no se lanza ninguna excepción
```

### AC2: Poller job detecta runs pendientes
```
Given un BacktestRun existe en DB con status=pending
When el poller job ejecuta (cada 10s)
Then detecta el run pendiente
  And trigger execution job con job_id = f"backtest_run_{run.id}"
  And almacena job_id en run.apscheduler_job_id
  And no crea duplicate jobs (idempotente)
```

### AC3: Ejecución automática via APScheduler
```
Given el poller job detectó un run pendiente
  And trigger execution job
When el execution job comienza
Then run.status cambia a 'running'
  And run.progress_pct = 0.0
  And ejecución comienza sin intervención manual
  And APScheduler muestra el job en get_jobs()
```

### AC4: Status transitions correctas
```
Given un backtest run en ejecución via APScheduler
When la ejecución progresa normalmente
Then status transitions: pending → running → completed
  And completed solo se alcanza cuando métricas están pobladas
  And timestamps (created_at, completed_at) están correctos
  And APScheduler job se auto-remueve al completar
```

### AC5: Progress tracking en tiempo real
```
Given un backtest run en status=running
When el usuario ve la tabla de runs en el UI
Then ve progreso actualizado:
  - progress_pct entre 0-100
  - processed_candles / total_candles
  - ETA en segundos (o "Calculating...")
  And progreso se actualiza cada ~5 segundos (autorefresh)
  And APScheduler job_id visible en run details (debugging)
```

### AC6: Métricas pobladas al completar
```
Given un backtest run completado exitosamente via APScheduler
When status=completed
Then las siguientes métricas están pobladas:
  - win_rate (porcentaje de trades ganadores)
  - profit_factor (gross profit / gross loss)
  - sharpe_ratio (ratio de Sharpe)
  - max_drawdown (máximo drawdown porcentual)
  - total_pnl (P&L total en USD o unidad base)
  - total_trades (número de trades ejecutados)
  And métricas son consistentes (no None, no NaN)
  And APScheduler job ya no existe en scheduler.get_jobs()
```

### AC7: Cancelación mid-execution via APScheduler
```
Given un backtest run en status=running
  And el usuario ve el run detail panel
When hace click en botón "Cancel"
Then cancel_requested flag se setea en DB
  And UI llama scheduler.remove_job(f"backtest_run_{run.id}")
  And execution job detecta flag en próximo checkpoint
  And execution job termina gracefully
  And status cambia a cancelled
  And progress_pct queda en valor actual (no resetea a 0)
  And botón Cancel se deshabilita después de click
  And APScheduler job es removido exitosamente
```

### AC8: Error handling - falla durante ejecución
```
Given un backtest run en ejecución via APScheduler
When ocurre una excepción (e.g., data fetch error, invalid config)
Then execution job captura excepción sin crash
  And status cambia a failed
  And error_message contiene descripción del error
  And traceback completo se loguea
  And UI muestra error_message en run details
  And APScheduler job termina (no retry)
```

### AC9: Stale run recovery al startup
```
Given la aplicación se reinicia (crash o restart)
  And hay BacktestRuns con status=running en DB
When APScheduler se inicializa al app startup (via init_scheduler)
Then recover_stale_runs() ejecuta antes de scheduler.start()
  And detecta runs con status=running (stale)
  And marca todos como status=failed
  And error_message = "Scheduler crashed or restarted"
  And intenta remover jobs residuales en scheduler (best effort)
  And poller job NO re-triggerea estos runs (porque status ya no es pending)
```

### AC10: UI polling con autorefresh
```
Given el usuario está en el tab Backtesting
When hay runs en ejecución
Then UI usa st.autorefresh con interval de 5 segundos
  And tabla de runs se actualiza mostrando progress actual
  And no hay flicker excesivo (smooth updates)
  And autorefresh solo ocurre en tab activo
```

### AC11: Run details panel (expandable)
```
Given un run en la tabla de runs
When el usuario expande run details (si implementado con expander)
Then ve:
  - Progress bar visual (st.progress)
  - Texto: "Processed X / Y candles"
  - ETA: "Estimated time remaining: Xs" o "Calculating..."
  - Status badge (color-coded)
  - APScheduler Job ID (para debugging): run.apscheduler_job_id
  - Botón Cancel (solo si status=running)
  - Error message (solo si status=failed)
  - Logs (últimas N líneas si disponibles, o placeholder MVP)
```

### AC12: Single backtest execution (MVP constraint)
```
Given un backtest run está en ejecución
When el usuario crea un segundo backtest run
Then segundo run entra en pending
  And poller detecta segundo run
  And poller triggerea execution job con max_instances=1
  And APScheduler encola el job (no ejecuta concurrentemente)
  And segundo run ejecuta solo después de que primer run complete/falle/cancele
  And no hay ejecución concurrente (ThreadPoolExecutor max_workers=1)
```

### AC13: Poller job es robusto
```
Given el poller job está corriendo
When detecta error en un run (e.g., DB query falla)
Then poller continúa ejecutando cada 10s (no crash)
  And error se loguea
  And próxima ejecución del poller sigue intentando
```

---

## Criterios de Regresión

### REG1: Crear backtest sin backend funciona
```
Given funcionalidad pre-existente de crear runs
When usuario crea backtest (form submit)
Then BacktestRun se crea en DB correctamente
  And todos los campos existentes (assets, timeframe, dates, config_snapshot) persisten
  And no hay breaking changes en schema
  And poller detecta el run (no manual trigger necesario)
```

### REG2: Listar runs existentes no cambia
```
Given runs existentes en DB (creados antes del feature)
When usuario ve tabla de runs
Then runs se listan correctamente
  And ordenamiento (por created_at desc) funciona
  And paginación/limit de 50 runs funciona
  And runs old tienen status inferido (completed si tienen metrics, pending otherwise)
```

### REG3: Profile universe selection funciona
```
Given usuario selecciona portfolio profile en form
  And deja campo "Assets" vacío
When crea backtest
Then sistema carga Universe desde profile
  And backtest se ejecuta con esos assets via APScheduler
  And no hay regresión en lógica existente
```

### REG4: APScheduler no interfiere con Streamlit hot-reloads
```
Given código Streamlit se modifica (hot-reload triggered)
  And hay un run en ejecución
When Streamlit reloads
Then APScheduler persiste (daemon mode)
  And ejecución continúa sin interrupción
  And UI reconecta y muestra progreso actual
  And @st.cache_resource evita re-init del scheduler
```

---

## Invariantes

### INV1: Status is single source of truth
- Un run **no** puede estar en status=completed sin métricas pobladas
- Un run **no** puede estar en status=running si APScheduler job no existe o no está activo
- Status transitions son unidireccionales: no volver de completed/failed/cancelled a pending

### INV2: APScheduler job lifecycle
- Job ID format: `backtest_run_{run.id}` (consistente)
- Job existe solo mientras status=running
- Job se auto-remueve al completar/fallar
- Job removal explícito al cancelar (best effort)

### INV3: Progress consistency
- `processed_candles <= total_candles` siempre
- `progress_pct` derivado de processed/total debe ser consistente
- `progress_pct=100` implica `status=completed` (o próximo a completar)

### INV4: Poller idempotencia
- Poller no crea duplicate jobs para mismo run_id
- Poller verifica existencia de job antes de triggerar: `scheduler.get_job(job_id)`

### INV5: Cancellation is graceful
- Cancel **no** deja state corrupto en DB
- Cancel **no** causa crash del execution job
- Cancel **puede** dejar análisis parciales en DB (es aceptable MVP)

### INV6: Error messages are actionable
- Error messages incluyen contexto útil (qué falló, dónde)
- No exponen secretos (API keys, passwords)
- Logs tienen traceback completo para debugging

---

## Oráculos de Validación

### Validación de migration
```bash
# Verificar que migration se aplicó
alembic current
alembic history | grep "add_backtest_progress_and_scheduler_fields"

# Verificar schema en DB
psql -d quantagent_dev -c "\d backtest_runs" | grep status
psql -d quantagent_dev -c "\d backtest_runs" | grep apscheduler_job_id
```

### Validación de APScheduler startup
```python
# En Python REPL
from quantagent.scheduler import get_scheduler

scheduler = get_scheduler()
assert scheduler.running == True

# Verificar poller job está registrado
jobs = scheduler.get_jobs()
poller_job = [j for j in jobs if j.id == 'backtest_poller']
assert len(poller_job) == 1
assert poller_job[0].trigger.interval.total_seconds() == 10
```

### Validación de ejecución E2E
```python
# Test manual en Streamlit
1. Abrir tab Backtesting
2. Crear run: assets=["AAPL"], timeframe="1h", dates=last 7 days
3. Esperar poller cycle (max 10s)
4. Verificar status → running en <15s (10s poller + ejecución start)
5. Verificar progreso actualiza cada ~5s
6. Esperar completion
7. Verificar status=completed y métricas != None
8. Verificar APScheduler job ya no existe: scheduler.get_job(f"backtest_run_{run_id}") == None
```

### Validación de cancelación
```python
# Test manual
1. Crear backtest largo (30+ días, múltiples assets)
2. Esperar hasta progress_pct ~ 30%
3. Click botón Cancel
4. Verificar cancel_requested=True en DB
5. Verificar scheduler.get_job(job_id) retorna None (job removido)
6. Verificar status → cancelled en <30s
7. Verificar progress_pct no resetea a 0
```

### Validación de stale run recovery
```bash
# Test manual
1. Crear run, dejar en running
2. Forzar crash (kill -9 streamlit process)
3. Restart app
4. Verificar en logs: "Recovered X stale runs"
5. Verificar run status=failed
6. Verificar error_message contiene "Scheduler crashed or restarted"
```

---

## Datos de Prueba

### Backtest simple (validación rápida)
```json
{
  "assets": ["AAPL"],
  "timeframe": "1h",
  "start_date": "2024-01-01",
  "end_date": "2024-01-02",
  "model_preset": "default",
  "profile": null,
  "mode": "Generate + Execute"
}
```
**Expected**: Poller detecta en <10s, completa en <1 min

### Backtest mediano (progress tracking)
```json
{
  "assets": ["AAPL", "MSFT", "GOOGL"],
  "timeframe": "1h",
  "start_date": "2024-01-01",
  "end_date": "2024-01-15",
  "model_preset": "default",
  "profile": "my_portfolio",
  "mode": "Generate + Execute"
}
```
**Expected**: Completa en 3-5 min, progreso visible

### Backtest largo (cancelación)
```json
{
  "assets": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
  "timeframe": "5m",
  "start_date": "2024-01-01",
  "end_date": "2024-03-01",
  "model_preset": "default",
  "profile": null,
  "mode": "Generate + Execute"
}
```
**Expected**: Toma >10 min, permite testar cancelación

### Backtest con error (data invalida)
```json
{
  "assets": ["INVALID_SYMBOL_XYZ"],
  "timeframe": "1h",
  "start_date": "2024-01-01",
  "end_date": "2024-01-02",
  "model_preset": "default",
  "profile": null,
  "mode": "Generate + Execute"
}
```
**Expected**: status=failed, error_message con detalle

---

## Edge Cases a Verificar

### Edge 1: Backtest con 0 datos
```
Given assets + date range que no tienen datos en DB
When backtest ejecuta via APScheduler
Then status=failed
  And error_message indica "No data available for range"
  And APScheduler job termina (no retry)
```

### Edge 2: Múltiples runs encolados
```
Given 3 runs creados en rápida sucesión (todos pending)
When poller procesa (detecta los 3)
Then poller triggerea 3 execution jobs
  And APScheduler encola con max_instances=1
  And ejecuta secuencialmente: run1 completa → run2 start → run2 completa → run3 start
  And solo 1 run tiene status=running a la vez
```

### Edge 3: Streamlit hot-reload durante ejecución
```
Given run en ejecución (status=running) via APScheduler
When código Streamlit se modifica (hot-reload triggered)
Then APScheduler persiste (daemon mode)
  And ejecución continúa sin interrupción
  And @st.cache_resource evita re-init
  And UI reconecta a mismo scheduler y muestra progreso actual
```

### Edge 4: Cancel después de completion
```
Given run con status=completed
When usuario intenta cancelar (botón no debería estar visible)
Then botón Cancel no está visible/enabled en UI
  And scheduler.get_job(job_id) retorna None (job ya removido)
```

### Edge 5: Run sin portfolio profile y sin assets
```
Given usuario crea run sin assets
  And sin portfolio profile seleccionado
When intenta submit form
Then UI muestra error: "Provide assets or select a portfolio profile with a Universe"
  And BacktestRun **no** se crea en DB
  And poller no detecta nada (no run creado)
```

### Edge 6: Poller error (DB query falla)
```
Given poller job está corriendo
When DB connection temporalmente falla
Then poller captura excepción
  And loguea error
  And no crash del scheduler
  And próxima ejecución (10s después) reintenta
```

### Edge 7: Job ID colisión (improbable pero posible)
```
Given run_id=123, job_id="backtest_run_123"
  And job ya existe (edge case: stale job no removido)
When poller triggerea job para run 123
Then poller usa replace_existing=True
  And scheduler reemplaza job existente
  And ejecución procede normalmente
```

---

## Métricas de Éxito (Post-Implementation)

- **Funcionalidad**: 100% de backtests creados se ejecutan automáticamente via APScheduler
- **Observabilidad**: Progress visible en <15s después de crear run (10s poller + 5s UI refresh)
- **Reliability**: <5% runs quedan stuck en running (debido a crashes, mitigado por recovery)
- **Cancelación**: Cancel request procesa en <30s (job removal + cooperative flag)
- **Performance**: UI autorefresh no causa lag perceptible (<500ms refresh time)
- **Scheduler overhead**: Poller job consume <5% CPU en idle (cada 10s query liviano)

---

## APScheduler-Specific Validations

### APSCH1: Job registrado correctamente
```python
from quantagent.scheduler import get_scheduler

scheduler = get_scheduler()
job = scheduler.get_job('backtest_poller')

assert job is not None
assert job.func.__name__ == 'poll_pending_backtest_runs'
assert job.trigger.interval.total_seconds() == 10
```

### APSCH2: Execution job tiene max_instances=1
```python
# Durante ejecución, verificar que solo 1 job "backtest_run_*" está running
jobs = scheduler.get_jobs()
running_backtest_jobs = [j for j in jobs if j.id.startswith('backtest_run_')]

# Solo debería haber 1 (o 0 si ninguno corriendo)
assert len(running_backtest_jobs) <= 1
```

### APSCH3: Job removal funciona
```python
run_id = 123
job_id = f"backtest_run_{run_id}"

# Verificar job existe antes de cancelar
assert scheduler.get_job(job_id) is not None

# Simular cancelación
scheduler.remove_job(job_id)

# Verificar job removido
assert scheduler.get_job(job_id) is None
```

### APSCH4: Scheduler persiste a través de Streamlit reloads
```python
# Test manual
1. Crear run, verificar job existe
2. Modificar código Streamlit (hot-reload)
3. Verificar scheduler.running == True (no restarted)
4. Verificar job sigue existiendo
5. Ejecución continúa sin interrupción
```

---

## Criterios de Aceptación NO Incluidos (Out of Scope)

- Replay execution (feature separada, Tab 5)
- Concurrent execution de múltiples backtests (MVP es secuencial)
- Distributed execution (multi-machine)
- Advanced scheduling (cron-like triggers para backtests)
- SQLAlchemy job store persistence (MVP usa in-memory)
- Artifact management changes (path-only ya decidido)
- Live streaming de logs en UI (MVP muestra últimas N líneas o placeholder)
- Notification system (email/Slack al completar backtest)
- Job priority queue (FIFO simple por ahora)

---

## Definition of Done (Completo)

- [ ] Migration aplicada y validada
- [ ] Model BacktestRun actualizado con campos scheduler
- [ ] APScheduler setup implementado (`scheduler.py`)
- [ ] Poller job corriendo cada 10s
- [ ] Execution jobs triggered por poller
- [ ] Status transitions correctas
- [ ] Progress tracking funcionando
- [ ] Métricas pobladas al completar
- [ ] Cancelación funciona (flag + job removal)
- [ ] UI polling con autorefresh
- [ ] Stale run recovery al startup
- [ ] Logging comprehensivo
- [ ] Tests E2E pasando
- [ ] No regresión en features existentes
- [ ] APScheduler no interfiere con Streamlit hot-reloads
- [ ] Ready for merge/integration con QuantAgent-3o4 TradingScheduler (si existe)
