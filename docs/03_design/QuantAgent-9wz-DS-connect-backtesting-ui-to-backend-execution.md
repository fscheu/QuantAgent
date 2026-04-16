# Design: Connect Backtesting UI to Backend Execution

**Issue ID**: QuantAgent-9wz  
**Type**: Feature - MVP Blocker  
**Priority**: 1

---

## Componentes Afectados

- `quantagent/scheduler.py` — APScheduler setup y poller job (nuevo o extendido desde 3o4)
- `quantagent/backtest_jobs.py` — APScheduler job functions (nuevo)
- `apps/streamlit/views/backtesting.py` — UI de progreso y cancelación
- `quantagent/models.py` (o schema.py) — Agregar campos de progreso a BacktestRun
- `alembic/versions/` — Migration para campos nuevos de BacktestRun
- `apps/streamlit/app.py` — Inicializar APScheduler al startup

---

## Arquitectura de Alto Nivel

```
[Streamlit UI]
    ↓ (crear run)
[BacktestRun DB] ← (status=pending)
    ↓
[APScheduler Poller Job] ← (interval 10s, query pending runs)
    ↓ (detecta pending run_id)
[APScheduler Execution Job] ← (date job, run once per run_id)
    ↓ (ejecuta)
[Backtesting Engine]
    ↓ (updates)
[BacktestRun DB] ← (status=running/completed, progress, metrics)
    ↑
[Streamlit UI] ← (polling con st.autorefresh)
```

---

## Decisiones Técnicas

### 1. APScheduler como Execution Mechanism

**Decisión**: Usar APScheduler con pattern Poller + Execution Jobs  
**Razón**:
- Alineado con QuantAgent-3o4 TradingScheduler (unified architecture)
- Permite job persistence con SQLAlchemy job store (futuro)
- Better recovery después de crashes (jobs can be persisted)
- Unified logging and monitoring de todos los background tasks

**Alternativas consideradas**:
- Thread pool + queue → rechazada (no alineada con 3o4, duplica infraestructura)
- Celery → rechazada (overkill para MVP single-machine)

### 2. Job Pattern: Poller + Execution

**Pattern elegido**: Interval poller job + on-demand execution jobs

**Poller Job**:
- Tipo: IntervalTrigger (cada 10s)
- Function: `poll_pending_backtest_runs()`
- Query DB: `BacktestRun.query.filter_by(status='pending').all()`
- Para cada run: trigger execution job si no existe ya

**Execution Job**:
- Tipo: Date trigger (run once, immediate)
- Function: `execute_backtest_run(run_id)`
- Job ID: `f"backtest_run_{run_id}"`
- max_instances: 1 (solo un backtest a la vez)

**Razón**:
- Desacopla detección de ejecución (más flexible)
- Permite rate limiting fácil (max_instances)
- Job IDs predecibles facilitan cancelación

### 3. Progress Tracking Fields

**Decisión**: Agregar campos explícitos a BacktestRun model  
**Campos nuevos**:
- `status`: ENUM('pending', 'running', 'completed', 'failed', 'cancelled')
- `progress_pct`: FLOAT (0-100)
- `processed_candles`: INT (nullable)
- `total_candles`: INT (nullable, estimado)
- `eta_seconds`: INT (nullable)
- `cancel_requested`: BOOLEAN (default False)
- `error_message`: TEXT (nullable)
- `apscheduler_job_id`: TEXT (nullable, para tracking/debugging)

**Razón**: Mismo que approach anterior — explícito, facilita queries y UI

### 4. APScheduler Configuration

**Job Store**: In-memory por defecto (MVP)  
**Executor**: ThreadPoolExecutor con max_workers=1 (secuencial)  
**Timezone**: UTC (consistente con timestamps de DB)

**Config mínima**:
```python
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor

executors = {
    'default': ThreadPoolExecutor(max_workers=1)  # Solo 1 backtest a la vez
}

scheduler = BackgroundScheduler(executors=executors, timezone='UTC')
```

**Futuro (post-MVP)**: SQLAlchemyJobStore para persistence
```python
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

jobstores = {
    'default': SQLAlchemyJobStore(url='postgresql://...')
}
scheduler = BackgroundScheduler(jobstores=jobstores, executors=executors)
```

### 5. Cancelación Mechanism

**Decisión**: Hybrid cancellation — APScheduler job removal + cooperative flag  
**Flujo**:
1. User clicks Cancel → UI set `BacktestRun.cancel_requested = True`
2. UI también llama `scheduler.remove_job(f"backtest_run_{run_id}")`
3. Execution job chequea `cancel_requested` flag cada N candles
4. Si flag=True, job termina gracefully, status=cancelled

**Razón**:
- `remove_job()` evita re-scheduling si poller vuelve a detectar run (edge case)
- Cooperative flag permite graceful shutdown (no force kill)
- DB flag persiste a través de scheduler restarts

### 6. Stale Run Recovery

**Decisión**: Detectar al startup y marcar como failed  
**Implementación**:
```python
def recover_stale_runs():
    """Mark stale running runs as failed."""
    with SessionLocal() as s:
        stale = s.query(BacktestRun).filter_by(status='running').all()
        for run in stale:
            run.status = 'failed'
            run.error_message = 'Scheduler crashed or restarted'
            # También remover job residual si existe
            try:
                scheduler.remove_job(f"backtest_run_{run.id}")
            except:
                pass
        s.commit()
```

**Llamar en**: `init_scheduler()` antes de `scheduler.start()`

---

## Contratos

### BacktestRun Model (campos nuevos)

```python
class BacktestRun(Base):
    # ... existing fields ...
    
    status = Column(
        Enum('pending', 'running', 'completed', 'failed', 'cancelled', name='backtest_status'),
        default='pending',
        nullable=False
    )
    progress_pct = Column(Float, default=0.0)
    processed_candles = Column(Integer, nullable=True)
    total_candles = Column(Integer, nullable=True)
    eta_seconds = Column(Integer, nullable=True)
    cancel_requested = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)
    apscheduler_job_id = Column(Text, nullable=True)  # Nuevo: tracking
```

### APScheduler Jobs API

```python
# quantagent/backtest_jobs.py

def poll_pending_backtest_runs():
    """
    Poller job: Query DB for pending runs, trigger execution jobs.
    Runs every 10s via APScheduler IntervalTrigger.
    """
    with SessionLocal() as s:
        pending_runs = s.query(BacktestRun).filter_by(status='pending').all()
        
        for run in pending_runs:
            job_id = f"backtest_run_{run.id}"
            
            # Check if job already exists (avoid duplicates)
            if scheduler.get_job(job_id):
                continue
            
            # Trigger execution job
            scheduler.add_job(
                execute_backtest_run,
                trigger='date',  # Run once, immediately
                args=[run.id],
                id=job_id,
                max_instances=1,
                replace_existing=True
            )
            
            # Store job_id in DB for tracking
            run.apscheduler_job_id = job_id
            s.commit()

def execute_backtest_run(run_id: int):
    """
    Execution job: Run backtest for given run_id.
    Triggered by poller, runs once per run.
    """
    try:
        with SessionLocal() as s:
            run = s.query(BacktestRun).get(run_id)
            if not run:
                logger.error(f"BacktestRun {run_id} not found")
                return
            
            # Check if cancelled before starting
            if run.cancel_requested:
                run.status = 'cancelled'
                s.commit()
                return
            
            # Update status to running
            run.status = 'running'
            run.progress_pct = 0.0
            s.commit()
        
        # Execute backtest (details below)
        _execute_backtest_logic(run_id)
        
        # Update status to completed
        with SessionLocal() as s:
            run = s.query(BacktestRun).get(run_id)
            run.status = 'completed'
            run.progress_pct = 100.0
            s.commit()
    
    except Exception as e:
        logger.error(f"Backtest {run_id} failed: {e}", exc_info=True)
        with SessionLocal() as s:
            run = s.query(BacktestRun).get(run_id)
            run.status = 'failed'
            run.error_message = str(e)
            s.commit()
```

### Scheduler Initialization

```python
# quantagent/scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor

_scheduler = None

def get_scheduler() -> BackgroundScheduler:
    """Get or create APScheduler instance (singleton)."""
    global _scheduler
    if _scheduler is None:
        executors = {
            'default': ThreadPoolExecutor(max_workers=1)
        }
        _scheduler = BackgroundScheduler(executors=executors, timezone='UTC')
    return _scheduler

def init_scheduler():
    """Initialize and start APScheduler."""
    scheduler = get_scheduler()
    
    if scheduler.running:
        logger.info("Scheduler already running")
        return
    
    # Recovery: mark stale runs as failed
    recover_stale_runs()
    
    # Add poller job
    scheduler.add_job(
        poll_pending_backtest_runs,
        trigger='interval',
        seconds=10,
        id='backtest_poller',
        replace_existing=True
    )
    
    # Start scheduler
    scheduler.start()
    logger.info("APScheduler started")

def shutdown_scheduler():
    """Shutdown APScheduler gracefully."""
    scheduler = get_scheduler()
    if scheduler.running:
        scheduler.shutdown(wait=True)
        logger.info("APScheduler stopped")
```

### Streamlit Integration

```python
# apps/streamlit/app.py

from quantagent.scheduler import init_scheduler

@st.cache_resource
def initialize_app():
    """Initialize app resources (called once)."""
    init_scheduler()
    return True

# En main:
initialize_app()
```

---

## Flujo de Ejecución Detallado

### Happy Path

```
1. User submits form in Streamlit
   ↓
2. backtesting.py creates BacktestRun(status=pending)
   ↓
3. Poller job (running every 10s) detects pending run
   ↓
4. Poller triggers execution job: scheduler.add_job(execute_backtest_run, args=[run.id])
   ↓
5. Execution job starts, updates status=running, progress_pct=0
   ↓
6. Execution job calls backtesting_engine.run_backtest(...)
   ↓
7. Engine calls progress_callback periodically
   ↓
8. Execution job updates processed_candles, eta_seconds in DB
   ↓
9. UI polls DB (st.autorefresh every 5s), shows progress
   ↓
10. Engine returns BacktestResult
   ↓
11. Execution job updates status=completed, populates metrics
   ↓
12. Execution job completes, APScheduler removes job
   ↓
13. UI shows final metrics in table
```

### Cancellation Path

```
1. User clicks "Cancel" button in run details
   ↓
2. UI sets cancel_requested=True in DB
   ↓
3. UI calls scheduler.remove_job(f"backtest_run_{run_id}") (best effort)
   ↓
4. Execution job's cancel_check() returns True
   ↓
5. Execution job terminates gracefully
   ↓
6. Execution job updates status=cancelled
   ↓
7. APScheduler removes job (already removed or self-removes)
   ↓
8. UI shows "Cancelled" status
```

### Error Path

```
1. Execution job executing run
   ↓
2. Exception raised (e.g., data fetch failure)
   ↓
3. Job catches exception in try/except
   ↓
4. Job updates status=failed, error_message=str(e)
   ↓
5. Job logs full traceback
   ↓
6. APScheduler removes job (no retry)
   ↓
7. UI shows error in run details
```

---

## Execution Logic Details

### _execute_backtest_logic(run_id)

```python
def _execute_backtest_logic(run_id: int):
    """Core execution logic called by APScheduler job."""
    with SessionLocal() as s:
        run = s.query(BacktestRun).get(run_id)
        
        # Prepare config
        assets = run.config_snapshot.get('assets') or run.profile.universe
        timeframe = run.config_snapshot['timeframe']
        start_date = run.start_date
        end_date = run.end_date
        
        # Progress callback
        def progress_callback(processed: int, total: int):
            with SessionLocal() as sess:
                r = sess.query(BacktestRun).get(run_id)
                r.processed_candles = processed
                r.total_candles = total
                r.progress_pct = (processed / total * 100) if total > 0 else 0
                r.eta_seconds = _estimate_eta(processed, total, r.created_at)
                sess.commit()
        
        # Cancel check
        def cancel_check() -> bool:
            with SessionLocal() as sess:
                r = sess.query(BacktestRun).get(run_id)
                return r.cancel_requested
        
        # Execute backtest
        from quantagent.backtesting.engine import run_backtest
        
        result = run_backtest(
            assets=assets,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            config_snapshot=run.config_snapshot,
            progress_callback=progress_callback,
            cancel_check=cancel_check
        )
        
        # Check if cancelled during execution
        if cancel_check():
            raise CancelledException()
        
        # Populate metrics
        run.win_rate = result.win_rate
        run.profit_factor = result.profit_factor
        run.sharpe_ratio = result.sharpe_ratio
        run.max_drawdown = result.max_drawdown
        run.total_pnl = result.total_pnl
        run.total_trades = result.total_trades
        s.commit()
```

---

## Compatibilidad con QuantAgent-3o4 TradingScheduler

### Si 3o4 ya existe
- Reusar scheduler instance: `from quantagent.trading_scheduler import get_scheduler`
- Agregar backtest poller job al scheduler existente
- Asegurar que job IDs no colisionen (prefijo `backtest_run_` vs `trading_*`)

### Si 3o4 no existe aún
- QuantAgent-9wz implementa APScheduler setup completo
- `quantagent/scheduler.py` actúa como base para futura integración con 3o4
- Cuando 3o4 se implemente, refactor a usar scheduler compartido

### Job ID Conventions (evitar colisiones)
- Backtest execution: `backtest_run_{run_id}`
- Backtest poller: `backtest_poller`
- Trading (futuro): `trading_*`
- Replay (futuro): `replay_run_{replay_id}`

---

## Estimación de ETA

**Algoritmo MVP** (mismo que approach anterior):
```python
def _estimate_eta(processed: int, total: int, started_at: datetime) -> int:
    if processed == 0:
        return None
    elapsed = (datetime.utcnow() - started_at).total_seconds()
    rate = processed / elapsed  # candles per second
    remaining = total - processed
    return int(remaining / rate)
```

---

## Testing Strategy

### Unit Tests
- Mock BacktestRun model + APScheduler
- Test poller job detects pending runs
- Test execution job status transitions
- Test progress updates
- Test cancellation logic (flag + job removal)

### Integration Tests
- E2E: crear run → poller detecta → ejecuta → completa
- E2E: crear run → cancela → status=cancelled
- E2E: crear run → simular error → status=failed
- E2E: scheduler restart → stale runs marked as failed

### Manual Tests (MVP)
- Crear backtest simple (1 asset, 1 día)
- Verificar progreso en UI con autorefresh
- Cancelar mid-execution
- Simular error (config inválido)
- Restart app con run en ejecución → verificar recovery

---

## Migration Strategy

### Alembic Migration

```python
# alembic/versions/xxx_add_backtest_progress_and_scheduler_fields.py

def upgrade():
    op.add_column('backtest_runs', sa.Column('status', sa.Enum(...), default='pending'))
    op.add_column('backtest_runs', sa.Column('progress_pct', sa.Float(), default=0.0))
    op.add_column('backtest_runs', sa.Column('processed_candles', sa.Integer(), nullable=True))
    op.add_column('backtest_runs', sa.Column('total_candles', sa.Integer(), nullable=True))
    op.add_column('backtest_runs', sa.Column('eta_seconds', sa.Integer(), nullable=True))
    op.add_column('backtest_runs', sa.Column('cancel_requested', sa.Boolean(), default=False))
    op.add_column('backtest_runs', sa.Column('error_message', sa.Text(), nullable=True))
    op.add_column('backtest_runs', sa.Column('apscheduler_job_id', sa.Text(), nullable=True))
    
    # Migrar runs existentes a status=pending (o completed si tienen metrics)
    op.execute("""
        UPDATE backtest_runs 
        SET status = CASE 
            WHEN total_trades IS NOT NULL THEN 'completed'
            ELSE 'pending'
        END
    """)

def downgrade():
    op.drop_column('backtest_runs', 'apscheduler_job_id')
    op.drop_column('backtest_runs', 'error_message')
    # ... drop other columns ...
```

---

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| APScheduler config compleja | Baja | Medio | Usar defaults simples MVP; in-memory job store |
| Job ID colisiones con 3o4 | Media | Alto | Prefijos claros (`backtest_run_*`) |
| Poller overhead (polling cada 10s) | Baja | Bajo | 10s es razonable; ajustable si necesario |
| Stale jobs después de crash | Media | Medio | Recovery logic al startup + job removal |
| ETA inaccurate | Alta | Bajo | Algoritmo simple MVP; mejorar luego |
| APScheduler lifecycle en Streamlit | Media | Alto | `@st.cache_resource` para singleton init |

---

## Alternativas Consideradas (Registro)

### Alt 1: Threading + Queue (approach anterior)
**Rechazada**: No alineada con QuantAgent-3o4; duplica infraestructura

### Alt 2: Celery + Redis
**Rechazada**: Overkill para MVP single-machine; considerar para production distribuida

### Alt 3: APScheduler con SQLAlchemy JobStore desde MVP
**Rechazada para MVP**: In-memory suficiente; agregar persistence en futuro si necesario

### Alt 4: On-demand only (no poller, trigger directo al crear run)
**Rechazada**: Acopla UI con scheduler; poller desacopla y es más robusto
