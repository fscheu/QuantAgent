# Acceptance Criteria: TradingScheduler for Automatic Paper Trading

**Issue ID**: QuantAgent-3o4  
**Type**: Feature  
**Level**: TIER 2 ESSENTIAL (MVP Blocker)

---

## Criterios de Aceptación

### AC-1: Scheduler Start (Happy Path)
```
Given configuración válida en settings.py:
  - scheduler.enabled = True
  - scheduler.interval_hours = 1.0
  - scheduler.assets = ["BTC", "SPX"]
  - scheduler.environment = "paper"
When TradingScheduler.start() se ejecuta
Then APScheduler se inicia correctamente
  And job "analyze_and_trade" está registrado con interval=1.0h
  And is_running = True
  And log contiene "Scheduler started, interval=1.0h, assets=['BTC', 'SPX']"
```

### AC-2: Scheduler Stop (Graceful Shutdown)
```
Given scheduler está running (is_running = True)
When TradingScheduler.stop() se ejecuta
Then APScheduler se detiene con wait=True (espera job actual)
  And is_running = False
  And log contiene "Scheduler stopped gracefully"
  And no hay jobs pendientes
```

### AC-3: Analysis Cycle - LONG Signal
```
Given scheduler running
  And DataProvider.fetch("BTC") retorna datos válidos
  And TradingGraph.analyze("BTC", data) retorna Decision(signal="LONG", confidence=0.8)
When scheduled job analyze_and_trade() se ejecuta
Then OrderManager.execute_decision() es llamado con:
  - decision.signal = "LONG"
  - environment = "paper"
  And log contiene "Order executed: BTC LONG [paper]"
  And database contiene orden con environment="paper"
```

### AC-4: Analysis Cycle - SHORT Signal
```
Given scheduler running
  And TradingGraph.analyze("SPX", data) retorna Decision(signal="SHORT", confidence=0.75)
When scheduled job analyze_and_trade() se ejecuta
Then OrderManager.execute_decision() es llamado con:
  - decision.signal = "SHORT"
  - environment = "paper"
  And log contiene "Order executed: SPX SHORT [paper]"
  And database contiene orden con environment="paper"
```

### AC-5: Analysis Cycle - HOLD Signal (No Action)
```
Given scheduler running
  And TradingGraph.analyze("BTC", data) retorna Decision(signal="HOLD", confidence=0.3)
When scheduled job analyze_and_trade() se ejecuta
Then OrderManager.execute_decision() NO es llamado
  And log contiene "No action for BTC: signal=HOLD"
  And no se crea orden en database
```

### AC-6: Error Handling - Transient (API Timeout)
```
Given scheduler running con assets=["BTC", "SPX"]
  And DataProvider.fetch("BTC") lanza TimeoutError
  And DataProvider.fetch("SPX") retorna datos válidos
When scheduled job analyze_and_trade() se ejecuta
Then log contiene warning "Failed to fetch data for BTC: timeout"
  And procesamiento continúa con "SPX"
  And SPX es procesado normalmente
  And log final contiene "Analysis cycle completed: 1/2 processed, 1 errors"
```

### AC-7: Error Handling - Analysis Failure
```
Given scheduler running
  And TradingGraph.analyze("BTC", data) lanza Exception("Model error")
When scheduled job analyze_and_trade() se ejecuta
Then log contiene error "Analysis failed for BTC: Model error"
  And procesamiento continúa con siguiente asset
  And scheduler sigue running (no crash)
```

### AC-8: Configuration Validation - Invalid Interval
```
Given SchedulerSettings con interval_hours = 0
When SchedulerSettings.__post_init__() se ejecuta
Then se lanza ValueError con mensaje "interval_hours must be > 0"
```

### AC-9: Configuration Validation - Empty Assets List
```
Given SchedulerSettings con assets = []
When SchedulerSettings.__post_init__() se ejecuta
Then se lanza ValueError con mensaje "assets list cannot be empty"
```

### AC-10: Stability Test - 24h Uptime
```
Given scheduler running con interval_hours=0.083 (5 minutes)
  And assets=["BTC", "SPX"]
  And test environment con mocks para external APIs
When scheduler corre por 24 horas continuas
Then uptime > 99% (permite <15 min downtime)
  And memory growth < 20% (baseline vs 24h)
  And análisis runs completados ≈ 288 por asset (12/hour × 24h)
  And success rate ≥ 95% (permite algunos errores transitorios)
  And no crashes o unhandled exceptions
  And log file size crece linealmente (no log spam)
```

### AC-11: Environment Tagging - Database Records
```
Given scheduler running con environment="paper"
  And decisiones LONG ejecutadas para BTC y SPX
When se consulta database
Then todas las órdenes tienen environment="paper":
  - SELECT * FROM orders WHERE asset IN ('BTC', 'SPX')
  - Todos los registros: environment='paper'
  And todas las señales tienen environment="paper":
  - SELECT * FROM signals WHERE asset IN ('BTC', 'SPX')
  - Todos los registros: environment='paper'
```

### AC-12: Entry Point - CLI Arguments Override
```
Given config por defecto: interval=1.0, assets=["BTC", "SPX"]
When entry point se ejecuta con: python apps/paper_trading.py --interval 0.5 --assets ETH,BNB
Then scheduler usa interval=0.5 hours
  And scheduler procesa assets=["ETH", "BNB"]
  And log confirma configuración: "interval=0.5h, assets=['ETH', 'BNB']"
```

### AC-13: Entry Point - Signal Handling (SIGTERM)
```
Given scheduler running via entry point
When proceso recibe SIGTERM (kill -TERM <pid>)
Then signal handler captura señal
  And TradingScheduler.stop() es llamado
  And APScheduler hace shutdown con wait=True
  And proceso termina con exit code 0
  And log contiene "Received signal 15, shutting down..."
  And log contiene "Scheduler stopped gracefully"
```

### AC-14: Entry Point - Signal Handling (Ctrl+C / SIGINT)
```
Given scheduler running via entry point
When usuario presiona Ctrl+C
Then KeyboardInterrupt o SIGINT es capturado
  And TradingScheduler.stop() es llamado
  And proceso termina limpiamente
  And log contiene "Scheduler stopped gracefully"
```

### AC-15: Idempotency - Double Start
```
Given scheduler ya está running (is_running = True)
When TradingScheduler.start() es llamado nuevamente
Then no se registra job duplicado
  And log contiene warning "Scheduler already running, ignoring start()"
  And scheduler continúa operando normalmente
```

### AC-16: Idempotency - Stop Without Start
```
Given scheduler NO está running (is_running = False)
When TradingScheduler.stop() es llamado
Then no se lanza excepción
  And log contiene warning "Scheduler not running, ignoring stop()"
  And operación es no-op
```

---

## Criterios de Regresión

### REG-1: TradingGraph Sin Cambios
```
Given TradingGraph existente (pre-scheduler)
When se importa y usa TradingGraph
Then comportamiento es idéntico a versión anterior
  And métodos analyze() y compile() sin cambios
  And tests existentes pasan sin modificaciones
```

### REG-2: OrderManager Sin Cambios
```
Given OrderManager existente (pre-scheduler)
When se llama execute_decision(..., environment='paper')
Then comportamiento es idéntico a versión anterior
  And parámetro environment es respetado
  And tests existentes pasan sin modificaciones
```

### REG-3: DataProvider Sin Cambios
```
Given DataProvider existente (pre-scheduler)
When se llama fetch(asset)
Then comportamiento es idéntico a versión anterior
  And tests existentes pasan sin modificaciones
```

---

## Invariantes del Sistema

### Logging Invariants
- **Todos los eventos logueados**: Inicio, análisis, decisiones, ejecuciones, errores, shutdown
- **Formato estructurado**: JSON-compatible con campos timestamp, level, message, extra metadata
- **Niveles correctos**: INFO (operaciones normales), WARNING (errores transitorios), ERROR (errores fatales)

### Database Invariants
- **Environment tagging**: 100% de órdenes y señales tagueadas con `environment='paper'`
- **No cross-contamination**: Ninguna orden `environment='live'` creada por scheduler paper
- **Consistency**: Cada decisión ejecutada (signal != HOLD) genera exactamente 1 orden

### Error Recovery Invariants
- **Resilience**: Errores transitorios (API, red) no detienen scheduler
- **Isolation**: Fallo en un asset no afecta procesamiento de otros assets
- **Graceful degradation**: Si N/M assets fallan, scheduler continúa con M-N exitosos

---

## Oráculos de Validación

### Validación de Instalación
```bash
# Verificar APScheduler instalado
pip list | grep -i apscheduler

# Verificar import exitoso
python -c "from apscheduler.schedulers.background import BackgroundScheduler; print('OK')"
```

### Validación de Configuración
```python
# Verificar SchedulerSettings en settings.py
from quantagent.settings import settings
assert hasattr(settings, 'scheduler')
assert settings.scheduler.interval_hours > 0
assert len(settings.scheduler.assets) > 0
```

### Validación de Scheduler
```python
# Test básico de start/stop
from quantagent.trading.scheduler import TradingScheduler
from unittest.mock import Mock

scheduler = TradingScheduler(
    trading_graph=Mock(),
    order_manager=Mock(),
    data_provider=Mock(),
    config=settings.scheduler,
)
scheduler.start()
assert scheduler.is_running
scheduler.stop()
assert not scheduler.is_running
```

### Validación de Environment Tagging
```sql
-- Verificar todas las órdenes tienen environment
SELECT COUNT(*) FROM orders WHERE environment IS NULL;
-- Expected: 0

-- Verificar órdenes paper trading
SELECT COUNT(*) FROM orders WHERE environment='paper';
-- Expected: > 0 (si scheduler corrió)
```

### Validación de Logs
```bash
# Verificar logs estructurados (parseable JSON)
tail -100 logs/scheduler.log | grep -E '"timestamp":.*"level":.*"message":'

# Verificar eventos clave
grep "Scheduler started" logs/scheduler.log
grep "Analysis cycle completed" logs/scheduler.log
grep "Order executed" logs/scheduler.log
```

---

## Datos de Prueba

### Configuración Mínima Válida (settings.py)
```python
scheduler = SchedulerSettings(
    enabled=True,
    interval_hours=1.0,
    assets=["BTC", "SPX"],
    environment='paper',
)
```

### Configuración para Testing Rápido (1 minuto)
```python
scheduler = SchedulerSettings(
    enabled=True,
    interval_hours=0.016,  # 1 minute
    assets=["BTC"],
    environment='paper',
)
```

### Configuración para Stability Test (5 minutos, 24h)
```python
scheduler = SchedulerSettings(
    enabled=True,
    interval_hours=0.083,  # 5 minutes
    assets=["BTC", "SPX"],
    environment='paper',
)
```

### Mock Decisions para Tests
```python
# LONG signal (should execute)
Decision(signal="LONG", confidence=0.8, asset="BTC")

# SHORT signal (should execute)
Decision(signal="SHORT", confidence=0.75, asset="SPX")

# HOLD signal (should NOT execute)
Decision(signal="HOLD", confidence=0.3, asset="ETH")
```

---

## Métricas de Éxito

### Cobertura de Tests
- **Unit tests**: ≥ 70% cobertura de `scheduler.py`
- **Integration test**: End-to-end flow (scheduler → database)
- **Stability test**: 24h uptime > 99%

### Performance
- **Latency**: Análisis de un asset < 30 segundos
- **Throughput**: N assets procesados en < N × 30 segundos

### Reliability
- **Error rate**: < 5% errores transitorios permitidos
- **Crash rate**: 0% (no unhandled exceptions)

### Operabilidad
- **Logs completos**: 100% eventos logueados
- **Graceful shutdown**: 100% éxito en shutdown (no estado corrupto)

---

## Checklist de Validación Final

Antes de cerrar issue QuantAgent-3o4, verificar:

- [ ] **AC-1 a AC-16**: Todos los acceptance criteria pasan
- [ ] **REG-1 a REG-3**: No hay regresiones en componentes existentes
- [ ] **Unit tests**: ≥ 70% coverage, todos pasan
- [ ] **Integration test**: End-to-end test pasa
- [ ] **Stability test**: 24h test completa con >99% uptime
- [ ] **Logging**: Todos los eventos logueados, formato estructurado
- [ ] **Database**: Environment tagging verificado (100% records='paper')
- [ ] **Documentation**: README actualizado con instrucciones de uso
- [ ] **Code review**: Aprobado por al menos 1 reviewer
- [ ] **Entry point**: `python apps/paper_trading.py` funciona sin errores

**Comando de Verificación Final**:
```bash
# Run all tests
pytest tests/trading/test_scheduler.py -v --cov=quantagent/trading/scheduler
pytest tests/integration/test_scheduler_integration.py -v

# Manual test: start scheduler
python apps/paper_trading.py --interval 0.016  # 1 minute for quick test

# In another terminal: monitor logs
tail -f logs/scheduler.log

# Wait for 3 cycles (3 minutes), then Ctrl+C to test graceful shutdown

# Verify database records
psql -d quantagent -c "SELECT COUNT(*) FROM orders WHERE environment='paper';"

# Expected: At least 3 orders (1 per minute × 3 minutes, if signals != HOLD)
```

---

## Criterios de Rechazo (Blockers)

Issue **NO** puede cerrarse si:

- ❌ **AC-10 (Stability)** falla: Scheduler crash o memory leak en 24h test
- ❌ **AC-11 (Environment Tagging)** falla: Alguna orden sin `environment='paper'`
- ❌ **AC-13/AC-14 (Graceful Shutdown)** falla: Shutdown corrompe estado o logs
- ❌ **REG-1/2/3 (Regresión)** falla: Componentes existentes rotos
- ❌ **Unit tests** < 70% coverage o tests fallan
- ❌ **Integration test** falla
- ❌ **Documentation** falta o incompleta

Estos son **MVP blockers** — deben resolverse antes de cerrar issue.
