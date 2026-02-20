# Requirements: TradingScheduler for Automatic Paper Trading

**Issue ID**: QuantAgent-3o4  
**Type**: Feature  
**Level**: TIER 2 ESSENTIAL (MVP Blocker)

---

## Objetivo

Implementar un scheduler que ejecute análisis de trading automáticamente cada N horas sin intervención manual, permitiendo paper trading continuo 24/7 para validación de estrategias antes de despliegue a broker real.

---

## Alcance

### Incluye
- Scheduler basado en APScheduler (interval-based, no cron)
- Ejecución automática de análisis cada N horas (configurable, default: 1 hora)
- Integración con TradingGraph (análisis), OrderManager (ejecución), DataProvider (datos)
- Environment tagging: todos los registros marcados como `environment='paper'`
- Logging estructurado de todas las actividades (inicio, análisis, decisiones, errores)
- Graceful shutdown (manejo de señales SIGTERM/SIGINT)
- Configuración via `settings.py` (enabled, interval, assets, environment)
- Entry point script: `apps/paper_trading.py`

### No Incluye
- Integración con broker real (solo paper trading)
- Human-in-the-loop approval (ejecución automática completa)
- Schedulers basados en cron expressions (solo interval-based)
- Scheduling por horarios de mercado específicos (corre 24/7)
- Dashboard/UI de monitoreo (solo logs y database)
- Parallel processing de múltiples assets (procesamiento secuencial)
- Adaptive intervals (intervalo fijo)

---

## Constraints

- **Estabilidad**: Debe operar continuamente ≥ 24h sin crashes ni memory leaks
- **Configurabilidad**: Intervalo y lista de assets configurables sin cambio de código
- **Logging**: Todos los eventos logueados en formato estructurado (JSON-compatible)
- **Error Handling**: Errores transitorios (API, red) no detienen el scheduler
- **Environment Isolation**: Todos los registros DB/logs deben identificarse como `environment='paper'`
- **Dependencies**: Reusar componentes existentes (TradingGraph, OrderManager) sin modificarlos
- **Graceful Shutdown**: Ctrl+C o SIGTERM detienen el scheduler sin pérdida de estado

---

## Flujo de Operación

### 1. Startup
```
Usuario inicia: python apps/paper_trading.py
    ↓
Carga configuración desde settings.py
    ↓
Valida config (interval > 0, assets no vacía)
    ↓
Inicializa dependencias (TradingGraph, OrderManager, DataProvider)
    ↓
Crea TradingScheduler instance
    ↓
Inicia APScheduler (BackgroundScheduler)
    ↓
Registra signal handlers (SIGTERM, SIGINT)
    ↓
Log: "Scheduler started, interval=1.0h, assets=['BTC', 'SPX']"
    ↓
Entra en loop infinito (mantiene proceso vivo)
```

### 2. Scheduled Run (cada N horas)
```
APScheduler dispara job
    ↓
TradingScheduler.analyze_and_trade()
    ↓
Itera sobre cada asset en config:
    Para asset en ['BTC', 'SPX']:
        ↓
        _process_asset(asset):
            ↓
            Fetch latest data (DataProvider)
            ↓
            Run analysis (TradingGraph)
            ↓
            Get decision (LONG/SHORT/HOLD)
            ↓
            Si decision != HOLD:
                → OrderManager.execute_decision(decision, environment='paper')
                → Log: "Order executed: BTC LONG 100 shares [paper]"
            Si decision == HOLD:
                → Log: "No action: BTC signal=HOLD"
            ↓
            Manejo de errores:
                - Transient (API timeout) → log warning, continúa con siguiente asset
                - Fatal (config error) → log error, re-raise exception
    ↓
Log: "Analysis cycle completed: 2/2 assets processed"
```

### 3. Shutdown
```
Usuario presiona Ctrl+C o envía SIGTERM
    ↓
Signal handler captura señal
    ↓
TradingScheduler.stop()
    ↓
APScheduler.shutdown(wait=True)
    ↓
Log: "Scheduler stopped gracefully"
    ↓
Exit
```

---

## Edge Cases

### Configuración
- **Interval = 0**: Validación debe rechazar (ValueError)
- **Assets vacía**: Validación debe rechazar (ValueError)
- **Asset no soportado**: DataProvider fallará → log warning, continúa con siguiente

### Runtime
- **API timeout durante fetch**: Log warning "Failed to fetch BTC data: timeout", continúa con siguiente asset
- **TradingGraph exception**: Log error "Analysis failed for SPX: <error>", continúa con siguiente asset
- **OrderManager rejection** (risk check): Log info "Order rejected: insufficient balance", continúa
- **Scheduler ya running**: Intentar start() dos veces debe ser idempotent (no-op o warning)

### Shutdown
- **Shutdown durante análisis**: APScheduler espera a que job actual termine (wait=True)
- **Shutdown sin start previo**: stop() debe ser idempotent (no error)

---

## Definition of Done

- [ ] APScheduler instalado y configurado (`pyproject.toml`)
- [ ] `SchedulerSettings` agregado a `settings.py` (enabled, interval_hours, assets, environment)
- [ ] `TradingScheduler` implementado en `quantagent/trading/scheduler.py`
- [ ] Entry point script `apps/paper_trading.py` funcional
- [ ] Todos los registros tagueados como `environment='paper'`
- [ ] Logging estructurado de inicio, análisis, decisiones, errores
- [ ] Unit tests ≥ 70% coverage
- [ ] Integration test end-to-end (scheduler → analysis → execution → database)
- [ ] Stability test: 24h uptime > 99%, memory growth < 20%
- [ ] Graceful shutdown verificado (Ctrl+C no corrompe estado)
- [ ] Documentación en README: configuración, inicio, monitoreo

---

## Dependencias

### Componentes Existentes (no modificar)
- `TradingGraph`: Para análisis y generación de decisiones
- `OrderManager`: Para ejecución de órdenes (ya soporta `environment` param)
- `DataProvider`: Para fetch de datos históricos/actuales
- `settings.py`: Para configuración central

### Nueva Dependencia Externa
- `APScheduler>=3.10.0,<4.0.0`: Scheduler library (especificado en phase1_roadmap.md)

---

## Non-Functional Requirements

### Performance
- **Latency**: Análisis de un asset debe completar en < 30 segundos
- **Throughput**: Procesar N assets secuencialmente (no parallel en MVP)

### Reliability
- **Uptime**: ≥ 99% durante 24h
- **Error Recovery**: Errores transitorios no detienen scheduler

### Observability
- **Logs**: Todos los eventos logueados con timestamp, asset, action, result
- **Database**: Señales y órdenes registradas con `environment='paper'` para filtrado

### Security
- **Environment Isolation**: Garantizar que paper trading no afecta cuentas reales
- **Credential Management**: API keys para datos de mercado (no para broker en MVP)

---

## Referencias

- **Requirement Source**: `docs/02_planning/trading_system_requirements.md` (2.2 TradingScheduler)
- **Roadmap**: `docs/02_planning/phase1_roadmap.md` Week 9-10
- **Planning Doc**: `docs/02_planning/QuantAgent-3o4-PL-trading-scheduler.md`
