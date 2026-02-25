# Design: TradingScheduler for Automatic Paper Trading

**Issue ID**: QuantAgent-3o4  
**Type**: Feature  
**Level**: TIER 2 ESSENTIAL (MVP Blocker)

---

## Componentes Afectados

- **Nuevo**: `quantagent/trading/scheduler.py` — TradingScheduler class
- **Nuevo**: `apps/paper_trading.py` — Entry point script
- **Modificado**: `quantagent/settings.py` — Agregar SchedulerSettings
- **Modificado**: `pyproject.toml` — Agregar APScheduler dependency

---

## Decisiones Técnicas

### 1. Scheduler Library: APScheduler vs alternativas
**Decisión**: Usar `APScheduler` (BackgroundScheduler)  
**Razón**: 
- Ya especificado en `phase1_roadmap.md` Week 9-10
- Maduro, estable, ampliamente usado
- Soporta interval-based scheduling (requerido)
- No requiere proceso/daemon separado (BackgroundScheduler corre en thread)
- Alternativas evaluadas:
  - `schedule`: Más simple pero menos features (no background threads nativos)
  - `celery`: Overkill para MVP, requiere broker externo (Redis/RabbitMQ)
  - `cron` + systemd: Menos portable, requiere configuración sistema

### 2. Scheduler Type: Interval vs Cron
**Decisión**: Interval-based (cada N horas)  
**Razón**: 
- Requirement especifica "every N hours" (no cron expression)
- Más simple de configurar (float en horas vs cron syntax)
- Suficiente para MVP (paper trading 24/7)
- Cron puede agregarse en Phase 2 si se requiere horarios específicos

### 3. Processing: Sequential vs Parallel
**Decisión**: Sequential (procesar assets uno por uno)  
**Razón**: 
- MVP scope limitado (2-5 assets esperado)
- Evita race conditions con OrderManager/Database
- Más simple de debuggear
- Parallel puede agregarse en futuro si N > 10 assets

### 4. Error Handling Strategy
**Decisión**: Fail-safe (errores transitorios no detienen scheduler)  
**Razón**: 
- Paper trading debe ser resiliente a interrupciones temporales
- Un asset fallando no debe bloquear otros assets
- Tipos de errores:
  - **Transient** (API timeout, network) → log warning, continúa
  - **Logic** (señal HOLD, risk rejection) → log info, continúa
  - **Fatal** (config invalid, dependencies missing) → log error, re-raise

### 5. Configuration: File vs Environment
**Decisión**: `settings.py` (dataclass + env vars)  
**Razón**: 
- Consistente con patrón existente en el proyecto
- Permite overrides via env vars (flexibilidad en despliegue)
- Type-safe (dataclass validation)

### 6. Logging: Print vs Structured
**Decisión**: Structured logging (JSON-compatible)  
**Razón**: 
- Parseable por herramientas de monitoreo
- Incluye metadata (timestamp, asset, action, result)
- Requirement especifica "logs all activities"
- Facilita debugging y auditoría

---

## Contratos

### SchedulerSettings (nuevo en settings.py)

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class SchedulerSettings:
    """Configuration for TradingScheduler."""
    enabled: bool = False
    interval_hours: float = 1.0
    assets: List[str] = field(default_factory=lambda: ["BTC", "SPX"])
    environment: str = 'paper'
    
    def __post_init__(self):
        if self.interval_hours <= 0:
            raise ValueError("interval_hours must be > 0")
        if not self.assets:
            raise ValueError("assets list cannot be empty")
        if self.environment not in ['paper', 'live']:
            raise ValueError("environment must be 'paper' or 'live'")
```

### TradingScheduler Class (scheduler.py)

```python
from apscheduler.schedulers.background import BackgroundScheduler
from typing import List
import logging

class TradingScheduler:
    """Automatic trading scheduler for paper/live trading."""
    
    def __init__(
        self,
        trading_graph: TradingGraph,
        order_manager: OrderManager,
        data_provider: DataProvider,
        config: SchedulerSettings,
        logger: logging.Logger = None,
    ):
        """
        Initialize scheduler with dependencies.
        
        Args:
            trading_graph: Agent system for analysis
            order_manager: Executor for trade decisions
            data_provider: Market data source
            config: Scheduler configuration
            logger: Optional logger (creates one if None)
        """
        self.graph = trading_graph
        self.order_manager = order_manager
        self.data_provider = data_provider
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self._validate_config()
        self.scheduler = BackgroundScheduler()
        self.is_running = False
    
    def _validate_config(self) -> None:
        """Validate configuration (raises ValueError if invalid)."""
        # Delegado a SchedulerSettings.__post_init__
        pass
    
    def start(self) -> None:
        """
        Start the scheduler.
        
        Registers analyze_and_trade() to run every interval_hours.
        Idempotent: calling start() twice is no-op.
        """
        if self.is_running:
            self.logger.warning("Scheduler already running, ignoring start()")
            return
        
        self.scheduler.add_job(
            func=self.analyze_and_trade,
            trigger='interval',
            hours=self.config.interval_hours,
            id='analyze_and_trade',
            replace_existing=True,
        )
        self.scheduler.start()
        self.is_running = True
        
        self.logger.info(
            "Scheduler started",
            extra={
                "interval_hours": self.config.interval_hours,
                "assets": self.config.assets,
                "environment": self.config.environment,
            }
        )
    
    def stop(self) -> None:
        """
        Stop the scheduler gracefully.
        
        Waits for current job to complete before shutting down.
        Idempotent: calling stop() twice is no-op.
        """
        if not self.is_running:
            self.logger.warning("Scheduler not running, ignoring stop()")
            return
        
        self.scheduler.shutdown(wait=True)
        self.is_running = False
        self.logger.info("Scheduler stopped gracefully")
    
    def analyze_and_trade(self) -> None:
        """
        Main scheduled job: analyze all assets and execute decisions.
        
        Iterates over config.assets, processes each sequentially.
        Errors are logged but do not stop processing of other assets.
        """
        self.logger.info(f"Starting analysis cycle for {len(self.config.assets)} assets")
        
        processed = 0
        errors = 0
        
        for asset in self.config.assets:
            try:
                self._process_asset(asset)
                processed += 1
            except Exception as e:
                errors += 1
                self.logger.error(
                    f"Failed to process asset: {asset}",
                    extra={"asset": asset, "error": str(e)},
                    exc_info=True
                )
        
        self.logger.info(
            f"Analysis cycle completed: {processed}/{len(self.config.assets)} processed, {errors} errors"
        )
    
    def _process_asset(self, asset: str) -> None:
        """
        Process a single asset: fetch data → analyze → execute.
        
        Args:
            asset: Asset symbol (e.g., "BTC", "SPX")
        
        Raises:
            Exception: If fatal error occurs (logged and re-raised)
        """
        self.logger.debug(f"Processing asset: {asset}")
        
        # Step 1: Fetch latest data
        try:
            data = self.data_provider.fetch(asset)
        except Exception as e:
            self.logger.warning(
                f"Failed to fetch data for {asset}: {e}",
                extra={"asset": asset, "error": str(e)}
            )
            return  # Skip this asset, continue with next
        
        # Step 2: Run analysis
        try:
            decision = self.graph.analyze(asset, data)
        except Exception as e:
            self.logger.error(
                f"Analysis failed for {asset}: {e}",
                extra={"asset": asset, "error": str(e)},
                exc_info=True
            )
            return  # Skip this asset
        
        # Step 3: Execute decision if not HOLD
        if decision.signal == "HOLD":
            self.logger.info(
                f"No action for {asset}: signal=HOLD",
                extra={"asset": asset, "signal": "HOLD"}
            )
            return
        
        try:
            result = self.order_manager.execute_decision(
                decision,
                environment=self.config.environment
            )
            self.logger.info(
                f"Order executed: {asset} {decision.signal}",
                extra={
                    "asset": asset,
                    "signal": decision.signal,
                    "environment": self.config.environment,
                    "order_id": result.order_id if result else None,
                }
            )
        except Exception as e:
            self.logger.warning(
                f"Order execution failed for {asset}: {e}",
                extra={"asset": asset, "decision": decision.signal, "error": str(e)}
            )
```

### Entry Point Script (apps/paper_trading.py)

```python
#!/usr/bin/env python3
"""
Paper Trading Scheduler Entry Point.

Usage:
    python apps/paper_trading.py [--interval HOURS] [--assets BTC,SPX]
"""
import signal
import sys
import time
import argparse
from quantagent.settings import settings
from quantagent.trading.scheduler import TradingScheduler
from quantagent.trading_graph import TradingGraph
from quantagent.trading.order_manager import OrderManager
from quantagent.data.provider import DataProvider

def main():
    parser = argparse.ArgumentParser(description='Run paper trading scheduler')
    parser.add_argument('--interval', type=float, help='Interval in hours (overrides config)')
    parser.add_argument('--assets', type=str, help='Comma-separated asset list (overrides config)')
    args = parser.parse_args()
    
    # Override config if CLI args provided
    if args.interval:
        settings.scheduler.interval_hours = args.interval
    if args.assets:
        settings.scheduler.assets = [a.strip() for a in args.assets.split(',')]
    
    # Initialize dependencies
    graph = TradingGraph()
    order_manager = OrderManager()
    data_provider = DataProvider()
    
    # Create scheduler
    scheduler = TradingScheduler(
        trading_graph=graph,
        order_manager=order_manager,
        data_provider=data_provider,
        config=settings.scheduler,
    )
    
    # Setup signal handlers for graceful shutdown
    def shutdown_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        scheduler.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    
    # Start scheduler
    scheduler.start()
    
    # Keep process alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.stop()

if __name__ == '__main__':
    main()
```

---

## Flujo de Datos

```
[User] → python apps/paper_trading.py
    ↓
[Entry Point] → TradingScheduler(graph, order_mgr, data_prov, config)
    ↓
[Scheduler] → scheduler.start()
    ↓
[APScheduler] → job registered (interval=1.0h)
    ↓
(wait interval)
    ↓
[APScheduler] → TradingScheduler.analyze_and_trade()
    ↓
[Scheduler] → for asset in ['BTC', 'SPX']:
    ↓
[Scheduler] → _process_asset('BTC'):
    ↓
[DataProvider] → fetch('BTC') → DataFrame
    ↓
[TradingGraph] → analyze('BTC', data) → Decision(signal=LONG, confidence=0.8)
    ↓
[Scheduler] → if signal != HOLD:
    ↓
[OrderManager] → execute_decision(decision, environment='paper') → Order
    ↓
[Database] → INSERT INTO orders (asset, signal, environment) VALUES ('BTC', 'LONG', 'paper')
    ↓
[Logger] → INFO "Order executed: BTC LONG [paper]"
    ↓
(repeat for 'SPX')
    ↓
[Logger] → INFO "Analysis cycle completed: 2/2 assets processed"
```

---

## Error Handling Strategy

### Categorías de Errores

| Error Type | Example | Handling | Impact |
|------------|---------|----------|--------|
| **Configuration** | `interval_hours = 0` | Raise ValueError at startup | Fatal (process exits) |
| **Transient** | API timeout, network error | Log warning, skip asset, continue | Non-fatal (resilience) |
| **Logic** | Signal = HOLD, risk rejection | Log info, skip execution, continue | Non-fatal (expected) |
| **Fatal Runtime** | Database corruption, OOM | Log error, re-raise exception | Fatal (process exits) |

### Implementation Pattern

```python
def _process_asset(self, asset: str) -> None:
    try:
        # Step 1: Fetch (transient errors possible)
        data = self.data_provider.fetch(asset)
    except (TimeoutError, ConnectionError) as e:
        self.logger.warning(f"Transient error: {e}")
        return  # Skip asset, continue with next
    
    try:
        # Step 2: Analyze (logic errors possible)
        decision = self.graph.analyze(asset, data)
    except Exception as e:
        self.logger.error(f"Analysis error: {e}", exc_info=True)
        return  # Skip asset
    
    # Step 3: Execute (logic + transient errors possible)
    if decision.signal != "HOLD":
        try:
            self.order_manager.execute_decision(decision, environment='paper')
        except RiskRejectionError:
            self.logger.info(f"Order rejected by risk manager")
        except Exception as e:
            self.logger.warning(f"Execution error: {e}")
```

---

## Testing Strategy

### Unit Tests (test_scheduler.py)

```python
import pytest
from unittest.mock import Mock, patch

def test_scheduler_start():
    """Verify scheduler starts APScheduler."""
    scheduler = TradingScheduler(mock_graph, mock_order, mock_data, config)
    scheduler.start()
    assert scheduler.is_running
    assert scheduler.scheduler.running

def test_analyze_and_trade_long_signal():
    """Verify LONG signal executes order."""
    mock_graph.analyze.return_value = Decision(signal="LONG", confidence=0.8)
    scheduler = TradingScheduler(mock_graph, mock_order, mock_data, config)
    scheduler.analyze_and_trade()
    mock_order.execute_decision.assert_called_once()

def test_analyze_and_trade_hold_signal():
    """Verify HOLD signal does not execute."""
    mock_graph.analyze.return_value = Decision(signal="HOLD", confidence=0.5)
    scheduler = TradingScheduler(mock_graph, mock_order, mock_data, config)
    scheduler.analyze_and_trade()
    mock_order.execute_decision.assert_not_called()

def test_error_handling_transient():
    """Verify transient errors do not stop processing."""
    mock_data.fetch.side_effect = [TimeoutError, Mock()]  # First fails, second succeeds
    scheduler = TradingScheduler(mock_graph, mock_order, mock_data, config)
    scheduler.analyze_and_trade()
    assert mock_data.fetch.call_count == 2  # Continues to second asset
```

### Integration Test (test_scheduler_integration.py)

```python
def test_scheduler_end_to_end():
    """Full flow: scheduler → analysis → execution → database."""
    # Setup
    scheduler = TradingScheduler(..., config=SchedulerSettings(
        interval_hours=0.016,  # 1 minute
        assets=["BTC", "SPX"],
        environment='paper'
    ))
    
    # Run for 3 cycles
    scheduler.start()
    time.sleep(3 * 60)  # 3 minutes
    scheduler.stop()
    
    # Verify
    orders = db.query("SELECT * FROM orders WHERE environment='paper'")
    assert len(orders) >= 3  # At least 3 analysis runs
    assert all(o.environment == 'paper' for o in orders)
```

---

## Alternativas Consideradas

### Alt 1: Cron-based scheduling (systemd timer o cron daemon)
**Rechazada**: 
- Menos portable (requiere configuración del sistema)
- Más complejo de testear
- No necesario para MVP (interval-based es suficiente)

### Alt 2: Celery + Redis
**Rechazada**: 
- Overkill para MVP
- Agrega complejidad (broker externo)
- APScheduler es más simple y suficiente

### Alt 3: Parallel asset processing (threading/asyncio)
**Rechazada para MVP**: 
- Más complejo
- Riesgo de race conditions con OrderManager/Database
- Sequential es suficiente para 2-5 assets
- Puede agregarse en Phase 2 si N > 10 assets

---

## Impacto en Componentes Existentes

### TradingGraph
- **Cambios**: Ninguno
- **Uso**: `graph.analyze(asset, data)` llamado por scheduler

### OrderManager
- **Cambios**: Ninguno (ya soporta `environment` param)
- **Uso**: `order_manager.execute_decision(decision, environment='paper')`

### DataProvider
- **Cambios**: Ninguno
- **Uso**: `data_provider.fetch(asset)` llamado por scheduler

### Database Schema
- **Cambios**: Ninguno (columna `environment` ya existe en `orders` y `signals`)
- **Uso**: Filtrar por `environment='paper'` para auditoría

---

## Logging Format (Structured)

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Order executed: BTC LONG",
  "asset": "BTC",
  "signal": "LONG",
  "environment": "paper",
  "order_id": "ord-abc123"
}
```

**Ventajas**:
- Parseable por herramientas (ELK, Splunk, etc.)
- Filtrable por campo (e.g., `environment='paper'`)
- Facilita debugging y auditoría

---

## Dependencias de Implementación

1. `pyproject.toml` → Agregar `APScheduler>=3.10.0,<4.0.0`
2. `settings.py` → Agregar `SchedulerSettings` dataclass
3. `scheduler.py` → Implementar `TradingScheduler` class
4. `apps/paper_trading.py` → Entry point script
5. Unit tests → `tests/trading/test_scheduler.py`
6. Integration test → `tests/integration/test_scheduler_integration.py`

**Critical Path**: 1 → 2 → 3 → 4 (5 y 6 en paralelo)
