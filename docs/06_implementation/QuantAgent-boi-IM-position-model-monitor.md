# Implementation: QuantAgent-boi - ActivePosition Model + PositionMonitor

**Issue ID**: QuantAgent-boi  
**Epic**: QuantAgent-nu7 (Active Position Monitoring System)  
**Phase**: 1 - Modelo y Monitoreo  
**Date**: 2026-01-10

---

## Qué se cambió

### 1. Modelo ActivePosition (`quantagent/models.py`)
- Agregado `ExitPolicy` enum con 4 políticas: `SL_TP_ONLY`, `TIME_BASED`, `REEVALUATE`, `TRAILING_STOP`
- Agregado modelo `ActivePosition` con todos los campos especificados en diseño:
  - Identificación: `symbol`, `side`
  - Precios: `entry_price`, `stop_loss`, `take_profit`, `quantity`
  - Horizonte temporal: `decision_timestamp`, `candles_since_entry`, `max_hold_candles`
  - Políticas: `exit_policy`, `prediction_horizon`
  - Tracking: `candles_direction` (JSON), `accuracy`
  - Trailing stop: `trailing_stop_pct`, `highest_price_seen`, `lowest_price_seen`
  - Links: `trade_id`, `signal_id`
  - Estado: `is_active`, `closed_at`, `close_reason`, `environment`

### 2. PositionMonitor (`quantagent/trading/position_monitor.py`)
Nueva clase con métodos de gestión de estado (NO decisión):
- `get_active_position(symbol)` - Obtiene posición activa
- `open_position(...)` - Crea nueva posición activa
- `update_candle_tracking(...)` - Actualiza contador y dirección de candles
- `close_position(...)` - Cierra posición y calcula accuracy

**Nota importante**: Usa `flag_modified()` para actualizar JSON field `candles_direction`

### 3. Migración Alembic (`alembic/versions/f7d3bad02cae_add_active_positions_table.py`)
- Crea tabla `active_positions` con todos los campos
- Crea índices: `idx_symbol_is_active`, `idx_active_positions_environment`
- Comentadas operaciones sobre tablas de checkpoints de LangGraph (no son manejadas por Alembic)

### 4. Tests (`tests/test_position_monitor.py`)
9 tests unitarios cubriendo:
- Creación de posiciones
- Obtención de posiciones activas
- Tracking de candles (up/down)
- Respeto del horizonte de predicción
- Cierre de posiciones con accuracy (LONG/SHORT)
- Constraint de una sola posición activa por símbolo

---

## Por qué

Implementación de **Fase 1** del epic QuantAgent-nu7, que establece la base de datos y lógica de estado para el sistema de monitoreo de posiciones activas.

**Decisión técnica (DT6)**: Modelo Híbrido
- PositionMonitor NO decide cuándo salir
- Solo gestiona estado en DB (CRUD operations)
- Exit logic será responsabilidad de `TradingStrategy.should_exit()` en Fase 2

---

## Cómo testear

### Unit tests
```bash
source .venv_wsl/bin/activate
pytest tests/test_position_monitor.py -v
# Expected: 9/9 passed
```

### Quality gates
```bash
# Formato
black --check quantagent/models.py quantagent/trading/position_monitor.py tests/test_position_monitor.py

# Imports
isort --check-only quantagent/models.py quantagent/trading/position_monitor.py tests/test_position_monitor.py

# Linting
flake8 quantagent/models.py quantagent/trading/position_monitor.py tests/test_position_monitor.py --max-line-length=120

# Compilation
python -m compileall -q quantagent/models.py quantagent/trading/position_monitor.py
```

### Aplicar migración (cuando esté listo)
```bash
alembic upgrade head
```

---

## Archivos modificados

- `quantagent/models.py` - +67 líneas (ExitPolicy enum, ActivePosition model)
- `quantagent/trading/position_monitor.py` - +108 líneas (nuevo archivo)
- `alembic/versions/f7d3bad02cae_add_active_positions_table.py` - +66 líneas (nueva migración)
- `tests/test_position_monitor.py` - +250 líneas (nuevo archivo)

---

## Próximos pasos (Fase 2)

**Issue QuantAgent-enn**: TradingStrategy Abstraction
- Crear ABC `TradingStrategy` con método `should_exit()` (con implementación default)
- Implementar `LLMAgentStrategy` (wrapper de TradingGraph)
- Implementar `RSIMeanReversionStrategy` (ejemplo sin LLM)
- Crear modelo `TradingSignal` estandarizado

La integración con Backtest será en Fase 3 (QuantAgent-on4).

---

## Desvíos del diseño

Ninguno. Implementación sigue exactamente el diseño en:
- `docs/03_design/QuantAgent-nu7-DS-active-position-monitoring.md`
- `docs/05_acceptance_tests/QuantAgent-nu7-AC-active-position-monitoring.md`

---

## Riesgos conocidos

1. **Migración de checkpoints**: La migración generada intentaba eliminar tablas de LangGraph. Se comentaron esas líneas para evitar conflictos.
2. **Mypy warnings**: Modelos existentes tienen warnings de tipo (pre-existentes, no introducidos por este cambio).
3. **Migración no aplicada**: La migración aún no se ha aplicado a DB. Se debe ejecutar `alembic upgrade head` cuando esté listo.

---

## Commit

```
feat(QuantAgent-boi): Add ActivePosition model and PositionMonitor

SHA: 4334633
Branch: feature/QuantAgent-nu7-active-position-monitoring
```
