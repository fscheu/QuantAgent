# Planning: Active Position Monitoring System

**Issue ID**: QuantAgent-nu7 (Epic)
**Type**: Feature Enhancement (Architectural)
**Estimated Total Effort**: 6-10 dias

**Real Issue IDs**:
- Epic: QuantAgent-nu7
- Fase 1: QuantAgent-boi
- Fase 2: QuantAgent-enn
- Fase 3: QuantAgent-on4
- Fase 4: QuantAgent-r6y

---

## Resumen de Fases

| Fase | Descripcion | Effort | Dependencias |
|------|-------------|--------|--------------|
| 1 | Modelo y Monitoreo | 1-2 dias | Ninguna |
| 2 | Abstraccion Strategy | 2-3 dias | Fase 1 |
| 3 | Integracion Backtest | 2-3 dias | Fase 1, 2 |
| 4 | Metricas y Validacion | 1-2 dias | Fase 3 |

---

## Fase 1: Modelo y Monitoreo

**Issue ID**: QuantAgent-boi
**Estimated Effort**: 1-2 dias
**Dependencies**: Ninguna

### Tareas

#### 1.1 Crear modelo ActivePosition
**Estimado**: 45 min

- Agregar `ExitPolicy` enum en `models.py`
- Agregar clase `ActivePosition` en `models.py`
- Incluir todos los campos definidos en DS

#### 1.2 Crear migracion Alembic
**Estimado**: 15 min

```bash
alembic revision --autogenerate -m "add_active_positions_table"
alembic upgrade head
```

#### 1.3 Implementar PositionMonitor
**Estimado**: 1.5-2 horas (simplificado)

Archivo: `quantagent/trading/position_monitor.py`

**IMPORTANTE**: PositionMonitor NO decide (ver DT6 en diseño)
- Gestiona estado en DB, NO lógica de exit
- Exit logic está en TradingStrategy.should_exit()

Metodos:
- `__init__(db_session)`
- `get_active_position(symbol) -> Optional[ActivePosition]`
- `update_candle_tracking(position, price, prev_close)` - tracking para métricas
- `close_position(position, reason, exit_price)` - cierra y calcula accuracy
- `open_position(signal, symbol, quantity, entry_price, trade_id, signal_id)` - crea posición

**Métodos ELIMINADOS** (ahora en Strategy):
- ~~check_position()~~ → `TradingStrategy.should_exit()`
- ~~check_stop_loss()~~ → en `TradingStrategy.should_exit()`
- ~~check_take_profit()~~ → en `TradingStrategy.should_exit()`
- ~~check_trailing_stop()~~ → en `TradingStrategy._check_trailing_stop()`

#### 1.4 Tests unitarios para PositionMonitor
**Estimado**: 45 min - 1 hora (simplificado)
Los tests serán desarrollados por el agente Tester
Archivo: `tests/test_position_monitor.py`

Tests (gestión de estado solamente):
- `test_get_active_position`
- `test_open_position_creates_record`
- `test_update_candle_tracking_increments`
- `test_candles_direction_tracking`
- `test_close_position_calculates_accuracy`
- `test_close_position_sets_inactive`

**Tests ELIMINADOS** (ahora en test_trading_strategy.py):
- ~~test_check_stop_loss_*~~ → test en TradingStrategy
- ~~test_check_take_profit_*~~ → test en TradingStrategy
- ~~test_trailing_stop_*~~ → test en TradingStrategy

### Checkpoints Fase 1

- [ ] Modelo ActivePosition en models.py
- [ ] Enum ExitPolicy creado
- [ ] Migracion aplicada exitosamente
- [ ] PositionMonitor implementado
- [ ] Tests unitarios pasando (8+ tests) 

---

## Fase 2: Abstraccion Strategy

**Issue ID**: QuantAgent-enn
**Estimated Effort**: 2-3 dias
**Dependencies**: Fase 1 (QuantAgent-boi) completada

### Tareas

#### 2.1 Crear TradingSignal y TradingStrategy
**Estimado**: 2-3 horas

Archivo: `quantagent/strategy/base.py`

- Clase `TradingSignal` (Pydantic BaseModel)
- ABC `TradingStrategy` con métodos:
  - `generate_signal()` - abstracto
  - `should_exit()` - **implementación default (fixed SL/TP + trailing)**
  - `_check_trailing_stop()` - helper default (% fijo, override para ATR)
  - `should_reevaluate()` - abstracto
  - `get_default_exit_policy()` - concreto

**IMPORTANTE (Template Method Pattern)**:
- `should_exit()` tiene lógica default completa
- Strategies heredan default o hacen override para custom logic
- Ver diseño: docs/03_design/QuantAgent-nu7-DS-active-position-monitoring.md (DT6)

#### 2.2 Implementar LLMAgentStrategy
**Estimado**: 2-3 horas

Archivo: `quantagent/strategy/llm_agent_strategy.py`

- Wrapper de TradingGraph existente
- Parseo de TradingDecision a TradingSignal (usa SL/TP del agente)
- Manejo de errores y fallback a HOLD
- `should_reevaluate()`: return False (LLM no re-evalúa posiciones)
- **NO override should_exit()** - usa default de TradingStrategy (fixed % SL/TP)

#### 2.3 Implementar RSIMeanReversionStrategy
**Estimado**: 1.5-2 horas

Archivo: `quantagent/strategy/rsi_strategy.py`

- Calculo de RSI usando datos kline (talib)
- Logica: RSI > 70 -> SHORT, RSI < 30 -> LONG
- Configuracion de thresholds (default: 30/70)
- Calculo de SL/TP basado en porcentaje fijo
- **NO override should_exit()** - usa default (ejemplo simple)

#### 2.4 Tests unitarios para strategies
**Estimado**: 2-3 horas
Los tests serán desarrollados por el agente Tester
Archivos:
- `tests/test_trading_strategy.py` - Base + default should_exit()
- `tests/test_llm_agent_strategy.py` - LLM wrapper
- `tests/test_rsi_strategy.py` - RSI strategy

Tests base (should_exit default):
- `test_should_exit_stop_loss_triggered`
- `test_should_exit_take_profit_triggered`
- `test_should_exit_trailing_stop_long`
- `test_should_exit_trailing_stop_short`
- `test_should_exit_time_expired`
- `test_should_exit_position_active`

Tests LLM:
- `test_trading_signal_validation`
- `test_llm_strategy_generates_signal`
- `test_llm_strategy_parses_decision`
- `test_llm_strategy_uses_default_should_exit` - verifica que NO override

Tests RSI:
- `test_rsi_strategy_overbought`
- `test_rsi_strategy_oversold`
- `test_rsi_strategy_neutral`

### Checkpoints Fase 2

- [ ] TradingSignal y TradingStrategy en base.py
- [ ] LLMAgentStrategy funcional
- [ ] RSIMeanReversionStrategy funcional
- [ ] Tests unitarios pasando (7+ tests)
- [ ] __init__.py exporta clases

---

## Fase 3: Integracion Backtest

**Issue ID**: QuantAgent-on4
**Estimated Effort**: 2-3 dias
**Dependencies**: Fase 1 (QuantAgent-boi) y Fase 2 (QuantAgent-enn) completadas

### Tareas

#### 3.1 Modificar Backtest.__init__
**Estimado**: 30 min

- Agregar parametro `strategy: Optional[TradingStrategy] = None`
- Crear `LLMAgentStrategy` por defecto si no se pasa
- Inicializar `PositionMonitor`

#### 3.2 Modificar _analyze_and_trade
**Estimado**: 2-3 horas

Refactorizar flujo:
1. Verificar posicion activa
2. Si activa: check_position y decidir accion
3. Si no activa o cerrada: invocar strategy
4. Crear ActivePosition cuando se abre posicion

#### 3.3 Implementar _open_position_from_signal
**Estimado**: 1 hora

- Crear Order via OrderManager
- Crear ActivePosition via PositionMonitor
- Vincular signal_id y trade_id

#### 3.4 Implementar _close_position
**Estimado**: 1 hora

- Cerrar posicion via OrderManager
- Marcar ActivePosition como inactiva
- Registrar close_reason
- Calcular y guardar accuracy

#### 3.5 Actualizar BacktestMetrics
**Estimado**: 45 min

Agregar campos:
- `agent_invocations`
- `invocations_saved`
- `invocation_reduction_pct`
- `close_reasons`

#### 3.6 Tests de integracion
**Estimado**: 2-3 horas
Los tests serán desarrollados por el agente Tester
Archivo: `tests/test_backtest_integration.py`

Tests:
- `test_backtest_with_position_monitor`
- `test_backtest_skips_invocation_when_active`
- `test_backtest_closes_on_sl`
- `test_backtest_closes_on_trailing`
- `test_backtest_with_rsi_strategy`
- `test_backtest_backwards_compatible`

### Checkpoints Fase 3

- [ ] Backtest acepta strategy parameter
- [ ] _analyze_and_trade usa PositionMonitor
- [ ] Posiciones se abren con SL/TP
- [ ] Posiciones se cierran con razon
- [ ] Tests de integracion pasando (6+ tests)
- [ ] Regresion: backtest legacy funciona

---

## Fase 4: Metricas y Validacion

**Issue ID**: QuantAgent-r6y
**Estimated Effort**: 1-2 dias
**Dependencies**: Fase 3 (QuantAgent-on4) completada

### Tareas

#### 4.1 Implementar Mean Directional Accuracy
**Estimado**: 1 hora

- Calcular en _calculate_metrics()
- Agregar a BacktestMetrics
- Calcular por candle (1, 2, 3)

#### 4.2 Agregar tracking de invocaciones
**Estimado**: 30 min

- Contador de invocaciones reales
- Contador de invocaciones ahorradas
- Calcular porcentaje de reduccion

#### 4.3 Ejecutar backtest comparativo
**Estimado**: 2-3 horas

- Correr backtest SIN position monitor (baseline)
- Correr backtest CON position monitor
- Documentar diferencias en metricas
- Verificar reduccion >= 80%

#### 4.4 Documentar resultados
**Estimado**: 1 hora

- Actualizar docs/ con resultados
- Agregar ejemplos de uso
- Documentar configuracion de strategies

### Checkpoints Fase 4

- [ ] MDA calculada correctamente
- [ ] Accuracy por candle disponible
- [ ] Reduccion de invocaciones >= 80%
- [ ] Backtest comparativo documentado
- [ ] Metricas close_reasons funcionando

---

## Riesgos

| Riesgo | Prob | Impacto | Mitigacion |
|--------|------|---------|------------|
| TradingDecision no tiene SL/TP | Media | Alto | Default SL/TP basado en % si None |
| Trailing stop demasiado agresivo | Media | Medio | Config de trailing_stop_pct tunable |
| Regresion en metricas | Baja | Alto | Tests de regresion antes de merge |
| Performance por queries DB | Baja | Medio | Index en symbol + is_active |

---

## Rollout

1. **Feature branch**: `feature/QuantAgent-nu7-active-position-monitoring`
2. **Fase 1**: PR independiente, merge a main
3. **Fase 2**: PR independiente, merge a main
4. **Fase 3**: PR independiente, merge a main
5. **Fase 4**: PR final con validacion

### Comando de test por fase

```bash
# Fase 1
pytest tests/test_position_monitor.py -v

# Fase 2
pytest tests/test_trading_strategy.py tests/test_llm_agent_strategy.py tests/test_rsi_strategy.py -v

# Fase 3
pytest tests/test_backtest_integration.py -v

# Fase 4
python examples/run_backtest.py --compare-modes
```

---

## Comandos bd para Crear Issues

### Epic

```bash
# ✅ YA CREADO: QuantAgent-nu7
bd create \
  --title "Active Position Monitoring System" \
  --description "Epic: Implementar sistema de monitoreo de posiciones activas que reduce invocaciones del agente en 80%+, usa SL/TP del TradingDecision, y alinea metricas con paper QuantAgent.

Fases:
1. Modelo ActivePosition + PositionMonitor
2. Abstraccion TradingStrategy
3. Integracion con Backtest
4. Metricas del paper + validacion

Docs: docs/01_requirements/QuantAgent-nu7-RQ-active-position-monitoring.md" \
  --type task \
  --priority 1 \
  --labels "epic,enhancement,langgraph,backtesting"
```

### Fase 1

```bash
# ✅ YA CREADO: QuantAgent-boi
bd create \
  --title "ActivePosition Model + PositionMonitor" \
  --description "Crear modelo ActivePosition, enum ExitPolicy, migracion Alembic, y clase PositionMonitor con logica de SL/TP/trailing stop.

Archivos:
- quantagent/models.py (modificar)
- quantagent/trading/position_monitor.py (nuevo)
- alembic/versions/ (nueva migracion)
- tests/test_position_monitor.py (nuevo)

AC: docs/05_acceptance_tests/QuantAgent-nu7-AC-active-position-monitoring.md (seccion Fase 1)" \
  --type task \
  --priority 1 \
  --labels "enhancement,langgraph,backtesting"
```

### Fase 2

```bash
# ✅ YA CREADO: QuantAgent-enn
bd create \
  --title "TradingStrategy Abstraction" \
  --description "Crear abstraccion TradingStrategy, TradingSignal, LLMAgentStrategy wrapper, y RSIMeanReversionStrategy ejemplo.

Archivos:
- quantagent/strategy/base.py (nuevo)
- quantagent/strategy/llm_agent_strategy.py (nuevo)
- quantagent/strategy/rsi_strategy.py (nuevo)
- tests/test_trading_strategy.py (nuevo)

AC: docs/05_acceptance_tests/QuantAgent-nu7-AC-active-position-monitoring.md (seccion Fase 2)" \
  --type task \
  --priority 1 \
  --labels "enhancement,langgraph"
```

### Fase 3

```bash
# ✅ YA CREADO: QuantAgent-on4
bd create \
  --title "Backtest Integration with PositionMonitor" \
  --description "Modificar Backtest para usar PositionMonitor y TradingStrategy, evitando invocaciones cuando hay posicion activa.

Archivos:
- quantagent/backtesting/backtest.py (modificar)
- tests/test_backtest_integration.py (nuevo)

AC: docs/05_acceptance_tests/QuantAgent-nu7-AC-active-position-monitoring.md (seccion Fase 3)" \
  --type task \
  --priority 1 \
  --labels "enhancement,backtesting"
```

### Fase 4

```bash
# ✅ YA CREADO: QuantAgent-r6y
bd create \
  --title "Paper Metrics + Validation" \
  --description "Implementar Mean Directional Accuracy, tracking de invocaciones, y validar reduccion >= 80%.

Tareas:
- Calcular MDA y accuracy por candle
- Agregar metricas a BacktestMetrics
- Ejecutar backtest comparativo
- Documentar resultados

AC: docs/05_acceptance_tests/QuantAgent-nu7-AC-active-position-monitoring.md (seccion Fase 4)" \
  --type task \
  --priority 2 \
  --labels "enhancement,backtesting,metrics"
```

### Establecer Dependencias

```bash
# ✅ YA EJECUTADO - Dependencias configuradas:
bd dep add QuantAgent-enn QuantAgent-boi       # Fase 2 depende de Fase 1
bd dep add QuantAgent-on4 QuantAgent-boi       # Fase 3 depende de Fase 1
bd dep add QuantAgent-on4 QuantAgent-enn       # Fase 3 depende de Fase 2
bd dep add QuantAgent-r6y QuantAgent-on4       # Fase 4 depende de Fase 3

# Parent-child con epic:
bd update QuantAgent-boi --parent QuantAgent-nu7
bd update QuantAgent-enn --parent QuantAgent-nu7
bd update QuantAgent-on4 --parent QuantAgent-nu7
bd update QuantAgent-r6y --parent QuantAgent-nu7
```

---

## Referencias

- Requirements: `docs/01_requirements/QuantAgent-nu7-RQ-active-position-monitoring.md`
- Design: `docs/03_design/QuantAgent-nu7-DS-active-position-monitoring.md`
- Acceptance: `docs/05_acceptance_tests/QuantAgent-nu7-AC-active-position-monitoring.md`
