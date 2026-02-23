# Requirements: Active Position Monitoring System

**Issue ID**: QuantAgent-apm (Epic)
**Type**: Feature Enhancement (Architectural)
**Level**: COMPREHENSIVE

---

## Objetivo

Implementar un sistema de monitoreo de posiciones activas que:
1. Reduce invocaciones del agente LLM en 80-87%
2. Utiliza stop_loss/take_profit generados por TradingDecision
3. Alinea el sistema con las metricas del paper (prediccion sobre 3 candles)
4. Abstrae estrategias para permitir implementaciones sin LLM

---

## Contexto y Problema

### Situacion Actual
- El sistema invoca `TradingGraph` en **cada tick** (cada 4h)
- Backtest de 20 dias con 2 assets genera 240+ invocaciones
- `TradingDecision` genera `stop_loss` y `take_profit` que **no se usan**
- El paper QuantAgent evalua prediccion sobre "next three candlesticks" pero el sistema re-evalua en cada candle

### Impacto
- Costos elevados de API (tokens)
- Desalineacion con metodologia del paper
- Comportamiento poco realista (trader real no re-evalua constantemente)

---

## Alcance

### Incluye
- Modelo `ActivePosition` para trackear posiciones con SL/TP/trailing
- Enum `ExitPolicy` para politicas de salida configurables
- Clase `PositionMonitor` para monitoreo automatico
- Abstraccion `TradingStrategy` (protocol/ABC)
- Implementacion `LLMAgentStrategy` (wrapper de TradingGraph)
- Implementacion `RSIMeanReversionStrategy` (ejemplo sin LLM)
- Modificacion de `Backtest._analyze_and_trade()` para skip de invocaciones
- Metricas de accuracy sobre horizonte de 3 candles
- Tracking de razones de cierre (SL/TP/trailing/time)

### No Incluye
- Cambios en UI Streamlit (futuro)
- Nuevas estrategias mas alla del ejemplo RSI
- Optimizacion de hiperparametros
- Integracion con brokers reales
- Cambios en agentes individuales (indicator, pattern, trend)

---

## Constraints

| Constraint | Descripcion |
|------------|-------------|
| Compatibilidad | Backtest existente debe seguir funcionando sin cambios de API |
| Performance | No introducir latencia adicional significativa |
| Datos | Usar modelos SQLAlchemy existentes como base |
| Migracion | Alembic migration para nuevo modelo |
| Testing | Mantener coverage existente, agregar tests nuevos |

---

## Modelo ActivePosition

### Campos Requeridos

**Identificacion:**
- `id`: Primary key
- `symbol`: Simbolo del activo
- `side`: OrderSide (BUY/SELL para LONG/SHORT)

**Precios:**
- `entry_price`: Precio de entrada
- `stop_loss`: Precio de stop loss (del TradingDecision)
- `take_profit`: Precio de take profit (del TradingDecision)
- `quantity`: Cantidad de la posicion

**Horizonte Temporal:**
- `decision_timestamp`: Momento de la decision del agente
- `candles_since_entry`: Contador de candles desde entrada
- `max_hold_candles`: Maximo candles antes de forzar cierre (opcional)

**Politica de Salida:**
- `exit_policy`: ExitPolicy enum
- `prediction_horizon`: Horizonte de prediccion (default: 3)

**Trailing Stop:**
- `trailing_stop_pct`: Porcentaje de trailing (opcional)
- `highest_price_seen`: Precio mas alto desde entrada (LONG)
- `lowest_price_seen`: Precio mas bajo desde entrada (SHORT)

**Tracking de Accuracy:**
- `candles_direction`: Lista de direcciones por candle (para validacion)

**Links:**
- `trade_id`: FK a Trade
- `signal_id`: FK a Signal

### ExitPolicy Enum

```
SL_TP_ONLY   - Solo cierra por SL o TP
TIME_BASED   - Cierra despues de N candles
REEVALUATE   - Re-evalua con agente despues de N candles
TRAILING_STOP - Trailing stop activo
```

---

## TradingStrategy Abstraction

### Interface

```
TradingStrategy:
  - generate_signal(kline_data, symbol, timeframe) -> Optional[TradingSignal]
  - should_reevaluate(position, current_price) -> bool
  - get_exit_policy() -> ExitPolicy
```

### TradingSignal (estandarizada)

```
TradingSignal:
  - decision: str (LONG/SHORT/HOLD)
  - confidence: float
  - entry_price: Optional[float]
  - stop_loss: Optional[float]
  - take_profit: Optional[float]
  - reasoning: str
```

---

## Flujos Funcionales

### Flujo Principal (Backtest con Monitor)

```
Para cada tick:
  1. position = position_monitor.get_active_position(symbol)

  2. Si position existe:
     a. action = position_monitor.check_position(position, current_price)
     b. Si action == ACTIVE:
        - Incrementar candles_since_entry
        - SKIP invocacion de strategy (ahorro)
        - return
     c. Si action == CLOSE_SL/CLOSE_TP/CLOSE_TRAILING/CLOSE_TIME:
        - Cerrar posicion con razon
        - Registrar accuracy
     d. Si action == REEVALUATE:
        - Invocar strategy
        - Si decision opuesta: cerrar y abrir nueva
        - Si misma direccion: actualizar SL/TP

  3. Si no hay position:
     a. signal = strategy.generate_signal(...)
     b. Si signal != HOLD:
        - Crear ActivePosition con SL/TP del signal
        - Abrir posicion via OrderManager
```

### Flujo de Trailing Stop

```
En cada check_position():
  1. Actualizar highest/lowest price seen
  2. Calcular trailing_stop_price:
     - LONG: highest_price_seen * (1 - trailing_stop_pct)
     - SHORT: lowest_price_seen * (1 + trailing_stop_pct)
  3. Si current_price cruza trailing_stop_price:
     - Retornar CLOSE_TRAILING
```

### Flujo de Accuracy Tracking

```
En cada tick mientras position activa:
  1. Determinar direccion del candle (up/down/flat)
  2. Agregar a candles_direction list
  3. Cuando position cierra:
     a. Calcular correct = candles en direccion esperada
     b. accuracy = correct / min(prediction_horizon, len(candles_direction))
     c. Persistir en metricas
```

---

## Edge Cases

| Caso | Comportamiento |
|------|----------------|
| SL == TP (invalido) | Rechazar posicion, log warning |
| Precio abre con gap sobre SL | Ejecutar SL al precio de apertura |
| position.max_hold_candles = 0 | Forzar cierre inmediato |
| Trailing activado pero no configurado | Usar SL fijo |
| Signal.stop_loss == None | Calcular SL default (ej: 2% del entry) |
| Multiples posiciones por simbolo | Permitir solo 1 (constraint existente) |

---

## Definition of Done

### Fase 1: Modelo y Monitoreo
- [ ] Modelo ActivePosition en models.py con todos los campos
- [ ] Enum ExitPolicy creado
- [ ] Migracion Alembic generada y aplicada
- [ ] Clase PositionMonitor con metodos check_*
- [ ] Tests unitarios para PositionMonitor

### Fase 2: Abstraccion Strategy
- [ ] ABC TradingStrategy definida
- [ ] TradingSignal modelo creado
- [ ] LLMAgentStrategy implementada (wrapper de TradingGraph)
- [ ] RSIMeanReversionStrategy implementada (ejemplo)
- [ ] Tests unitarios para ambas strategies

### Fase 3: Integracion Backtest
- [ ] Backtest recibe TradingStrategy como parametro
- [ ] _analyze_and_trade usa PositionMonitor
- [ ] Posiciones se crean con SL/TP del signal
- [ ] Razones de cierre se registran
- [ ] Tests de integracion pasando

### Fase 4: Metricas y Validacion
- [ ] Mean Directional Accuracy calculada
- [ ] Metricas por candle (1, 2, 3) disponibles
- [ ] Backtest comparativo ejecutado
- [ ] Reduccion de invocaciones verificada (>80%)
- [ ] Documentacion de resultados

---

## Metricas de Exito

| Metrica | Objetivo | Medicion |
|---------|----------|----------|
| Reduccion de invocaciones | >= 80% | agent_calls_after / agent_calls_before |
| Mean Directional Accuracy | >= 50% | correct_direction_candles / total_candles |
| Latencia por tick | < 50ms (sin LLM) | tiempo cuando position activa |
| Test coverage | >= 80% | pytest --cov |

---

## Referencias

- Paper QuantAgent: arXiv:2509.09995
- Codigo actual: `quantagent/backtesting/backtest.py`
- TradingDecision con SL/TP: `quantagent/agent_models.py:129-135`
- Modelos existentes: `quantagent/models.py`
