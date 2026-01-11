# Implementation Notes: TradingStrategy Abstraction

**Issue ID**: QuantAgent-enn  
**Fecha**: 2026-01-10  
**Status**: Completed  

---

## Resumen

Implementación de la abstracción `TradingStrategy` usando Template Method Pattern para permitir strategies con y sin LLM, con lógica de exit configurable.

---

## Archivos Modificados/Creados

### Nuevos
- `quantagent/strategy/base.py` — TradingStrategy ABC + TradingSignal + should_exit() default
- `quantagent/strategy/llm_agent_strategy.py` — LLMAgentStrategy (wrapper de TradingGraph)
- `quantagent/strategy/rsi_strategy.py` — RSIMeanReversionStrategy (ejemplo sin LLM)
- `tests/test_trading_strategy.py` — Tests de Template Method Pattern
- `tests/test_llm_agent_strategy.py` — Tests de LLM strategy
- `tests/test_rsi_strategy.py` — Tests de RSI strategy

### Modificados
- `quantagent/strategy/__init__.py` — Exports de TradingStrategy, TradingSignal, strategies

---

## Decisiones de Implementación

### 1. Template Method Pattern (DT6)

**Problema Original:**  
Diseño inicial consideraba poner exit logic en `PositionMonitor` (centralizado), pero esto asume que todas las strategies usan misma lógica (fixed % SL/TP).

**Realidad:**  
Diferentes strategies necesitan lógica diferente:
- LLM Strategy: Fixed % stops (simple)
- Triple Screen: ATR-based trailing stops (dinámico)
- Future strategies: Time-based, volatility-based, etc.

**Solución Implementada:**  
- `TradingStrategy.should_exit()` tiene implementación **DEFAULT** (fixed % SL/TP + trailing stop)
- Strategies que necesitan lógica custom hacen **override**
- `PositionMonitor` gestiona estado (DB operations) pero **NO decide**

**Ventajas:**
- ✅ Cohesión: Strategy que crea posición decide cómo salir
- ✅ Flexibilidad: Cada strategy puede tener exit logic custom
- ✅ Reutilización: Default implementation evita duplicación
- ✅ Testing: Cada strategy testeable independientemente

### 2. TradingSignal vs TradingDecision

**Decisión:** Crear nuevo modelo `TradingSignal` (no reutilizar `TradingDecision`).

**Razón:**
- `TradingDecision` es específico del graph LLM (tiene `risk_level`, estructura particular)
- `TradingSignal` es interface genérica para **cualquier** strategy
- Permite strategies sin LLM (RSI, Triple Screen, etc.)

**Campos de TradingSignal:**
```python
decision: str  # LONG/SHORT/HOLD
confidence: float  # 0.0-1.0
entry_price: Optional[float]
stop_loss: Optional[float]
take_profit: Optional[float]
reasoning: str
exit_policy: ExitPolicy
trailing_stop_pct: Optional[float]
max_hold_candles: Optional[int]
```

### 3. LLMAgentStrategy Parsing

**Challenge:** `TradingGraph` retorna dict con `final_trade_decision` (string libre).

**Solución:** Parser robusto con regex para extraer:
- Decision: "LONG", "SHORT", "HOLD"
- Confidence: busca floats entre 0.0-1.0 en el string

**Ejemplo:**
- Input: `"LONG with 0.75 confidence"`
- Output: `("LONG", 0.75)`

### 4. RSI Strategy - Pure Python

**Implementación:**
- Calcula RSI usando pandas (no talib, para evitar dependencia adicional)
- Formula estándar: `RSI = 100 - (100 / (1 + RS))` donde `RS = avg_gain / avg_loss`
- Thresholds configurables: oversold < 30, overbought > 70

**Sin LLM:** Zero invocaciones de modelo, solo math puro.

---

## Cómo Testear

### Unit Tests
```bash
source .venv_wsl/bin/activate
pytest tests/test_trading_strategy.py tests/test_llm_agent_strategy.py tests/test_rsi_strategy.py -v
```

**Cobertura:** 32 tests, todos pasando:
- 14 tests: Template Method Pattern (should_exit default logic)
- 9 tests: LLMAgentStrategy (graph integration, parsing)
- 9 tests: RSIMeanReversionStrategy (RSI logic, no LLM)

### Manual Testing

**Test LLM Strategy:**
```python
from quantagent.trading_graph import TradingGraph
from quantagent.strategy import LLMAgentStrategy

graph = TradingGraph()
strategy = LLMAgentStrategy(graph)

signal = strategy.generate_signal(
    kline_data=[...],  # 30+ candles
    symbol="BTCUSDT",
    timeframe="4h",
    current_price=50000.0
)

print(signal.decision, signal.confidence)
```

**Test RSI Strategy:**
```python
from quantagent.strategy import RSIMeanReversionStrategy

strategy = RSIMeanReversionStrategy(
    rsi_period=14,
    oversold_threshold=30.0,
    overbought_threshold=70.0
)

signal = strategy.generate_signal(
    kline_data=[...],  # Downtrend data
    symbol="BTCUSDT",
    timeframe="4h",
    current_price=50000.0
)

# Should return LONG if oversold
assert signal.decision == "LONG"
```

---

## Quality Gates

✅ **Black:** Formatting checked and applied  
✅ **Isort:** Imports sorted  
✅ **Flake8:** No linting errors  
✅ **Pytest:** 32/32 tests passing  
✅ **Compileall:** Syntax validated  

---

## Acceptance Criteria (Fase 2)

Según `docs/05_acceptance_tests/QuantAgent-nu7-AC-active-position-monitoring.md`:

- ✅ **AC2.1:** LLMAgentStrategy genera signal válido (LONG/SHORT/HOLD)
- ✅ **AC2.2:** LLMAgentStrategy usa TradingGraph existente (no re-evalúa)
- ✅ **AC2.3:** RSIMeanReversionStrategy genera signal sin LLM
- ✅ **AC2.4:** Strategy retorna HOLD/None cuando no hay señal clara
- ✅ **AC2.5:** TradingSignal incluye exit_policy

---

## Próximos Pasos

**Bloqueado:** QuantAgent-on4 (Backtest Integration with PositionMonitor)

**Necesario para on4:**
1. Modificar `Backtest.__init__()` para aceptar `strategy` param
2. Modificar `Backtest._analyze_and_trade()` para usar `PositionMonitor` + `TradingStrategy`
3. Implementar flujo híbrido:
   - Si hay posición activa: `strategy.should_exit()` decide
   - Si no hay posición: `strategy.generate_signal()` crea nueva

**Referencias:**
- Diseño: `docs/03_design/QuantAgent-nu7-DS-active-position-monitoring.md` (DT6)
- AC: `docs/05_acceptance_tests/QuantAgent-nu7-AC-active-position-monitoring.md` (Fase 3)

---

## Riesgos / Deudas

**Ninguna:** Implementación completa según diseño. No hay deuda técnica introducida.

**Nota:** Cambios en `assembler.py` y `test_strategy_assembler.py` fueron automáticos (isort), no impactan funcionalidad.

---

## Commit

**Commit Hash:** 8759c57  
**Branch:** feature/QuantAgent-nu7-active-position-monitoring  
**Mensaje:** "feat(QuantAgent-enn): Implement TradingStrategy abstraction with Template Method Pattern"
