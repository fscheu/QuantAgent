# Design: Active Position Monitoring System

**Issue ID**: QuantAgent-apm (Epic)
**Type**: Feature Enhancement (Architectural)
**Level**: COMPREHENSIVE

---

## Componentes Afectados

### Nuevos
- `quantagent/models.py` - Agregar ActivePosition, ExitPolicy
- `quantagent/trading/position_monitor.py` - Nueva clase PositionMonitor
- `quantagent/strategy/base.py` - TradingStrategy ABC, TradingSignal
- `quantagent/strategy/llm_agent_strategy.py` - LLMAgentStrategy
- `quantagent/strategy/rsi_strategy.py` - RSIMeanReversionStrategy

### Modificados
- `quantagent/backtesting/backtest.py` - Integrar PositionMonitor y Strategy
- `quantagent/backtesting/metrics.py` - Agregar metricas del paper (si existe, o en backtest.py)
- `alembic/versions/` - Nueva migracion

---

## Diagrama de Arquitectura

```
                    +------------------+
                    |     Backtest     |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+        +----------v----------+
    |  PositionMonitor  |        |   TradingStrategy   |
    +-------------------+        +---------------------+
    | - get_active()    |        | - generate_signal() |
    | - open_position() |        | - should_exit() ← ★ |
    | - close_position()|        | - should_reevaluate()|
    | - track_accuracy()|        +----------^----------+
    +---------+---------+                   |
              |                  +----------+----------+
              v                  |                     |
    +-------------------+   +----v--------+   +--------v--------+
    |  ActivePosition   |   |LLMAgentStrat|   |TripleScreenStrat|
    |     (Model)       |   +-------------+   +-----------------+
    +-------------------+   | Default SL/TP|  | ATR-based stops |
                            | (hereda)    |   | (override)      |
                            +-------------+   +-----------------+

★ Template Method Pattern:
  - should_exit() tiene implementación default (fixed SL/TP)
  - Strategies pueden override para lógica custom (ATR, time-based, etc.)
```

---

## Decisiones Tecnicas

### DT1: ActivePosition como modelo SQLAlchemy vs Pydantic
**Decision**: SQLAlchemy model persistido en DB
**Razon**:
- Consistencia con modelos existentes (Trade, Order, Signal)
- Auditoria y reproducibilidad
- Permite retomar backtests interrumpidos

### DT2: Strategy como Protocol vs ABC
**Decision**: ABC (Abstract Base Class)
**Razon**:
- Mas explicito para implementadores
- Permite metodos con implementacion default
- Consistencia con patrones existentes en el repo

### DT3: ExitPolicy como atributo de ActivePosition vs Strategy
**Decision**: Atributo de ActivePosition, configurado por Strategy
**Razon**:
- Permite cambiar politica mid-trade si es necesario
- Cada posicion puede tener politica diferente
- Strategy sugiere, Position guarda

### DT4: Trailing Stop - porcentaje vs ATR
**Decision**: Porcentaje fijo
**Razon**:
- Simplicidad para MVP
- ATR requiere calculo adicional por tick
- Extensible a ATR en futuro

### DT5: Accuracy tracking - por posicion vs agregado
**Decision**: Por posicion, agregado en metricas
**Razon**:
- Granularidad para analisis
- Permite segmentar por asset, timeframe, etc.

### DT6: Exit logic - PositionMonitor centralizado vs dentro de Strategy
**Decision**: Modelo Híbrido (Template Method Pattern)
**Razon**:
- **Problema con PositionMonitor centralizado**: Asume que todas las strategies usan misma lógica (fixed % SL/TP)
- **Realidad**: Diferentes strategies necesitan lógica diferente:
  - LLM Strategy: Fixed % stops (simple)
  - Triple Screen: ATR-based trailing stops (dinámico)
  - Future strategies: Time-based, volatility-based, etc.
- **Solución híbrida**:
  - `TradingStrategy.should_exit()` tiene implementación default (fixed SL/TP)
  - Strategies que necesitan lógica custom hacen override
  - PositionMonitor gestiona estado (DB operations) pero NO decide
- **Ventajas**:
  - Cohesión: Strategy que crea posición decide cómo salir
  - Flexibilidad: Cada strategy puede tener exit logic custom
  - Reutilización: Default implementation evita duplicación
  - Testing: Cada strategy testeable independientemente

---

## Modelo de Datos

### ActivePosition (nuevo)

```python
class ExitPolicy(str, enum.Enum):
    SL_TP_ONLY = "sl_tp_only"
    TIME_BASED = "time_based"
    REEVALUATE = "reevaluate"
    TRAILING_STOP = "trailing_stop"

class ActivePosition(Base):
    __tablename__ = "active_positions"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(Enum(OrderSide), nullable=False)

    entry_price = Column(Numeric(18, 8), nullable=False)
    stop_loss = Column(Numeric(18, 8), nullable=False)
    take_profit = Column(Numeric(18, 8), nullable=False)
    quantity = Column(Numeric(18, 8), nullable=False)

    decision_timestamp = Column(DateTime, nullable=False)
    candles_since_entry = Column(Integer, default=0)

    exit_policy = Column(Enum(ExitPolicy), nullable=False)
    max_hold_candles = Column(Integer, nullable=True)

    prediction_horizon = Column(Integer, default=3)
    candles_direction = Column(JSON, default=list)

    trailing_stop_pct = Column(Float, nullable=True)
    highest_price_seen = Column(Numeric(18, 8), nullable=True)
    lowest_price_seen = Column(Numeric(18, 8), nullable=True)

    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)
    signal_id = Column(Integer, ForeignKey("signals.id"), nullable=True)

    is_active = Column(Boolean, default=True, index=True)
    closed_at = Column(DateTime, nullable=True)
    close_reason = Column(String(50), nullable=True)

    environment = Column(Enum(Environment), default=Environment.BACKTEST)
```

---

## Contratos de Interface

### TradingStrategy (ABC)

```python
class TradingSignal(BaseModel):
    decision: str  # LONG, SHORT, HOLD
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    exit_policy: ExitPolicy = ExitPolicy.TRAILING_STOP
    trailing_stop_pct: Optional[float] = None
    max_hold_candles: Optional[int] = None

class TradingStrategy(ABC):
    @abstractmethod
    def generate_signal(
        self,
        kline_data: List[Dict],
        symbol: str,
        timeframe: str,
        current_price: float,
    ) -> Optional[TradingSignal]:
        """Genera senal de trading basada en datos"""
        pass

    def should_exit(
        self,
        position: ActivePosition,
        current_price: float,
        ohlc_data: pd.DataFrame,
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide si salir de posición activa.

        DEFAULT IMPLEMENTATION: Fixed % SL/TP check
        Strategies pueden override para lógica custom (ATR, time-based, etc.)

        Args:
            position: Posición activa actual
            current_price: Precio actual del asset
            ohlc_data: Datos OHLC para cálculos (e.g., ATR)

        Returns:
            (should_exit, reason)
            - should_exit: True si debe cerrar posición
            - reason: "STOP_LOSS", "TAKE_PROFIT", "TRAILING_STOP", etc.
        """
        # Default: Fixed SL/TP check
        if current_price <= position.stop_loss:
            return (True, "STOP_LOSS")

        if current_price >= position.take_profit:
            return (True, "TAKE_PROFIT")

        # Trailing stop (si está habilitado)
        if position.exit_policy == ExitPolicy.TRAILING_STOP:
            if self._check_trailing_stop(position, current_price):
                return (True, "TRAILING_STOP")

        # Time-based (si está configurado)
        if position.max_hold_candles and position.candles_since_entry >= position.max_hold_candles:
            if position.exit_policy == ExitPolicy.TIME_BASED:
                return (True, "TIME_EXPIRED")

        return (False, None)

    def _check_trailing_stop(self, position: ActivePosition, current_price: float) -> bool:
        """
        Helper para verificar trailing stop (default: % fijo).
        Strategies pueden override para usar ATR u otra métrica.
        """
        if not position.trailing_stop_pct:
            return False

        # Update highest/lowest seen
        if position.side == OrderSide.BUY:
            if position.highest_price_seen is None or current_price > position.highest_price_seen:
                position.highest_price_seen = current_price

            # Trailing stop desde el máximo
            trailing_sl = position.highest_price_seen * (1 - position.trailing_stop_pct)
            return current_price < trailing_sl

        else:  # SHORT
            if position.lowest_price_seen is None or current_price < position.lowest_price_seen:
                position.lowest_price_seen = current_price

            trailing_sl = position.lowest_price_seen * (1 + position.trailing_stop_pct)
            return current_price > trailing_sl

    @abstractmethod
    def should_reevaluate(
        self,
        position: ActivePosition,
        current_price: float,
    ) -> bool:
        """Determina si debe re-evaluar posicion activa (solo para REEVALUATE policy)"""
        pass

    def get_default_exit_policy(self) -> ExitPolicy:
        """Politica de salida por defecto"""
        return ExitPolicy.TRAILING_STOP
```

### PositionMonitor

**Rol simplificado**: Gestión de estado y persistencia, NO lógica de decisión.

```python
class PositionMonitor:
    """
    Gestiona el estado de posiciones activas en DB.
    NO decide cuándo salir - eso es responsabilidad de TradingStrategy.
    """

    def __init__(self, db_session: Session):
        self.db = db_session

    def get_active_position(self, symbol: str) -> Optional[ActivePosition]:
        """Obtiene posicion activa para un simbolo"""
        return (
            self.db.query(ActivePosition)
            .filter(
                ActivePosition.symbol == symbol,
                ActivePosition.is_active == True
            )
            .first()
        )

    def update_candle_tracking(
        self,
        position: ActivePosition,
        current_price: float,
        prev_close: float,
    ) -> None:
        """
        Actualiza tracking para métricas del paper (3-candle accuracy).
        Solo gestión de estado, no decisión.
        """
        position.candles_since_entry += 1

        # Track dirección de candle para accuracy
        if len(position.candles_direction) < position.prediction_horizon:
            direction = "up" if current_price > prev_close else "down"
            position.candles_direction.append(direction)

        self.db.commit()

    def close_position(
        self,
        position: ActivePosition,
        reason: str,
        exit_price: float,
    ) -> None:
        """Cierra posicion y calcula accuracy"""
        position.is_active = False
        position.closed_at = datetime.utcnow()
        position.close_reason = reason

        # Calcular accuracy del paper si aplica
        if position.candles_direction:
            expected_direction = "up" if position.side == OrderSide.BUY else "down"
            correct = sum(1 for d in position.candles_direction if d == expected_direction)
            position.accuracy = correct / len(position.candles_direction)

        self.db.commit()

    def open_position(
        self,
        signal: TradingSignal,
        symbol: str,
        quantity: Decimal,
        entry_price: float,
        trade_id: int,
        signal_id: int,
    ) -> ActivePosition:
        """Crea nueva posicion activa"""
        position = ActivePosition(
            symbol=symbol,
            side=OrderSide.BUY if signal.decision == "LONG" else OrderSide.SELL,
            entry_price=entry_price,
            stop_loss=signal.stop_loss or entry_price * 0.98,
            take_profit=signal.take_profit or entry_price * 1.03,
            quantity=quantity,
            decision_timestamp=datetime.utcnow(),
            exit_policy=signal.exit_policy,
            max_hold_candles=signal.max_hold_candles,
            trailing_stop_pct=signal.trailing_stop_pct,
            trade_id=trade_id,
            signal_id=signal_id,
        )

        self.db.add(position)
        self.db.commit()
        return position
```

---

## Flujo de Integracion con Backtest

### Flujo Actualizado (Modelo Híbrido)

```python
# En Backtest.__init__
def __init__(self, ..., strategy: Optional[TradingStrategy] = None):
    # Si no se pasa strategy, usar LLMAgentStrategy por defecto
    self.strategy = strategy or LLMAgentStrategy(self.trading_graph)
    self.position_monitor = PositionMonitor(self.db)

# En Backtest._analyze_and_trade
def _analyze_and_trade(self, asset: str, current_date: datetime):
    current_price = self._get_current_price(asset, current_date)
    ohlc_data = self._get_ohlc_data(asset, current_date)

    # Verificar si hay posicion activa
    active_pos = self.position_monitor.get_active_position(asset)

    if active_pos:
        # ★ CLAVE: Strategy decide si salir
        should_exit, reason = self.strategy.should_exit(
            active_pos,
            current_price,
            ohlc_data  # Para strategies que usen ATR, etc.
        )

        if should_exit:
            # Cerrar posición
            self.position_monitor.close_position(active_pos, reason, current_price)
            self.order_manager.close_trade(active_pos.trade_id, current_price)
            logger.info(f"{asset}: Closed position - {reason}")
            # Continuar para potencialmente abrir nueva posicion

        else:
            # Posición activa, solo actualizar tracking
            prev_close = ohlc_data.iloc[-2]['close']
            self.position_monitor.update_candle_tracking(
                active_pos, current_price, prev_close
            )
            return  # ✅ NO invocar strategy - ahorro de invocación

    # Sin posicion activa o recién cerrada: invocar strategy
    signal = self.strategy.generate_signal(
        kline_data, asset, self.timeframe, current_price
    )

    if signal and signal.decision != "HOLD":
        self._open_position_from_signal(asset, signal, current_price)
```

### Ejemplos de Strategies con Lógica Custom

**LLMAgentStrategy (usa default):**
```python
class LLMAgentStrategy(TradingStrategy):
    def generate_signal(self, ...):
        # Invoca TradingGraph
        result = self.trading_graph.invoke(...)
        return TradingSignal(
            stop_loss=result.stop_loss,
            take_profit=result.take_profit,
            ...
        )

    def should_reevaluate(self, position, price):
        return False  # LLM no re-evalúa

    # ✅ NO override should_exit() → usa fixed % SL/TP default
```

**TripleScreenStrategy (override con ATR):**
```python
class TripleScreenStrategy(TradingStrategy):
    def should_exit(self, position, current_price, ohlc_data):
        # Custom: ATR-based trailing stop
        atr = talib.ATR(ohlc_data['high'], ohlc_data['low'], ohlc_data['close'])
        current_atr = atr.iloc[-1]

        if position.side == OrderSide.BUY:
            # Update highest
            if position.highest_price_seen is None or current_price > position.highest_price_seen:
                position.highest_price_seen = current_price

            # ATR trailing stop
            trailing_sl = position.highest_price_seen - (2 * current_atr)
            if current_price < trailing_sl:
                return (True, "ATR_TRAILING_STOP")

        # También check SL/TP fijos como fallback
        return super().should_exit(position, current_price, ohlc_data)
```

---

## Metricas del Paper

### Mean Directional Accuracy (MDA)

```
alpha = C / T

Donde:
- C = candles con direccion correcta
- T = total candles evaluados (hasta prediction_horizon)

Direccion correcta:
- LONG: candle cierra > abre
- SHORT: candle cierra < abre
```

### BacktestMetrics Extendido

```python
@dataclass
class BacktestMetrics:
    # Existentes...

    # Nuevos
    agent_invocations: int
    invocations_saved: int
    invocation_reduction_pct: float

    mean_directional_accuracy: float
    accuracy_by_candle: Dict[int, float]  # {1: 0.55, 2: 0.52, 3: 0.48}

    close_reasons: Dict[str, int]  # {"close_sl": 10, "close_tp": 5, ...}
```

---

## Migracion Alembic

### Cambios Requeridos

1. Crear tabla `active_positions`
2. Agregar columna `close_reason` a `trades` (opcional)

### Comando

```bash
alembic revision --autogenerate -m "add_active_positions_table"
alembic upgrade head
```

---

## Compatibilidad

### Backtest Existente
- Si no se pasa `strategy`, usa `LLMAgentStrategy` (comportamiento actual)
- Si no hay `ActivePosition` para un simbolo, invoca strategy (comportamiento actual)
- Metricas existentes siguen funcionando

### API Publica
- `Backtest.run()` no cambia signature
- `BacktestMetrics` solo agrega campos (no rompe)

---

## Estructura de Archivos

```
quantagent/
  models.py                    # + ActivePosition, ExitPolicy
  trading/
    position_monitor.py        # NEW
  strategy/
    __init__.py                # exports
    base.py                    # TradingStrategy, TradingSignal
    llm_agent_strategy.py      # LLMAgentStrategy
    rsi_strategy.py            # RSIMeanReversionStrategy
  backtesting/
    backtest.py                # MODIFIED
```

---

## Dependencias de Implementacion

```
Fase 1 (sin dependencias externas):
  models.py: ActivePosition, ExitPolicy
  trading/position_monitor.py: PositionMonitor
  alembic migration

Fase 2 (depende de Fase 1):
  strategy/base.py: TradingStrategy, TradingSignal
  strategy/llm_agent_strategy.py: usa TradingGraph existente
  strategy/rsi_strategy.py: puro Python

Fase 3 (depende de Fase 1 y 2):
  backtesting/backtest.py: integra todo

Fase 4 (depende de Fase 3):
  metricas y validacion
```

---

## Testing Strategy

### Unit Tests
- `test_position_monitor.py`: check_sl, check_tp, check_trailing, accuracy
- `test_trading_strategy.py`: LLMAgentStrategy, RSIStrategy
- `test_active_position.py`: modelo, validaciones

### Integration Tests
- `test_backtest_with_monitor.py`: flujo completo
- `test_invocation_reduction.py`: verificar ahorro real

### Regression Tests
- Backtest existente sin strategy param debe comportarse igual
