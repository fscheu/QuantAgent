# Acceptance Criteria: Active Position Monitoring System

**Issue ID**: QuantAgent-apm (Epic)
**Type**: Feature Enhancement (Architectural)

---

## Criterios de Aceptacion - Fase 1: Modelo y Monitoreo

### AC1.1: Modelo ActivePosition persistido
```
Given un backtest que genera una senal LONG
When se crea una posicion a partir del signal
Then ActivePosition se persiste en la tabla active_positions
  And contiene symbol, side, entry_price, stop_loss, take_profit
  And is_active = True
  And candles_since_entry = 0
```

### AC1.2: ExitPolicy configurado correctamente
```
Given un TradingSignal con exit_policy = TRAILING_STOP
When se crea ActivePosition
Then position.exit_policy == ExitPolicy.TRAILING_STOP
  And position.trailing_stop_pct tiene valor configurado
```

### AC1.3: Stop Loss ejecutado
```
Given una posicion LONG activa con stop_loss = 100
When current_price = 99 (< stop_loss)
Then PositionMonitor.check_position() retorna PositionAction.CLOSE_SL
```

### AC1.4: Take Profit ejecutado
```
Given una posicion LONG activa con take_profit = 110
When current_price = 111 (> take_profit)
Then PositionMonitor.check_position() retorna PositionAction.CLOSE_TP
```

### AC1.5: Trailing Stop actualiza precio maximo
```
Given una posicion LONG activa con entry_price = 100, trailing_stop_pct = 0.05
When current_price sube a 110
Then position.highest_price_seen = 110
  And trailing_stop_price = 110 * 0.95 = 104.5
```

### AC1.6: Trailing Stop ejecutado
```
Given una posicion LONG con highest_price_seen = 110, trailing_stop_pct = 0.05
When current_price baja a 104 (< 104.5)
Then PositionMonitor.check_position() retorna PositionAction.CLOSE_TRAILING
```

### AC1.7: Posicion activa no invoca agente
```
Given una posicion activa que no cumple SL/TP/trailing/time
When check_position() se ejecuta
Then retorna PositionAction.ACTIVE
  And candles_since_entry incrementa en 1
```

### AC1.8: Tracking de direccion por candle
```
Given una posicion LONG activa
When un candle cierra con close > open
Then "up" se agrega a position.candles_direction
```

---

## Criterios de Aceptacion - Fase 2: Abstraccion Strategy

### AC2.1: LLMAgentStrategy genera signal valido
```
Given datos de mercado validos (30+ candles)
When LLMAgentStrategy.generate_signal() se ejecuta
Then retorna TradingSignal con decision in (LONG, SHORT, HOLD)
  And confidence entre 0.0 y 1.0
  And si decision != HOLD: stop_loss y take_profit tienen valor
```

### AC2.2: LLMAgentStrategy usa TradingGraph existente
```
Given LLMAgentStrategy inicializada con TradingGraph
When generate_signal() se ejecuta
Then invoca graph.invoke() internamente
  And parsea TradingDecision de resultado
```

### AC2.3: RSIMeanReversionStrategy genera signal sin LLM
```
Given datos de mercado con RSI > 70 (overbought)
When RSIMeanReversionStrategy.generate_signal() se ejecuta
Then retorna TradingSignal con decision = SHORT
  And NO invoca ningun LLM
```

### AC2.4: Strategy retorna HOLD cuando no hay senal clara
```
Given datos de mercado con RSI = 50 (neutral)
When RSIMeanReversionStrategy.generate_signal() se ejecuta
Then retorna TradingSignal con decision = HOLD
  Or retorna None
```

### AC2.5: TradingSignal incluye exit_policy
```
Given una strategy con get_default_exit_policy() = TRAILING_STOP
When generate_signal() retorna un signal
Then signal.exit_policy == ExitPolicy.TRAILING_STOP
```

---

## Criterios de Aceptacion - Fase 3: Integracion Backtest

### AC3.1: Backtest acepta strategy como parametro
```
Given un RSIMeanReversionStrategy
When Backtest(strategy=rsi_strategy) se inicializa
Then backtest.strategy es la instancia pasada
  And NO se crea TradingGraph
```

### AC3.2: Backtest usa LLMAgentStrategy por defecto
```
Given Backtest() inicializado sin strategy
Then backtest.strategy es instancia de LLMAgentStrategy
  And usa TradingGraph internamente
```

### AC3.3: Posicion activa evita invocacion de strategy
```
Given un backtest en ejecucion
  And una posicion LONG activa en BTC
When _analyze_and_trade(BTC, date) se ejecuta
  And posicion no cumple SL/TP/trailing
Then strategy.generate_signal() NO se invoca
  And candles_since_entry incrementa
```

### AC3.4: Cierre de posicion registra razon
```
Given una posicion activa cerrada por stop_loss
When se registra el Trade resultante
Then trade.notes contiene "close_reason: close_sl"
  Or close_reason se registra en ActivePosition.close_reason
```

### AC3.5: Nueva posicion se crea con SL/TP del signal
```
Given un TradingSignal con stop_loss=95, take_profit=110
When se crea ActivePosition
Then position.stop_loss = 95
  And position.take_profit = 110
```

### AC3.6: Compatibilidad con backtest existente
```
Given un backtest sin strategy parameter (legacy)
When run() se ejecuta
Then comportamiento es identico a version anterior
  And genera metricas validas
```

---

## Criterios de Aceptacion - Fase 4: Metricas del Paper

### AC4.1: Mean Directional Accuracy calculada
```
Given un backtest completado con multiples trades
When metrics se calculan
Then mean_directional_accuracy esta entre 0.0 y 1.0
  And representa (candles correctos / total candles evaluados)
```

### AC4.2: Accuracy por candle disponible
```
Given posiciones con prediction_horizon = 3
When backtest completa
Then metrics.accuracy_by_candle = {1: X, 2: Y, 3: Z}
  Donde X, Y, Z son floats entre 0.0 y 1.0
```

### AC4.3: Reduccion de invocaciones medida
```
Given un backtest de 20 dias con 2 assets (240 ticks potenciales)
When backtest completa con posiciones que duran multiples candles
Then metrics.invocations_saved > 0
  And metrics.invocation_reduction_pct calculado correctamente
```

### AC4.4: Reduccion >= 80% con trailing stop
```
Given configuracion con exit_policy = TRAILING_STOP
  And condiciones de mercado tipicas
When backtest de 20 dias completa
Then invocation_reduction_pct >= 80%
```

### AC4.5: Close reasons agregadas
```
Given un backtest completado
When metrics se calculan
Then metrics.close_reasons = {
  "close_sl": N1,
  "close_tp": N2,
  "close_trailing": N3,
  "close_time": N4
}
```

---

## Criterios de Regresion

### REG1: Backtest sin cambios de API
```
Given codigo de usuario que usa Backtest(start, end, assets, timeframe, capital)
When se ejecuta con nueva version
Then funciona sin modificaciones
  And retorna BacktestMetrics validas
```

### REG2: Metricas existentes no cambian
```
Given un backtest con mismos datos y configuracion
When se ejecuta antes y despues del cambio
Then total_trades, win_rate, sharpe_ratio son similares
  (diferencia aceptable por trailing stop vs hold constante)
```

### REG3: Signals se persisten correctamente
```
Given un backtest que genera signals
When backtest completa
Then Signal records en DB tienen todos los campos
  And link a Order es correcto
```

---

## Invariantes

- **Una posicion activa por simbolo**: No pueden existir dos ActivePosition con mismo symbol e is_active=True
- **SL < entry < TP para LONG**: stop_loss < entry_price < take_profit
- **SL > entry > TP para SHORT**: stop_loss > entry_price > take_profit
- **candles_since_entry monotonico**: Solo incrementa, nunca decrementa
- **close_reason siempre presente**: Si is_active=False, close_reason != None

---

## Oraculos de Validacion

### Reduccion de invocaciones
```bash
# Comparar logs antes/despues
grep "TradingGraph.invoke" backtest_old.log | wc -l  # N1
grep "TradingGraph.invoke" backtest_new.log | wc -l  # N2
# Reduccion = (N1 - N2) / N1 * 100
```

### Accuracy del paper
```python
# En Python
positions = session.query(ActivePosition).filter(
    ActivePosition.is_active == False
).all()

correct = sum(
    1 for p in positions
    for i, d in enumerate(p.candles_direction[:p.prediction_horizon])
    if (p.side == OrderSide.BUY and d == "up") or
       (p.side == OrderSide.SELL and d == "down")
)
total = sum(min(len(p.candles_direction), p.prediction_horizon) for p in positions)
mda = correct / total if total > 0 else 0
```

### Integridad de posiciones
```sql
-- No debe haber duplicados activos
SELECT symbol, COUNT(*)
FROM active_positions
WHERE is_active = TRUE
GROUP BY symbol
HAVING COUNT(*) > 1;
-- Debe retornar 0 filas
```

---

## Datos de Prueba

### Escenario: LONG con trailing stop exitoso
```
Entry: price=100, sl=95, tp=120, trailing=5%
Tick 1: price=102 -> highest=102, trailing_price=96.9
Tick 2: price=108 -> highest=108, trailing_price=102.6
Tick 3: price=105 -> CLOSE_TRAILING (105 < 102.6? No, continua)
Tick 4: price=101 -> CLOSE_TRAILING (101 < 102.6? Si)
Resultado: Profit = 1%, cerrado por trailing
```

### Escenario: SHORT con stop loss
```
Entry: price=100, sl=105, tp=90, trailing=5%
Tick 1: price=98 -> lowest=98, trailing_price=102.9
Tick 2: price=106 -> CLOSE_SL (106 > 105)
Resultado: Loss = -6%, cerrado por SL
```
