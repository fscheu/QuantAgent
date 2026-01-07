# Calculate P&L for Trade Objects When Closing Positions

**Issue:** QuantAgent-r78
**Type:** Bug
**Priority:** P1 (High)
**Date:** 2026-01-07
**Status:** Analysis Complete

---

## 1. Resumen Ejecutivo

### Problema

Los objetos `Trade` creados cuando se cierran posiciones no calculan los campos `pnl` (profit/loss) y `pnl_pct` (porcentaje de ganancia/pérdida), resultando en valores `None` o `$0.00`. Esto invalida todas las métricas del backtest (win rate, profit factor, total P&L, etc.).

### Impacto

- **Severidad:** P1 (High)
- **Scope:** Todas las métricas de backtest son inválidas
- **Risk:** Imposible evaluar estrategias de trading correctamente

### Estado Actual

- Trades se crean correctamente con `entry_price` y `exit_price`
- Campo `pnl` nunca se calcula → queda como `None`
- Métricas muestran: Total P&L = $0.00, Win Rate = 0%, Winning/Losing Trades = 0

---

## 2. Contexto

### Configuración del Backtest

```
Symbol: BTC
Period: 2024-10-01 to 2024-12-31 (3 meses)
Timeframe: 1h
Initial Capital: $100,000
Model: openai/gpt-4o-mini
Slippage: 1%
```

### Observación

El backtest reporta 4 trades ejecutados, pero:
- `Winning Trades: 0`
- `Losing Trades: 0`
- `Total P&L: $0.00`

Esto es inconsistente con un backtest de 3 meses donde hubo actividad de trading.

---

## 3. Análisis del Problema

### 3.1 Evidencia del Log

**Resultados del Backtest:**
```
============================================================
BACKTEST RESULTS
============================================================
Total Trades:      4
Winning Trades:    0
Losing Trades:     0
Win Rate:          0.00%
Profit Factor:     0.00
Sharpe Ratio:      -1.24
Max Drawdown:      13.63%
Total P&L:         $0.00
Total Return:      0.00%
Average Win:       $0.00
Average Loss:      $0.00
Largest Win:       $0.00
Largest Loss:      $0.00
============================================================
```

**Observación:** Aunque `Total Trades: 4`, todas las métricas derivadas de P&L son cero.

### 3.2 Root Cause

**Ubicación:** `quantagent/portfolio/manager.py`, método `execute_trade()`, líneas ~129-140

**Código actual:**
```python
trade = Trade(
    symbol=symbol,
    order_id=order.id,
    entry_price=entry_price,
    exit_price=exit_price,
    quantity=Decimal(str(fill_qty)),
    side=order.side,
    commission=Decimal(str(0)),
    environment=self.environment,
    opened_at=opened_at or datetime.utcnow(),
    closed_at=closed_at,
)
# FALTA: trade.pnl = calculated_pnl
# FALTA: trade.pnl_pct = calculated_pnl_pct
```

**Problema:**
- El objeto `Trade` se crea con `entry_price` y `exit_price`
- Los campos `pnl` y `pnl_pct` nunca se calculan
- Al persistir en DB, quedan como `None`

### 3.3 Impacto en Métricas

**Archivo:** `quantagent/backtesting/backtest.py`, método `_calculate_metrics()`, líneas ~537-538

```python
winning_trades = [t for t in trades if t.pnl and float(t.pnl) > 0]
losing_trades = [t for t in trades if t.pnl and float(t.pnl) < 0]
```

Como `t.pnl` es `None` para todos los trades:
- `winning_trades = []` (lista vacía)
- `losing_trades = []` (lista vacía)
- `total_pnl = sum([t.pnl for t in trades])` = suma de valores `None` = `0.00`

Esto explica por qué todas las métricas son cero.

### 3.4 Drawdown vs P&L

**Nota importante:** El drawdown reportado (13.63%) **es real** pero no proviene de P&L de trades:

- El drawdown se calcula desde `equity_curve` (portfolio value over time)
- El portfolio value cambia por unrealized P&L (posiciones abiertas)
- Pero el P&L **realizado** nunca se registra en `Trade.pnl`

El drawdown representa variación de valor por precios de mercado, **NO** ganancias/pérdidas de trades cerrados.

---

## 4. Impacto y Riesgos

### 4.1 Funcionalidades Afectadas

| Funcionalidad | Impacto |
|---------------|---------|
| Métricas de backtest | ❌ Todas inválidas (win rate, profit factor, etc.) |
| Evaluación de estrategias | ❌ Imposible comparar estrategias |
| Audit trail | ⚠️ Trades registrados pero sin P&L |
| Reportes | ❌ Reportes muestran $0.00 en todos los trades |

### 4.2 Riesgos

| Riesgo | Severidad | Descripción |
|--------|-----------|-------------|
| Decisiones erróneas | ALTO | Estrategias se evalúan sin datos de P&L |
| Pérdida de confianza | MEDIO | Usuarios no confían en resultados de backtest |
| Tiempo desperdiciado | MEDIO | Backtests inválidos deben re-ejecutarse |

### 4.3 Impacto en Desarrollo

**Bloqueadores:**
- No se pueden validar estrategias de trading
- Tests de integración que validan P&L fallan
- Imposible demostrar que el sistema funciona correctamente

---

## 5. Solución Propuesta

### 5.1 Corrección

**Archivo:** `quantagent/portfolio/manager.py`, método `execute_trade()`

**Código propuesto:**
```python
# Calculate P&L for closing trades
pnl = None
pnl_pct = None

if is_closing_long or is_closing_short:
    if entry_price and exit_price and fill_qty:
        # Convert to float for calculation
        entry = float(entry_price)
        exit = float(exit_price)
        qty = float(fill_qty)

        if is_closing_long:
            # LONG: profit = (exit - entry) * qty
            pnl = (exit - entry) * qty
        elif is_closing_short:
            # SHORT: profit = (entry - exit) * qty
            pnl = (entry - exit) * qty

        # Calculate P&L percentage
        if entry > 0:
            pnl_pct = (pnl / (entry * qty)) * 100

trade = Trade(
    symbol=symbol,
    order_id=order.id,
    entry_price=entry_price,
    exit_price=exit_price,
    quantity=Decimal(str(fill_qty)),
    side=order.side,
    pnl=Decimal(str(pnl)) if pnl is not None else None,  # NUEVO
    pnl_pct=pnl_pct,  # NUEVO
    commission=Decimal(str(0)),
    environment=self.environment,
    opened_at=opened_at or datetime.utcnow(),
    closed_at=closed_at,
)
```

### 5.2 Validaciones

Agregar validaciones para casos edge:

```python
# Validar que tenemos datos necesarios
if not entry_price or not exit_price:
    logger.warning(
        f"{symbol}: Cannot calculate P&L - missing prices "
        f"(entry={entry_price}, exit={exit_price})"
    )

# Validar que no hay división por cero
if entry_price <= 0:
    logger.warning(f"{symbol}: Invalid entry_price for P&L calculation: {entry_price}")
```

### 5.3 Logging

Agregar logging para debug:

```python
if pnl is not None:
    logger.info(
        f"{symbol}: Trade P&L calculated - "
        f"pnl=${pnl:.2f}, pnl_pct={pnl_pct:.2f}%, "
        f"side={order.side}, qty={fill_qty}"
    )
```

---

## 6. Tests Requeridos

### 6.1 Test Unitario: P&L para LONG

```python
def test_pnl_calculation_long():
    """Test que P&L se calcula correctamente para posiciones LONG."""
    # Setup
    entry_price = Decimal("60000.00")
    exit_price = Decimal("65000.00")
    quantity = Decimal("0.1")

    # Expected P&L: (65000 - 60000) * 0.1 = 500.00
    expected_pnl = 500.00
    expected_pnl_pct = (5000 / 6000) * 100  # 83.33%

    # Action
    trade = create_closing_trade_long(
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity
    )

    # Assert
    assert trade.pnl is not None
    assert float(trade.pnl) == expected_pnl
    assert abs(trade.pnl_pct - expected_pnl_pct) < 0.01
```

### 6.2 Test Unitario: P&L para SHORT

```python
def test_pnl_calculation_short():
    """Test que P&L se calcula correctamente para posiciones SHORT."""
    # Setup
    entry_price = Decimal("65000.00")
    exit_price = Decimal("60000.00")
    quantity = Decimal("0.1")

    # Expected P&L: (65000 - 60000) * 0.1 = 500.00
    expected_pnl = 500.00
    expected_pnl_pct = (5000 / 6500) * 100  # 76.92%

    # Action
    trade = create_closing_trade_short(
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity
    )

    # Assert
    assert trade.pnl is not None
    assert float(trade.pnl) == expected_pnl
    assert abs(trade.pnl_pct - expected_pnl_pct) < 0.01
```

### 6.3 Test Integración: Métricas de Backtest

```python
def test_backtest_metrics_with_pnl():
    """Test que métricas de backtest se calculan correctamente con P&L."""
    # Setup: Ejecutar backtest corto
    backtest = Backtest(
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 10, 7),  # 1 semana
        assets=["BTC"],
        timeframe="1h",
        initial_capital=100000.0
    )

    # Action
    metrics = backtest.run()

    # Assert
    assert metrics.total_trades > 0
    assert metrics.total_pnl != 0.00  # CRÍTICO: No debe ser cero
    assert metrics.winning_trades + metrics.losing_trades > 0
    assert metrics.win_rate >= 0.0 and metrics.win_rate <= 1.0
```

### 6.4 Criterios de Éxito

| Test | Criterio |
|------|----------|
| P&L calculado | `trade.pnl is not None` |
| P&L correcto (LONG) | `pnl = (exit - entry) * qty` |
| P&L correcto (SHORT) | `pnl = (entry - exit) * qty` |
| P&L % correcto | `pnl_pct` en rango razonable |
| Métricas válidas | `total_pnl != 0.00` si hay trades |

---

## 7. Validación

### 7.1 Backtest Regression Test

Re-ejecutar el mismo backtest después del fix:

```bash
python examples/run_backtest.py
```

**Expectativas:**
- ✅ `Total P&L != $0.00`
- ✅ `Winning Trades + Losing Trades > 0`
- ✅ `Win Rate` calculado correctamente (0% < rate < 100%)
- ✅ `Profit Factor` calculado (no 0.00)
- ✅ `Average Win` y `Average Loss` != $0.00

### 7.2 Criterios de Aceptación

- [ ] Todos los trades tienen `pnl` calculado
- [ ] Métricas de backtest son != 0.00
- [ ] Winning trades y losing trades clasificados correctamente
- [ ] Log muestra: `Trade P&L calculated - pnl=$XXX.XX`

### 7.3 Validación Manual

Revisar trades en base de datos:

```sql
SELECT
    symbol,
    entry_price,
    exit_price,
    quantity,
    pnl,
    pnl_pct,
    side
FROM trades
WHERE closed_at IS NOT NULL
ORDER BY closed_at DESC
LIMIT 10;
```

**Expectativa:** Todos los trades cerrados deben tener `pnl` y `pnl_pct` != NULL.

---

## 8. Referencias

### 8.1 Issue Relacionado

- **QuantAgent-r78**: Calculate P&L for Trade objects when closing positions

### 8.2 Archivos Afectados

| Archivo | Ubicación | Cambio |
|---------|-----------|--------|
| `quantagent/portfolio/manager.py` | Líneas 129-140 | Agregar cálculo de P&L |

### 8.3 Documentación Relacionada

- [Trade Model](/docs/database/models.md#trade-model)
- [Portfolio Management](/docs/03_design/POSITION_MANAGEMENT_STRATEGIES.md)
- [Backtest Metrics](/docs/backtesting/metrics.md)

---

## Appendix A: Evidencia del Log

### Métricas Inválidas

```
============================================================
BACKTEST RESULTS
============================================================
Total Trades:      4
Winning Trades:    0          ← Debería ser > 0
Losing Trades:     0          ← Debería ser > 0
Win Rate:          0.00%      ← Inválido
Profit Factor:     0.00       ← Inválido
Total P&L:         $0.00      ← PROBLEMA: debería != 0
Average Win:       $0.00      ← Derivado de P&L
Average Loss:      $0.00      ← Derivado de P&L
Largest Win:       $0.00      ← Derivado de P&L
Largest Loss:      $0.00      ← Derivado de P&L
============================================================
```

### Drawdown Real pero P&L Cero

```
Max Drawdown:      13.63%     ← Real (calculado de equity curve)
Total P&L:         $0.00      ← Bug (no calculado en trades)
```

Esta inconsistencia confirma que el portfolio value cambia (drawdown existe) pero el P&L de trades cerrados no se registra.

---

## Appendix B: Fórmulas de P&L

### LONG Position

```
P&L = (Exit Price - Entry Price) × Quantity
P&L % = (P&L / (Entry Price × Quantity)) × 100
```

**Ejemplo:**
- Entry: $60,000
- Exit: $65,000
- Quantity: 0.1 BTC
- P&L = ($65,000 - $60,000) × 0.1 = **$500.00**
- P&L % = ($500 / ($60,000 × 0.1)) × 100 = **8.33%**

### SHORT Position

```
P&L = (Entry Price - Exit Price) × Quantity
P&L % = (P&L / (Entry Price × Quantity)) × 100
```

**Ejemplo:**
- Entry: $65,000
- Exit: $60,000
- Quantity: 0.1 BTC
- P&L = ($65,000 - $60,000) × 0.1 = **$500.00**
- P&L % = ($500 / ($65,000 × 0.1)) × 100 = **7.69%**
