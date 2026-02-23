# Improve Error Handling in Position Reversal Flow

**Issue:** QuantAgent-2mu
**Type:** Enhancement
**Priority:** P2 (Medium)
**Date:** 2026-01-07
**Status:** Analysis Complete
**Dependencies:** QuantAgent-8vb (debe resolverse primero)

---

## 1. Resumen Ejecutivo

### Problema

El flujo de position reversal (proceso de dos pasos: close + open) no maneja errores adecuadamente. Cuando el paso 1 (close) falla, el error se registra pero no se propaga correctamente, dejando el sistema en un estado potencialmente inconsistente.

### Impacto

- **Severidad:** P2 (Enhancement)
- **Scope:** Robustez del sistema de position reversal
- **Risk:** Estado inconsistente del portfolio tras errores en reversal

### Estado Actual

- Errores se capturan y logean
- Errores **no se propagan** adecuadamente al caller
- El paso 2 (open nueva posición) podría ejecutarse aunque paso 1 falló
- Estado del portfolio puede quedar inconsistente

---

## 2. Contexto

### Flujo de Position Reversal

El position reversal es un proceso de **dos pasos**:

1. **Paso 1: Close existing position**
   - Crear orden de cierre (BUY para SHORT, SELL para LONG)
   - Ejecutar orden de cierre
   - Actualizar portfolio

2. **Paso 2: Open new position**
   - Crear orden de apertura (lado opuesto)
   - Ejecutar orden de apertura
   - Actualizar portfolio

**Problema:** Si paso 1 falla, paso 2 no debería ejecutarse.

### Configuración del Backtest

```
Symbol: BTC
Period: 2024-10-01 to 2024-12-31 (3 meses)
Timeframe: 1h
Initial Capital: $100,000
```

---

## 3. Análisis del Problema

### 3.1 Evidencia del Log

**Error durante reversal:**
```
2026-01-06 22:04:52,045 - quantagent.trading.order_manager - INFO - BTC: Position reversal detected - existing qty: -0.04807692307692308, new side: OrderSide.BUY
2026-01-06 22:04:52,045 - quantagent.trading.order_manager - ERROR - BTC: Close order portfolio update failed - Conversion 'ConversionSyntax' received SELL -0.04807692307692308 for attribute 'side'
```

**Observaciones:**
1. El error se registra como `ERROR`
2. No hay evidencia de que el reversal se abortó
3. No hay logging sobre el estado del portfolio después del error
4. No hay indicación de si el paso 2 se intentó ejecutar

### 3.2 Análisis del Código

**Ubicación:** `quantagent/trading/order_manager.py`, método `_execute_reversal()` (línea ~517)

**Problema 1: Error handling silencioso**

```python
try:
    # Paso 1: Close existing position
    close_result = self._execute_close_order(symbol, close_order)
except Exception as e:
    logger.error(f"{symbol}: Close order portfolio update failed - {e}")
    # ERROR: No se re-lanza la excepción
    # ERROR: No hay return/abort aquí
    # El código continúa...

# Paso 2: Open new position
# Este código podría ejecutarse aunque paso 1 falló!
open_result = self._execute_open_order(symbol, open_order)
```

**Problema 2: Falta de validación de estado**

No hay validación de que el paso 1 completó exitosamente antes de ejecutar paso 2:

```python
# No hay check como:
if not close_result or not close_result.success:
    logger.error(f"{symbol}: Reversal aborted - close failed")
    return None
```

**Problema 3: Logging insuficiente**

El logging no captura:
- Estado del portfolio antes del reversal
- Estado del portfolio después del error
- Si el reversal fue completado, abortado, o quedó en estado intermedio

### 3.3 Escenarios de Falla

| Escenario | Comportamiento Actual | Comportamiento Deseado |
|-----------|----------------------|------------------------|
| Paso 1 falla (close) | Error loggeado, paso 2 podría ejecutarse | Abort reversal, rollback |
| Paso 2 falla (open) | Error loggeado, posición cerrada pero no abierta | Rollback paso 1 o retry paso 2 |
| Error de red temporal | Falla inmediata | Retry con backoff |
| Error de validación | Error loggeado sin contexto | Error detallado con estado |

---

## 4. Impacto y Riesgos

### 4.1 Funcionalidades Afectadas

| Funcionalidad | Impacto |
|---------------|---------|
| Position reversals | ⚠️ Pueden quedar en estado inconsistente |
| Portfolio state | ⚠️ Puede desincronizarse tras errores |
| Error recovery | ❌ No existe mecanismo de rollback |
| Debugging | ⚠️ Difícil diagnosticar problemas |

### 4.2 Riesgos

| Riesgo | Severidad | Descripción |
|--------|-----------|-------------|
| Estado inconsistente | MEDIO | Portfolio puede quedar en estado intermedio |
| Pérdida de capital | BAJO | Posición puede quedar expuesta sin protección |
| Dificultad de debug | MEDIO | Errores difíciles de diagnosticar sin logging adecuado |

### 4.3 Dependencias

**Blocker:** QuantAgent-8vb debe resolverse primero

- El error `ConversionSyntax` (QuantAgent-8vb) es la causa raíz del error observado
- Una vez resuelto 8vb, podremos testear mejoras de error handling más fácilmente
- Los escenarios de error actuales están "contaminados" por el bug de 8vb

---

## 5. Solución Propuesta

### 5.1 Mejoras de Error Handling

**Archivo:** `quantagent/trading/order_manager.py`, método `_execute_reversal()`

**Propuesta 1: Propagación de errores**

```python
def _execute_reversal(self, symbol: str, signal: TradingSignal) -> Optional[ReversalResult]:
    """Execute position reversal with proper error handling."""

    # Validate initial state
    if symbol not in self.portfolio.positions:
        logger.error(f"{symbol}: Cannot reverse - no existing position")
        return None

    existing_qty = self.portfolio.positions[symbol]["qty"]

    # Log initial state
    logger.info(
        f"{symbol}: Starting reversal - "
        f"existing_qty={existing_qty}, new_side={signal.side}, "
        f"portfolio_value=${self.portfolio.total_value:.2f}"
    )

    try:
        # Paso 1: Close existing position
        close_result = self._execute_close_order(symbol, close_order)

        if not close_result or not close_result.success:
            logger.error(
                f"{symbol}: Reversal ABORTED - close order failed. "
                f"Portfolio state unchanged."
            )
            return ReversalResult(
                success=False,
                step_completed=1,
                error="Close order failed",
                portfolio_state=self._get_portfolio_snapshot()
            )

        logger.info(f"{symbol}: Reversal step 1/2 complete - position closed")

        # Paso 2: Open new position
        open_result = self._execute_open_order(symbol, open_order)

        if not open_result or not open_result.success:
            logger.error(
                f"{symbol}: Reversal PARTIAL FAILURE - close succeeded but open failed. "
                f"Position is FLAT (no exposure)."
            )
            return ReversalResult(
                success=False,
                step_completed=2,
                error="Open order failed after close succeeded",
                portfolio_state=self._get_portfolio_snapshot()
            )

        logger.info(
            f"{symbol}: Reversal COMPLETE - "
            f"new_qty={self.portfolio.positions[symbol]['qty']}, "
            f"portfolio_value=${self.portfolio.total_value:.2f}"
        )

        return ReversalResult(success=True, step_completed=2)

    except Exception as e:
        logger.exception(
            f"{symbol}: Reversal EXCEPTION - {type(e).__name__}: {str(e)}"
        )
        return ReversalResult(
            success=False,
            step_completed=0,
            error=f"Exception: {str(e)}",
            portfolio_state=self._get_portfolio_snapshot()
        )
```

**Propuesta 2: Validaciones pre-reversal**

```python
def _validate_reversal_preconditions(self, symbol: str, signal: TradingSignal) -> bool:
    """Validate that reversal can proceed."""

    # Check 1: Position exists
    if symbol not in self.portfolio.positions:
        logger.error(f"{symbol}: No position to reverse")
        return False

    # Check 2: Sufficient portfolio value
    if self.portfolio.total_value <= 0:
        logger.error(f"{symbol}: Portfolio value is zero, cannot reverse")
        return False

    # Check 3: Valid signal side
    if signal.side not in [OrderSide.BUY, OrderSide.SELL]:
        logger.error(f"{symbol}: Invalid signal side: {signal.side}")
        return False

    return True
```

**Propuesta 3: Snapshot de estado**

```python
def _get_portfolio_snapshot(self) -> Dict[str, Any]:
    """Capture current portfolio state for error recovery."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "positions": dict(self.portfolio.positions),
        "cash": float(self.portfolio.cash),
        "total_value": float(self.portfolio.total_value),
    }
```

### 5.2 Mejoras de Logging

**Structured logging con contexto completo:**

```python
logger.info(
    f"{symbol}: Reversal event",
    extra={
        "event": "reversal_start",
        "symbol": symbol,
        "existing_qty": existing_qty,
        "new_side": signal.side.name,
        "portfolio_value": self.portfolio.total_value,
    }
)
```

### 5.3 Retry Logic (Opcional)

Para errores transitorios (red, API rate limits):

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def _execute_close_order_with_retry(self, symbol: str, order: Order):
    """Execute close order with retry on transient errors."""
    return self._execute_close_order(symbol, order)
```

---

## 6. Tests Requeridos

### 6.1 Test: Reversal con Close Failure

```python
def test_reversal_aborts_on_close_failure():
    """Test que reversal se aborta si close order falla."""
    # Setup: Posición existente
    portfolio.positions["BTC"] = {"qty": -0.048, "avg_cost": 96382.0}
    initial_state = portfolio.get_state()

    # Mock: Forzar que close order falle
    with patch.object(order_manager, '_execute_close_order', return_value=None):
        # Action: Intentar reversal
        result = order_manager._execute_reversal("BTC", buy_signal)

    # Assert
    assert result is not None
    assert result.success is False
    assert result.step_completed == 1
    assert result.error == "Close order failed"

    # Assert: Portfolio state unchanged
    final_state = portfolio.get_state()
    assert final_state == initial_state
```

### 6.2 Test: Reversal con Open Failure

```python
def test_reversal_partial_on_open_failure():
    """Test que reversal maneja correctamente fallo en open order."""
    # Setup: Posición existente
    portfolio.positions["BTC"] = {"qty": -0.048, "avg_cost": 96382.0}

    # Mock: Close exitoso, open falla
    with patch.object(order_manager, '_execute_close_order', return_value=SuccessResult()):
        with patch.object(order_manager, '_execute_open_order', return_value=None):
            # Action
            result = order_manager._execute_reversal("BTC", buy_signal)

    # Assert
    assert result is not None
    assert result.success is False
    assert result.step_completed == 2
    assert "open failed" in result.error.lower()

    # Assert: Position is FLAT (closed but not reopened)
    assert portfolio.positions["BTC"]["qty"] == 0
```

### 6.3 Test: Logging Completo

```python
def test_reversal_logging(caplog):
    """Test que reversal genera logging completo."""
    # Setup
    portfolio.positions["BTC"] = {"qty": -0.048, "avg_cost": 96382.0}

    # Action
    with caplog.at_level(logging.INFO):
        result = order_manager._execute_reversal("BTC", buy_signal)

    # Assert: Logging events presentes
    log_messages = [record.message for record in caplog.records]

    assert any("Starting reversal" in msg for msg in log_messages)
    assert any("step 1/2 complete" in msg for msg in log_messages)
    assert any("COMPLETE" in msg or "FAILED" in msg for msg in log_messages)
```

### 6.4 Test: Validación de Precondiciones

```python
def test_reversal_validates_preconditions():
    """Test que reversal valida condiciones antes de ejecutar."""
    # Setup: No position exists
    portfolio.positions = {}

    # Action
    result = order_manager._execute_reversal("BTC", buy_signal)

    # Assert
    assert result is not None
    assert result.success is False
    assert "no position" in result.error.lower()
```

### 6.5 Criterios de Éxito

| Test | Criterio |
|------|----------|
| Close failure | Reversal abortado, estado sin cambios |
| Open failure | Position cerrada (FLAT), error reportado |
| Logging | Eventos clave registrados |
| Preconditions | Validaciones ejecutadas |
| Exception handling | Excepciones capturadas y loggeadas |

---

## 7. Validación

### 7.1 Simulación de Errores

Crear test scenarios que fuercen diferentes tipos de errores:

```python
# Error de conversión (como ConversionSyntax)
# Error de red (timeout)
# Error de validación (invalid order)
# Error de estado (position not found)
```

### 7.2 Backtest con Errores Inyectados

Ejecutar backtest con inyección de errores aleatorios:

```python
# Configurar error injection rate
error_injection_rate = 0.1  # 10% de reversals fallarán

# Run backtest
metrics = backtest.run(
    name="Error Injection Test",
    error_injection=error_injection_rate
)

# Validar que sistema se recupera correctamente
assert backtest.portfolio_state_is_consistent()
```

### 7.3 Criterios de Aceptación

- [ ] Errores en paso 1 abortan reversal completamente
- [ ] Errores en paso 2 reportados con estado de portfolio
- [ ] Logging estructurado captura eventos clave
- [ ] No hay estados inconsistentes tras errores
- [ ] Portfolio value correcto después de errores

---

## 8. Referencias

### 8.1 Issues Relacionados

- **QuantAgent-8vb**: Fix ConversionSyntax error (blocker)
- **QuantAgent-2mu**: Improve error handling (este issue)

### 8.2 Archivos Afectados

| Archivo | Ubicación | Cambio |
|---------|-----------|--------|
| `quantagent/trading/order_manager.py` | Método `_execute_reversal()` | Mejoras de error handling |

### 8.3 Documentación Relacionada

- [Position Reversal Implementation](/docs/06_implementation/QuantAgent-g3c-IM-position-reversal-fix.md)
- [Error Handling Best Practices](/docs/development/error-handling.md)

---

## Appendix A: ReversalResult Model

Propuesta para structured result object:

```python
@dataclass
class ReversalResult:
    """Result of position reversal operation."""

    success: bool
    step_completed: int  # 0=none, 1=close only, 2=both
    error: Optional[str] = None
    portfolio_state: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        if self.success:
            return "Reversal completed successfully"
        else:
            return f"Reversal failed at step {self.step_completed}: {self.error}"
```

---

## Appendix B: Logging Levels

**Guía de logging para position reversal:**

| Evento | Level | Ejemplo |
|--------|-------|---------|
| Reversal iniciado | INFO | `Starting reversal - existing_qty=-0.048` |
| Paso completado | INFO | `Reversal step 1/2 complete` |
| Reversal exitoso | INFO | `Reversal COMPLETE - new_qty=0.05` |
| Reversal abortado | ERROR | `Reversal ABORTED - close order failed` |
| Fallo parcial | ERROR | `Reversal PARTIAL FAILURE - position FLAT` |
| Excepción | EXCEPTION | `Reversal EXCEPTION - ConversionSyntax: ...` |

---

## Appendix C: Portfolio State Consistency Checks

Validaciones para asegurar consistencia de estado:

```python
def validate_portfolio_consistency(self) -> List[str]:
    """Validate portfolio state consistency."""
    issues = []

    # Check 1: Positions with zero quantity
    for symbol, position in self.positions.items():
        if position["qty"] == 0:
            issues.append(f"{symbol}: Position with qty=0 should be removed")

    # Check 2: Cash + position value = total value
    calculated_value = self.cash + sum(
        pos["qty"] * pos["current_price"]
        for pos in self.positions.values()
    )
    if abs(calculated_value - self.total_value) > 0.01:
        issues.append(
            f"Total value mismatch: calculated={calculated_value}, "
            f"stored={self.total_value}"
        )

    # Check 3: Negative cash
    if self.cash < 0:
        issues.append(f"Negative cash: {self.cash}")

    return issues
```
