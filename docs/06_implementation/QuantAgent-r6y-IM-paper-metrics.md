# Implementation Notes: Paper Metrics + Validation

**Issue ID**: QuantAgent-r6y  
**Type**: Task  
**Epic**: QuantAgent-nu7 (Active Position Monitoring System)  
**Branch**: `feature/QuantAgent-nu7-active-position-monitoring`  
**Commit**: `2d1b726`  
**Date**: 2026-01-11

---

## Objetivo

Implementar Mean Directional Accuracy (MDA), tracking de invocaciones de agentes, y métricas de razones de cierre para validar la reducción >= 80% en invocaciones LLM según el paper arXiv:2509.09995.

---

## Cambios Realizados

### 1. BacktestMetrics Extendido

**Archivo**: `quantagent/backtesting/backtest.py` (líneas ~40-70)

Agregados 6 nuevos campos al dataclass:

```python
@dataclass
class BacktestMetrics:
    # ... campos existentes ...
    
    # Phase 4: Active Position Monitoring metrics
    agent_invocations: int = 0
    invocations_saved: int = 0
    invocation_reduction_pct: float = 0.0
    mean_directional_accuracy: float = 0.0
    accuracy_by_candle: Dict[int, float] = None
    close_reasons: Dict[str, int] = None
    
    def __post_init__(self):
        if self.accuracy_by_candle is None:
            self.accuracy_by_candle = {}
        if self.close_reasons is None:
            self.close_reasons = {}
```

**Rationale**: Estos campos son necesarios para validar el AC4.1 - AC4.5 del issue QuantAgent-nu7.

---

### 2. Tracking de Invocaciones

**Archivo**: `quantagent/backtesting/backtest.py`

#### 2.1 Inicialización de contadores (líneas ~175-180)

```python
# Phase 4: Invocation tracking
self.agent_invocations = 0
self.total_candles_processed = 0
```

#### 2.2 Tracking en _analyze_and_trade (líneas ~355-400)

```python
# Count total candles processed
self.total_candles_processed += 1

if active_pos:
    # ... exit logic ...
    if not should_exit:
        # Position still active: NO INVOKE
        # ... update tracking ...
        return  # Early return - invocation saved

# No active position: generate signal
# Count agent invocations
self.agent_invocations += 1

signal = self.strategy.generate_signal(...)
```

**Rationale**: 
- `total_candles_processed` cuenta cada tick analizado (base para calcular reducción)
- `agent_invocations` solo incrementa cuando realmente se invoca `strategy.generate_signal()`
- Early return cuando posición activa → invocación ahorrada

---

### 3. Cálculo de Mean Directional Accuracy

**Método**: `_calculate_directional_accuracy()` (líneas ~790-845)

```python
def _calculate_directional_accuracy(self) -> tuple[float, Dict[int, float]]:
    """
    Calculate Mean Directional Accuracy (MDA) and per-candle accuracy.
    
    MDA = (correct_candles / total_candles_evaluated)
    """
    positions = (
        self.db.query(ActivePosition)
        .filter(
            ActivePosition.is_active.is_(False),
            ActivePosition.decision_timestamp >= self.start_date,
            ActivePosition.decision_timestamp <= self.end_date,
        )
        .all()
    )
    
    # Track correct predictions per candle index
    correct_by_candle = {}
    total_by_candle = {}
    
    for pos in positions:
        expected_direction = "up" if pos.side == OrderSide.BUY else "down"
        
        # Evaluate up to prediction_horizon candles
        for i, direction in enumerate(pos.candles_direction[:pos.prediction_horizon]):
            candle_idx = i + 1  # 1-indexed
            
            # ... accumulate stats ...
            if direction == expected_direction:
                correct_by_candle[candle_idx] += 1
    
    # Calculate accuracy per candle
    accuracy_by_candle = {
        candle: correct / total
        for candle in sorted(total_by_candle.keys())
    }
    
    # Calculate overall MDA
    mda = total_correct / total_candles if total_candles > 0 else 0.0
    
    return mda, accuracy_by_candle
```

**Lógica**:
1. Query todas las ActivePosition cerradas en el rango de fechas del backtest
2. Para cada posición:
   - Determinar dirección esperada: "up" si LONG, "down" si SHORT
   - Evaluar hasta `prediction_horizon` candles (típicamente 3)
   - Contar aciertos y totales por índice de candle (1, 2, 3...)
3. Calcular accuracy por candle: `correct / total` para cada índice
4. Calcular MDA global: suma de aciertos / suma de totales

**Rationale**: Según AC4.1 y AC4.2, MDA mide qué tan bien el modelo predice la dirección del precio en los N candles siguientes.

---

### 4. Cálculo de Close Reasons

**Método**: `_calculate_close_reasons()` (líneas ~847-869)

```python
def _calculate_close_reasons(self) -> Dict[str, int]:
    """
    Calculate distribution of position close reasons.
    
    Returns:
        Dict mapping close_reason to count
    """
    positions = (
        self.db.query(ActivePosition)
        .filter(
            ActivePosition.is_active.is_(False),
            ActivePosition.decision_timestamp >= self.start_date,
            ActivePosition.decision_timestamp <= self.end_date,
        )
        .all()
    )
    
    close_reasons = {}
    for pos in positions:
        reason = pos.close_reason or "unknown"
        close_reasons[reason] = close_reasons.get(reason, 0) + 1
    
    return close_reasons
```

**Rationale**: AC4.5 requiere tracking de razones de cierre. Útil para debugging y análisis de comportamiento del sistema.

---

### 5. Integración en _calculate_metrics

**Archivo**: `quantagent/backtesting/backtest.py` (líneas ~695-735)

```python
def _calculate_metrics(self) -> BacktestMetrics:
    # ... cálculos existentes ...
    
    # Phase 4: Calculate MDA and accuracy metrics
    mda, accuracy_by_candle = self._calculate_directional_accuracy()
    
    # Phase 4: Calculate close reasons distribution
    close_reasons = self._calculate_close_reasons()
    
    # Phase 4: Calculate invocation reduction
    invocations_saved = self.total_candles_processed - self.agent_invocations
    invocation_reduction_pct = (
        (invocations_saved / self.total_candles_processed * 100)
        if self.total_candles_processed > 0
        else 0.0
    )
    
    return BacktestMetrics(
        # ... campos existentes ...
        agent_invocations=self.agent_invocations,
        invocations_saved=invocations_saved,
        invocation_reduction_pct=invocation_reduction_pct,
        mean_directional_accuracy=mda,
        accuracy_by_candle=accuracy_by_candle,
        close_reasons=close_reasons,
    )
```

**Caso especial**: El branch de "no trades" también calcula métricas Phase 4 (líneas ~640-675), porque puede haber posiciones abiertas sin trades completos.

---

## Dependencias

### Asumidas (implementadas en fases anteriores)
- `ActivePosition` modelo existe y tiene campos:
  - `candles_direction` (JSON list)
  - `prediction_horizon` (int)
  - `close_reason` (string)
  - `side` (OrderSide enum)
  - `is_active` (bool)
  - `decision_timestamp` (datetime)
- `PositionMonitor.update_candle_tracking()` popula correctamente `candles_direction`
- `PositionMonitor.close_position()` setea correctamente `close_reason`

---

## Cómo Testear

### 1. Smoke Test (sintaxis)

```bash
cd /mnt/c/Users/BAISCF/repos_local/QuantAgent/.worktrees/qa-nu7
source .venv_wsl/bin/activate
python3 -m py_compile quantagent/backtesting/backtest.py
```

### 2. Unit Tests Existentes

```bash
pytest tests/test_backtest.py::TestBacktest::test_calculate_metrics_with_no_trades -v
pytest tests/test_backtest.py -k "metrics" -v
```

**Nota**: Algunos tests pueden requerir ajustes para validar los nuevos campos.

### 3. Integration Test (backtest completo)

```bash
# Ejecutar backtest de ejemplo
python3 examples/run_backtest.py

# O crear script de validación:
python3 -c "
from quantagent.backtesting.backtest import Backtest
from datetime import datetime

bt = Backtest(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 20),
    assets=['BTC', 'ETH'],
    timeframe='1h',
)

metrics = bt.run()

print(f'Agent Invocations: {metrics.agent_invocations}')
print(f'Total Candles: {metrics.agent_invocations + metrics.invocations_saved}')
print(f'Reduction: {metrics.invocation_reduction_pct:.1f}%')
print(f'MDA: {metrics.mean_directional_accuracy:.3f}')
print(f'Accuracy by candle: {metrics.accuracy_by_candle}')
print(f'Close reasons: {metrics.close_reasons}')
"
```

### 4. Validación AC4.4 (reducción >= 80%)

```bash
# Backtest de 20 días con 2 assets (según AC)
# Configuración con exit_policy = TRAILING_STOP
# Verificar: metrics.invocation_reduction_pct >= 80.0
```

### 5. Validación Oracle de Accuracy (AC4.2)

```sql
-- Verificar cálculo manual de MDA
SELECT 
    ap.id,
    ap.side,
    ap.candles_direction,
    ap.prediction_horizon,
    ap.accuracy
FROM active_positions ap
WHERE ap.is_active = 0
  AND ap.decision_timestamp >= '2024-01-01'
  AND ap.decision_timestamp <= '2024-01-20';
```

Comparar con cálculo Python manual según oracle en acceptance tests.

---

## Quality Gates Ejecutados

✅ **Pasaron**:
- `python3 -m py_compile` → OK (syntax check)
- `black quantagent/backtesting/backtest.py` → reformateado y aplicado
- `isort quantagent/backtesting/backtest.py` → reformateado y aplicado

⚠️ **Warnings no bloqueantes**:
- `flake8` → E501 (líneas largas pre-existentes), F401 (imports no usados pre-existentes)

❌ **No completados** (timeouts en entorno WSL):
- `mypy` → timeout >30s
- `pytest -k backtest` → 24/39 tests pasaron, luego timeout >120s

**Recomendación**: Ejecutar en entorno más rápido o con subset específico de tests.

---

## Archivos Modificados

- `quantagent/backtesting/backtest.py` (+144 líneas, -9 líneas)

**Imports agregados**:
- `ActivePosition` desde `quantagent.models`

---

## Riesgos y Limitaciones

1. **Asunción de datos correctos**: El cálculo de MDA asume que `ActivePosition.candles_direction` está correctamente populado por `PositionMonitor`. No se valida integridad aquí.

2. **Performance**: Query de todas las ActivePosition cerradas en _calculate_metrics puede ser lento si hay miles de posiciones. Considerar paginación o agregación SQL si es problema.

3. **Edge cases**:
   - Si `prediction_horizon` varía entre posiciones, `accuracy_by_candle` puede tener índices desiguales (correcto, pero podría confundir)
   - Si una posición cierra antes de alcanzar `prediction_horizon`, solo evaluará los candles disponibles

4. **No validado empíricamente**: Falta ejecutar backtest real de 20 días para confirmar reducción >= 80% (AC4.4).

---

## Próximos Pasos (para Tester)

### Tests a escribir

1. **Test de `_calculate_directional_accuracy`**:
   - Mock 3 ActivePosition con candles_direction conocidos
   - Verificar MDA y accuracy_by_candle
   - Casos: 100% accuracy, 0% accuracy, mixto

2. **Test de `_calculate_close_reasons`**:
   - Mock posiciones con diferentes close_reason
   - Verificar distribución correcta

3. **Test de tracking de invocaciones**:
   - Mock backtest con 10 candles, 2 posiciones activas que duran 3 candles cada una
   - Verificar: agent_invocations = 4, invocations_saved = 6, reduction = 60%

4. **Test de integración end-to-end**:
   - Backtest completo con datos reales
   - Validar todos los campos de Phase 4 están presentes y tienen sentido

### Validaciones manuales

1. Ejecutar backtest de 20 días con configuración del paper
2. Verificar `invocation_reduction_pct >= 80%`
3. Verificar `mean_directional_accuracy` está en rango razonable (0.45 - 0.65 según literatura)
4. Documentar resultados en este archivo (sección "Resultados Empíricos")

---

## Referencias

- **AC**: `docs/05_acceptance_tests/QuantAgent-nu7-AC-active-position-monitoring.md` (Fase 4)
- **Design**: `docs/03_design/QuantAgent-nu7-DS-active-position-monitoring.md` (sección BacktestMetrics Extendido)
- **Paper**: arXiv:2509.09995 - "Price-Driven Multi-Agent LLMs for High-Frequency Trading"

---

## Resultados Empíricos

_Pendiente: ejecutar por tester/humano_

### Backtest: 2024-01-01 to 2024-01-20 (20 días, BTC+ETH, 1h)

```
(Placeholder para resultados)

Agent Invocations: ?
Total Candles: ?
Invocations Saved: ?
Reduction: ?%
MDA: ?
Accuracy by Candle: {1: ?, 2: ?, 3: ?}
Close Reasons: {...}

✅/❌ AC4.4: Reducción >= 80%
```

---

**Implementado por**: Implementer Agent  
**Fecha**: 2026-01-11  
**Commit**: `2d1b726`
