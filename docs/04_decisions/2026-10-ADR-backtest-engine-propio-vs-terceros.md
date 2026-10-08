# ADR: Engine de backtesting propio vs. alternativas de terceros

**Issue:** QuantAgent-lcv · **Estado:** Propuesta (decisión de Fede) · **Fecha:** 2026-10-08

## Contexto y Evidencia

QuantAgent-832 y QuantAgent-hx0 cerraron la auditoría de métricas y la cross-validation del motor:
- **Auditoría de métricas (`QuantAgent-hx0`):** PnL (hx0.1), win rate y profit factor (hx0.2), max drawdown (hx0.3) y Sharpe ratio (hx0.4) auditados y validados contra recálculos matemáticos independientes (`scripts/recalc_metrics.py`).
- **Cross-validation (`QuantAgent-832`):** `docs/05_acceptance_tests/QuantAgent-832-AC-crossval-trade-por-trade.md` demostró coincidencia trade por trade exacta entre el motor propio y `backtesting.py` evaluados al Close (227 trades idénticos, PnL 13084.979, equity con diferencia máxima 5.79e-05 USD).
- **Experimento intravela (`QuantAgent-832`):** Evaluar SL/TP contra High/Low dentro de la vela (`backtesting.py --intrabar`) produce 340 trades, PnL 5803.02 y win rate 33.53% (vs. 227 trades y PnL 13084.98 al Close). La evaluación solo al Close distorsiona el comportamiento de stops.
- **Límite de la evidencia:** Validado sobre una sola estrategia (`rsi`), un fixture sintético (`spy-90d`, 1h continuo 24/7) y sin comisiones ni slippage.

## Hechos Verificados (fuente citada)

1. **`backtesting.py`:**
   - *Licencia:* AGPL-3.0 (fuente: `.venv/lib/python3.12/site-packages/backtesting-0.6.6.dist-info/METADATA`, classifier `License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)`). Exige liberar código si se ofrece como servicio.
   - *Multi-activo:* NO; `Backtest(data: pd.DataFrame, ...)` solo admite un único DataFrame OHLCV por corrida (fuente: firma y docstring de `backtesting.Backtest`).
   - *Última versión publicada:* 0.6.6 el 2026-07-22 (fuente: PyPI API metadata).
2. **Motor propio y stack compartido:**
   - Comparte arquitectura con el camino de paper trading:
     - `quantagent/trading/position_monitor.py`: seguimiento de posiciones activas (usado en `backtest.py` y `scheduler.py`).
     - `quantagent/strategy/assembler.py`: cableado unificado de `RiskManager`, `PortfolioManager`, `PositionSizer`, `PaperBroker` y `OrderManager`.
   - *Límite del stack compartido:* la regla de salida de posiciones NO está compartida; está duplicada entre `quantagent/strategy/base.py:should_exit` (usada en backtest, con trailing stop y razones en mayúsculas) y `quantagent/trading/scheduler.py:_check_exit_conditions` (líneas 583-620, sin trailing stop y con razones en minúsculas como `'stop_loss'`).
   - *Comisiones:* soporte implementado en `quantagent/trading/paper_broker.py` (`none` | `fixed` | `pct`), apagado por defecto (`commission_model="none"`).
3. **`vectorbt`:**
   - *Estado en VM:* No instalado en el entorno (`ModuleNotFoundError`).
   - *Última versión publicada:* 1.1.1 el 2026-09-26 (fuente: PyPI API metadata).
   - *Soporte multi-activo y modelo comunitario vs PRO de pago:* no verificado sin instalación en el entorno.

## Opciones Evaluadas

| Opción | Qué implica | Pros | Contras | Costo (PRs ≤100 lín.) |
|---|---|---|---|:---:|
| **A: Motor propio tal cual** | Mantener simulación actual al Close | Cero cambios; 227 trades idénticos a backtesting.py al close | Ignora mechas intravela; falsea efectividad real de SL/TP | 0 PRs |
| **B: Motor propio con SL/TP intravela + crossval permanente** *(Recomendada)* | Evaluar SL/TP contra High/Low de la vela en el motor propio; crossval como test CI | Corrige sesgo intravela; mantiene stack compartido con paper trading; sin riesgo AGPL; oráculo continuo | Requiere testear desempates High vs Low; regla de salida duplicada con scheduler.py pendiente de unificar | 3 PRs (T23b–T23d: QuantAgent-al2.1, al2.2, al2.3) |
| **C: `backtesting.py` bajo interfaz Strategy** | Reemplazar loop de simulación por `backtesting.py` | Engine maduro y validado por la comunidad | Licencia AGPL-3.0; sin soporte multi-activo nativo; desacopla backtest de PositionMonitor y RiskManager | ~5–6 PRs |
| **D: `vectorbt`** | Reemplazar por simulación vectorizada | Altísima velocidad de optimización | No verificado en VM; vectorbt PRO es comercial; paradigma vectorizado choca con agentes LLM interactivos | ~6–8 PRs |

## Recomendación

Se recomienda la **Opción B** (alineada con la posición preliminar de Fede en PR #55):
1. **Validación matemática demostrada:** El motor propio es contablemente idéntico a `backtesting.py` (5.79e-05 USD de diferencia en equity). No hay divergencias de contabilidad pendientes.
2. **Arquitectura y riesgo operativo:** Mantener el motor propio preserva la reutilización de `PositionMonitor`, `RiskManager` y `assembler.py` entre backtest y paper trading (con el límite de que la regla de salida sigue duplicada en `scheduler.py:_check_exit_conditions` y requerirá unificación). Adoptar `backtesting.py` bifurcaría aún más los caminos de ejecución e introduciría restricciones de licencia AGPL-3.0 y la limitación a un único activo.
3. **Costo acotado:** Resolver la evaluación intravela en el motor propio insume 3 PRs chicos ya en cola como T23b–T23d (`QuantAgent-al2.1`, `QuantAgent-al2.2` y `QuantAgent-al2.3`), usando el harness existente de `backtesting.py` (`crossval_compare.py --intrabar`) como oráculo y test de regresión permanente.

## Decisión

*Campo a completar por Fede en la revisión de este PR (posición preliminar: Opción B).*
