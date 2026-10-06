# Backtest Metrics: Especificación y Auditoría de PnL

Documento de diseño y auditoría para las métricas de PnL del motor de backtest (QuantAgent-hx0.1).

---

## 1. PnL: Fórmulas y Definiciones

### 1.1 PnL por Trade
Para cada operación cerrada (`Trade` con `closed_at` y `exit_price`), el PnL se calcula a partir de los precios ejecutados reportados por el broker (`entry_price` y `exit_price`), los cuales ya incorporan el slippage por lado (default 0,05% según QuantAgent-hx0.9). Las comisiones son actualmente 0 (`commission = 0`, scope de QuantAgent-les):

- **LONG (compra inicial, venta de cierre):**
  $$\text{pnl} = (\text{exit\_price} - \text{entry\_price}) \times \text{qty} - \text{commission}$$
  $$\text{pnl\_pct} = \frac{\text{pnl}}{\text{entry\_price} \times \text{qty}} \times 100$$

- **SHORT (venta inicial, compra de cierre):**
  $$\text{pnl} = (\text{entry\_price} - \text{exit\_price}) \times \text{qty} - \text{commission}$$
  $$\text{pnl\_pct} = \frac{\text{pnl}}{\text{entry\_price} \times \text{qty}} \times 100$$

### 1.2 Total PnL y Retorno Porcentual
- **Total PnL:** Suma de `Trade.pnl` de todas las operaciones cerradas del run:
  $$\text{total\_pnl} = \sum_{t \in \text{trades}} t.\text{pnl}$$

- **Total Return %:** Rendimiento porcentual sobre el capital inicial:
  $$\text{total\_return\_pct} = \left(\frac{\text{total\_pnl}}{\text{initial\_capital}}\right) \times 100$$

### 1.3 Identidad de Capital y Portafolio
Al finalizar el backtest, las posiciones abiertas remanentes se cierran forzosamente (`_close_remaining_positions()`). Con todas las posiciones liquidadas a precios de salida:
$$\text{equity\_final} = \text{portfolio.cash} = \text{initial\_capital} + \text{total\_pnl}$$
$$\text{equity\_final} - \text{initial\_capital} = \text{total\_pnl}$$

---

## 2. Auditoría sobre `spy-90d` / RSI (slippage 0,05%, base limpia)

| Fuente / Métrica | Valor verificado |
|---|---|
| `sum(pnl CSV)` (`scripts/recalc_metrics.py`) | 12120.98 |
| `total_pnl engine` (`_calculate_metrics()`) | 12120.98 |
| `equity_final - initial_capital` (portfolio cash tras cierre final) | 12120.98 |
| `PnL por trade` (recalc vs trade log) | 227/227 coinciden |

### Nota sobre `equity_curve` intradía (`--equity-out`)
La serie registrada en `bt.equity_curve` por vela durante el backtest acumula estados previos al cierre final y sobrevalúa temporalmente posiciones cortas en `get_total_value()` (duplicación de nocional al sumar cash y liability). Esto afecta Sharpe y Max Drawdown intradía, cuyo recálculo y auditoría corresponden a los tickets T11/T12 (`QuantAgent-hx0.6` / `QuantAgent-hx0.3`).
