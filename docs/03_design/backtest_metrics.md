# Backtest Metrics: Especificación y Auditoría

Documento de diseño y auditoría para las métricas del motor de backtest (QuantAgent-hx0.1, QuantAgent-hx0.2).

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

## 2. Win Rate y Profit Factor: Fórmulas y Políticas

### 2.1 Conteo de Operaciones y Trades Neutros ($pnl = 0$)
Para todo conjunto de trades cerrados con $pnl$ liquidado:
- **Winning trades ($W$):** $\{t \in \text{trades} \mid t.pnl > 0\}$
- **Losing trades ($L$):** $\{t \in \text{trades} \mid t.pnl < 0\}$
- **Breakeven / Neutros ($B$):** $\{t \in \text{trades} \mid t.pnl = 0 \lor t.pnl \text{ es None}\}$
- **Total trades ($N$):** $|W| + |L| + |B|$

### 2.2 Win Rate
Existen dos políticas para el tratamiento de operaciones con $pnl = 0$:

- **Política A (Actual / Estándar):**
  Los trades neutros se contabilizan en el total de trades pero no como ganadores.
  $$\text{win\_rate} = \frac{|W|}{N} = \frac{\text{COUNTIF}(pnl > 0)}{\text{COUNT}(pnl)}$$
  *Ejemplo (3 ganadores, 2 perdedores, 1 neutro):* $\text{win\_rate} = \frac{3}{6} = 50.00\%$.

- **Política B (Propuesta a decidir):**
  Los trades neutros se excluyen del cálculo del porcentaje de acierto y se reportan por separado.
  $$\text{win\_rate} = \frac{|W|}{|W| + |L|}$$
  *Ejemplo (3 ganadores, 2 perdedores, 1 neutro):* $\text{win\_rate} = \frac{3}{3 + 2} = 60.00\%$.

*Implementación:* El motor y `recalc_metrics.py` aplican la **Política A**.

### 2.3 Profit Factor
El factor de beneficio mide la relación entre ganancias brutas y pérdidas brutas:
$$\text{profit\_factor} = \frac{\sum_{t \in W} t.pnl}{|\sum_{t \in L} t.pnl|} = \frac{\text{SUMIF}(pnl > 0)}{|\text{SUMIF}(pnl < 0)|}$$

**Tratamiento sin trades perdedores ($L = \emptyset$):**
- **Persistencia (`BacktestRun.profit_factor`):** Nunca se persiste `float("inf")`. Si no hay pérdidas o no hay operaciones, se guarda `None` (`NULL` en SQL) o `0.0` (si tampoco hay ganancias), garantizado por validador en `BacktestRun` y `_update_backtest_run`.
- **Salida en CLI:** Propuesta `"n/a"` cuando `profit_factor is None` / infinito.

---

## 3. Auditoría sobre `spy-90d` / RSI (slippage 0,05%, base limpia)

| Métrica / Fórmula | Valor CSV (`recalc_metrics.py`) | Valor Engine (`_calculate_metrics`) | Coincide |
|---|---|---|---|
| Total trades (`COUNT(pnl)`) | 227 | 227 | Sí |
| Ganadores (`COUNTIF(pnl > 0)`) | 113 | 113 | Sí |
| Perdedores (`COUNTIF(pnl < 0)`) | 114 | 114 | Sí |
| Neutros (`COUNTIF(pnl = 0)`) | 0 | 0 | Sí |
| Win rate (`COUNTIF(pnl > 0) / COUNT(pnl)`) | 49.78% | 49.78% | Sí |
| Gross profit (`SUMIF(pnl > 0)`) | 19307.91 | 19307.91 | Sí |
| Gross loss (`ABS(SUMIF(pnl < 0))`) | 7186.94 | 7186.94 | Sí |
| Profit factor (`Gross profit / Gross loss`) | 2.69 | 2.69 | Sí |
| Total PnL (`SUM(pnl)`) | 12120.98 | 12120.98 | Sí |
| Capital identity (`equity_final - initial`) | 12120.98 | 12120.98 | Sí |
| PnL por trade (recalc vs trade log) | 227/227 | 227/227 | Sí |
| Max drawdown (antes 0.090128) | 0.001204 | 0.001204 | Sí |
| Sharpe ratio (antes 8.74, original 0.57) | 25.14 | 25.14 | Sí |

---

## 4. Equity Curve y Max Drawdown (QuantAgent-hx0.3)

### 4.1 Construcción de la Curva de Equity
- **Frecuencia:** Un punto por vela (al final de cada período analizado).
- **Punto inicial:** La curva inicia con `equity = initial_capital` (`positions_value = 0.0`, `cash = initial_capital`).
- **Valuación a Mercado (Mark-to-Market):**
  En cada vela $t$, con precio de cierre $P_t$:
  $$\text{equity}_t = \text{cash}_t + \sum_{\text{longs}} \text{qty} \times P_t - \sum_{\text{shorts}} |\text{qty}| \times P_t$$
  Abrir una posición (long o short) preserva la equity salvo por el slippage de ejecución ($0,05\%$ por lado).
  Al finalizar el backtest (`_close_remaining_positions()`), las posiciones abiertas remanentes se cierran al precio final y el último punto de la curva refleja el capital neto liquidado, cumpliendo:
  $$\text{equity\_final} - \text{initial\_capital} = \text{total\_pnl} \quad (\pm 0.01)$$

### 4.2 Drawdown y Max Drawdown
Para la serie temporal de equity $\{\text{equity}_t\}_{t=0}^{T}$:
- **Running peak:** $\text{peak}_t = \max_{0 \le i \le t} \text{equity}_i$
- **Drawdown porcentual:** $\text{dd}_t = \frac{\text{peak}_t - \text{equity}_t}{\text{peak}_t}$
- **Maximum Drawdown:** $\text{max\_drawdown} = \max_{0 \le t \le T} \text{dd}_t$

---

## 5. Sharpe Ratio y Anualización (QuantAgent-hx0.4)

### 5.1 Fórmula y Fuente de Retornos
El ratio de Sharpe anualizado mide el retorno excedente sobre la tasa libre de riesgo normalizado por la volatilidad muestral:
$$\text{Sharpe} = \left(\frac{\overline{r} - \frac{r_f}{N}}{\sigma_r}\right) \times \sqrt{N}$$

- **Fuente de retornos:** Retornos porcentuales por vela calculados a partir de la serie temporal de equity $\{\text{equity}_t\}_{t=0}^{T}$:
  $$r_t = \frac{\text{equity}_t - \text{equity}_{t-1}}{\text{equity}_{t-1}}, \quad t = 1, \dots, T$$
  $\overline{r} = \text{mean}(r)$, $\sigma_r = \text{stdev}(r)$ (desviación estándar muestral).
- **Tasa libre de riesgo ($r_f$):** Anual, default $0.02$ (2%).

### 5.2 Regla de Periodos por Año ($N$ - Opción D)
En lugar de una tabla fija por timeframe que presupone horario bursátil NYSE, $N$ se deriva de las observaciones de la corrida:
$$N = \frac{\text{len}(r)}{\text{años transcurridos}}$$
donde:
$$\text{años transcurridos} = \frac{(t_T - t_0)\text{ en segundos}}{365.25 \times 86400}$$
- Si la curva tiene menos de 2 puntos o cubre un lapso nulo ($\text{años transcurridos} \le 0$), $\text{Sharpe} = 0.0$.
- Para `spy-90d` (2159 retornos horarios entre 2026-01-01T00:00:00 y 2026-03-31T23:00:00):
  $\text{días} = 89.9583$, $\text{años} = 0.24629$, $N = 2159 / 0.24629 = 8766.0$.
  Sharpe pasa de 8.74 (con la tabla fija $N=1638$) a **25.14** (con $N=8766.0$).
