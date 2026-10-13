# Diagnóstico: rsi/GLD 243 trades al cierre vs 242 en apertura siguiente (V27c)

- **Issue:** QuantAgent-u8lf (Fila V27c, Lote 4 / L7)
- **Tipo:** Diagnóstico y Decisión
- **Estado:** Aceptado

---

## 1. Contexto

En la matriz 4x12 de comparación con datos reales (`etf-1d-2026-10`, 2007–2018, `--costos etf`, 100%), la sensibilidad de apertura siguiente (`--fill next-open`) arroja exactamente la misma cantidad de trades que el llenado al cierre (`--fill close`) en 47 de las 48 celdas. La única excepción es `rsi/GLD`, que registra 243 trades al cierre y 242 trades en apertura siguiente.

Este documento diagnostica la causa exacta de la divergencia, identifica el primer trade discrepante, detalla las reglas de código involucradas y dictamina si se trata de un comportamiento esperado o de un defecto del adaptador.

---

## 2. Comandos de Reproducción

Ambas corridas se reproducen de forma aislada y determinista con los siguientes comandos:

```bash
# Corrida al cierre (243 trades)
python scripts/bt_adapter.py --strategy rsi --snapshot etf-1d-2026-10 --symbol GLD --from 2007-01-03 --to 2018-12-31 --costos etf --fill close

# Corrida en apertura siguiente (242 trades)
python scripts/bt_adapter.py --strategy rsi --snapshot etf-1d-2026-10 --symbol GLD --from 2007-01-03 --to 2018-12-31 --costos etf --fill next-open
```

Conteos observados:
- `fill close`: 243 trades (100 ganadores, 143 perdedores).
- `fill next-open`: 242 trades (99 ganadores, 143 perdedores).

---

## 3. Primer Trade que Diverge

Los primeros 242 trades (trades 1 a 242) coinciden uno a uno entre ambas corridas en:
- Dirección (`side`: compra o venta).
- Motivo de salida (`exit_reason`: `STOP_LOSS`, `TAKE_PROFIT` o salida por regla).

El primer (y único) trade que diverge es el **trade número 243**:
- **Fecha de entrada:** `2018-12-31` (última sesión del rango de backtest).
- **Fecha de salida:** `2018-12-31`.
- **Dirección:** venta (`sell`).
- **Motivo de salida:** `backtest_end`.
- **En `fill close`:** el trade se ejecuta (entra y sale en la misma sesión).
- **En `fill next-open`:** el trade **no existe** (0 ejecuciones en esa sesión).

---

## 4. Regla del Código que Explica la Divergencia

La diferencia se origina en la interacción entre la generación de señal en la última vela de la historia y el momento de ejecución de cada modo:

1. **Generación de señal (`scripts/bt_adapter.py`, función `StrategyAdapter._maybe_enter`):**
   En la sesión `2018-12-31` (la última vela del dataset filtrado con `--to 2018-12-31`), la posición anterior (trade 242) ya había cerrado el `2018-12-26`. En `2018-12-31`, el RSI genera una señal de venta (`sell`) y coloca una orden de mercado en el broker (`self.sell(...)`).

2. **Ejecución al cierre (`fill close`, `scripts/bt_adapter.py`, funciones `run_adapter` y `StrategyAdapter.next`):**
   - Se utiliza `trade_on_close=True` y un DataFrame con vela centinela artificial (`crossval_rsi.load_fixture`).
   - La orden colocada en `_maybe_enter` se ejecuta al cierre de `2018-12-31`.
   - Inmediatamente, la regla de fin de backtest en `StrategyAdapter.next` (líneas 162-163: `if self.fill_close and n == self.n_real_bars and self.pos is not None: self._close("backtest_end")`) detecta que se alcanzó la última vela real y envía una orden opuesta para forzar el cierre con motivo `backtest_end`.
   - El resultado es una posición de duración cero que entra y sale al cierre de la misma vela.

3. **Ejecución en apertura siguiente (`fill next-open`, `scripts/bt_adapter.py`, función `run_adapter`):**
   - Se utiliza `trade_on_close=False`, `finalize_trades=True` y el DataFrame sin vela centinela (`df.iloc[:-1]`).
   - La orden colocada en `_maybe_enter` al cierre de `2018-12-31` está programada por la librería para ejecutarse en la **apertura de la siguiente vela** (`2019-01-02`).
   - Dado que el backtest está acotado al período 2007-01-03 a 2018-12-31, **no existe una vela siguiente** en la simulación.
   - Al finalizar el bucle de barras, `backtesting.py` ejecuta `finalize_trades`, pero este mecanismo únicamente liquida operaciones que ya están abiertas (`self._broker.trades`). Las órdenes pendientes en `self._broker.orders` nunca llegan a ejecutarse porque la sesión en que debían abrir nunca ocurre.
   - En consecuencia, el trade número 243 nunca se abre ni se cierra.

---

## 5. Conclusión: Comportamiento Esperado

**Es el comportamiento esperado** del modelo de llenado en apertura siguiente, y **no constituye un defecto del adaptador**.

Razones:
1. Por definición física de la apertura siguiente, una señal emitida al cierre de la sesión $T$ se ejecuta en la apertura de $T+1$. Si $T$ es la última sesión de la ventana de backtest, la ejecución requeriría datos de una sesión fuera de la muestra evaluada (e invadiría el tramo de prueba 2019–2022).
2. El trade número 243 de `fill close` es un artefacto de la ejecución intra-vela/cierre: una operación ficticia de duración cero que abre y cierra en el mismo instante únicamente para saldar la cartera antes de terminar.
3. En los otros 11 ETFs analizados (DBC, EEM, EFA, IWM, SHY, SPY, TLT, XLE, XLF, XLK, XLU), ninguna posición abrió en la última vela: o bien la posición venía abierta de sesiones anteriores (y por ende `backtest_end` la cerró en ambos modos), o bien no hubo señal en la última sesión. GLD fue el único caso donde una señal coincidió exactamente con la última barra del histórico disponible.

No se requiere emitir ticket de corrección.
