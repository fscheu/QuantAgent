# QuantAgent-jzb6: por qué difieren las curvas de equity del motor y del adaptador (V33.2d)

Celda: fixture sintético `spy-2y-1d` (snapshot temporal, sin datos reales), `fifty-two-week-high`, llenado al cierre,
slippage 0.0005, comisión 0, tamaño 1. Script: `docs/loop/diagnosticos/QuantAgent-jzb6_curvas.py`.

**Primera diferencia:** 2024-11-16 (vela 320, la primera entrada), -10.18 USD.

| vela 320 | equity | efectivo | cantidad | valor posición |
|---|---|---|---|---|
| motor | 99 989.82 | 79 628.23 | 50.0396 | 20 361.59 |
| adaptador | 100 000.00 | 100 000.00 | 0 | 0 |

Ambas curvas tienen 520 puntos con las mismas fechas y arrancan en 100 000. Solo difieren 6 velas, las 6 entradas
(320, 355, 390, 425, 460, 495), y en cada una la diferencia es el slippage de la entrada (20 361.59 × 0.0005 = 10.18).
En las salidas coinciden. Al día siguiente las curvas vuelven a ser iguales (diferencia < 1e-5).

**Causa:** `backtesting.py` escribe `_equity[i]` en `_Broker.next()` *antes* de llamar a `Strategy.next()` de la vela i.
Con `trade_on_close=True`, la orden de mercado que el adaptador pone en la vela i se llena recién al procesar la vela
i+1, a `Close[i]` y con `entry_bar = i`. Así, `Equity[i]` no incluye el costo del llenado (slippage y comisión) y el
motor sí lo incluye, porque registra la equity después de ejecutar la orden. Las seis salidas de esta celda son
`STOP_LOSS`: la librería las llena dentro de la vela (orden contingente, `time_index = i`) antes de registrar la
equity, y por eso coinciden. Por inferencia, sin verificar en esta celda: el mismo desfase afecta las salidas por regla
y `backtest_end`, que también son órdenes de mercado al cierre (rsi/SPY no se corrió acá: necesita datos reales).

**Verificación:** si a `Equity` se le resta el costo de cada entrada en su vela, la diferencia máxima con el motor es
6e-06 USD y el Sharpe (tasa 0, frecuencia observada) pasa de -4.6302 a -4.6756, igual al motor (-4.6756).
El valor del bloqueo `QuantAgent-hnlg` (-4.63 frente a -4.68) es este mismo efecto. Descartado: distinta grilla de
fechas, distinta equity inicial, distinto valor de la posición (ambas marcan a `close`) y diferencias en las salidas.

**Ticket de fix propuesto** (≤100 líneas, `scripts/bt_adapter.py` y su test):
- Agregar `equity_curve(stats)` en `scripts/bt_adapter.py`: con llenado al cierre, devuelve `Equity` truncada a
  `n_real_bars`, restando en la vela del llenado el costo (slippage + comisión sobre el nocional de referencia) de cada
  orden de mercado llenada al cierre: entradas, salidas por regla y `backtest_end` (no stops ni take profit).
- Test que falla hoy: en la celda de arriba, `equity_curve(stats)` iguala la curva del motor (`get_equity_curve()`)
  con tolerancia 1e-4 USD en las 520 velas, y el Sharpe tasa 0 coincide a 2 decimales (hoy: 10.18 USD y -4.63/-4.68).
  Un segundo caso con salida por regla (p. ej. `rsi` sobre el mismo fixture) cubre el lado de la salida.
- No cambia el motor. Lo que use la curva del adaptador (V33.2) pasa a leer `equity_curve(stats)`.
