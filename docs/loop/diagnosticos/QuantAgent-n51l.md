# QuantAgent-n51l: por qué `rsi --costos cripto` da 46 trades en el motor y 31 en el adaptador

Celda: fixture sintético `spy-2y-1d` (snapshot temporal, sin datos reales), `rsi`, llenado al cierre, perfil `cripto`
(slippage 0.0005, comisión 0.001), tamaño 1. Script: `docs/loop/diagnosticos/QuantAgent-n51l_margen.py`.

**Primera entrada cancelada:** 2024-12-14 (vela 348), **venta en corto** de 259.19436816 a close 392.07. Las dos cuentas
llegan iguales: equity 101 622.34 en ambos (el corto anterior, 2024-12-10, salió por stop en esa misma vela).

| 2024-12-14 | efectivo antes | cantidad | llenado | nocional | comisión | resultado |
|---|---|---|---|---|---|---|
| motor | 101 622.34 | 259.19436816 | 391.873965 (close × (1 − s)) | 101 571.52 | 101.57 | ejecuta; efectivo después 203 092.29 |
| librería | margin_available 101 622.34 | 259.19436816 | — | — | — | exige 259.19 × 392.266035 = 101 673.15; faltante 50.81: cancela |

**Causa:** `_Broker._process_orders` (backtesting 0.6.6) cancela si `|size| × (adjusted_price + comisión/|size|) >
margin_available × leverage`. Para un corto `adjusted_price = close × (1 − s)`, así que el costo por unidad es
`close × (1 − s + c)`. Con tamaño 1 y confianza 1 el nocional es el equity entero; si `c > s` el costo lo supera y la
orden se cancela. Solo pasa con `cripto` (0.001 > 0.0005): con `etf` y `etf-conservador` `c ≤ s` y entra.
El motor no exige eso: `RiskManager` check 1 compara `cash >= qty × price`, sin comisión, y
`OrderManager._cap_entry_qty` deja los cortos sin recortar (la venta suma efectivo).

La hipótesis del ticket era parcial: no son compras. El recorte del adaptador (`QuantAgent-gti6`, en `_maybe_enter`) solo
cubre largos, y para cortos no alcanza con recortar: el motor ejecuta la cantidad entera, un recorte daría otra `qty`.

**Alcance:** 40 intentos cancelados, todos cortos, en 5 rachas (2024-12-14, 2025-01-18, 2025-02-22, 2025-03-29,
2025-05-03): el adaptador reintenta cada vela mientras la señal sigue y entra cuando pide menos que el
equity (2024-12-22: 67 290 de 101 622). Eso explica los 15 trades de diferencia. Contraprueba en el script: la misma corrida con `margin = 1 / (1 + c)` (apalancamiento
1 + c) da `coinciden 46 de 46`; con `margin = 1.0`, `motor=46, port=31`.

**Ticket de fix propuesto** (≤100 líneas, `scripts/bt_adapter.py` y `tests/test_compare_control.py`):
- Que la librería acepte la orden que el motor ejecuta: el chequeo de margen de la librería no corresponde a ninguna
  regla del motor para cortos. Recomiendo `margin=1 / (1 + commission_pct)` en `run_adapter` (una línea, comentada).
  Riesgo: un largo con nocional apenas mayor al efectivo (entre `cash` y `cash × (1 + c) / (1 + s + c)`), que el motor
  rechaza por check 1, la librería lo aceptaría. Hoy solo es alcanzable con `--tamano-posicion > 1`. Si se quiere
  cero riesgo: subir `broker._leverage` solo mientras se procesan entradas en corto, en el wrapper de `_process_orders`.
- Test que falla hoy: `compare` + `compare_control.py --costos cripto` sobre `spy-2y-1d` con `rsi` da
  `coinciden 46 de 46` (hoy: `distinta cantidad de trades (motor=46, port=31)`).
- No cambia el motor.
