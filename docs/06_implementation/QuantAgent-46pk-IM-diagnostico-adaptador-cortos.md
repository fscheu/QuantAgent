# QuantAgent-46pk — Diagnóstico del adaptador en F2

## Resultado

Se reprodujo F2 sobre `etf-1d-2026-10-amplio` con el snapshot real en solo lectura y los resultados/registros aislados en `/tmp/f2-diag`. Para capturar el fallo original por activo, se reejecutó el control con el comportamiento anterior del adaptador (stop `0.01`, take `1000000000.0`). No se abrió 2019–2022, la reserva ni el candado; `local-data/intentos.csv` no se modificó.

Etapa 0 **PASA**; etapa 1 **PASA**, Sharpe 0,54; etapa 2 **CORTA**. El adaptador entregaba al broker del control un bracket long-only para una orden SHORT: stop `0.01` y take profit `1000000000.0`. `backtesting.py` lo rechaza con `ValueError: Short orders require: TP (1000000000.0) < LIMIT (<precio>) < SL (0.01)`. El rechazo era del adaptador, antes de comparar operaciones.

El adaptador ahora resuelve el stop corto con la regla existente de stop y sustituye el take profit inalcanzable por 3% debajo del precio de referencia. La prueba sintética `test_adaptador_normaliza_brackets_no_alcanzables_en_corto` reproduce ese caso; ya no lanza la excepción y genera un trade sell.

Las 12 celdas que antes fallaban, con el mensaje exacto de `backtesting.py` (los niveles están expresados en la escala del broker):

| Activo | Error anterior |
|---|---|
| QQQ | `Short orders require: TP (7.450580596923828) < LIMIT (2.884358208372072e-07) < SL (7.450580596923828e-11)` |
| XLP | `Short orders require: TP (7.450580596923828) < LIMIT (1.2234294337667526e-07) < SL (7.450580596923828e-11)` |
| XLV | `Short orders require: TP (7.450580596923828) < LIMIT (1.8130688260778785e-07) < SL (7.450580596923828e-11)` |
| EWJ | `Short orders require: TP (7.450580596923828) < LIMIT (2.734377411319316e-07) < SL (7.450580596923828e-11)` |
| XLY | `Short orders require: TP (7.450580596923828) < LIMIT (9.577929746154696e-08) < SL (7.450580596923828e-11)` |
| SPY | `Short orders require: TP (7.450580596923828) < LIMIT (7.268318154805527e-07) < SL (7.450580596923828e-11)` |
| IWM | `Short orders require: TP (7.450580596923828) < LIMIT (4.1028225801937284e-07) < SL (7.450580596923828e-11)` |
| VNQ | `Short orders require: TP (7.450580596923828) < LIMIT (2.0942050559274853e-07) < SL (7.450580596923828e-11)` |
| XLK | `Short orders require: TP (7.450580596923828) < LIMIT (6.76476641080901e-08) < SL (7.450580596923828e-11)` |
| XLI | `Short orders require: TP (7.450580596923828) < LIMIT (1.9221107252325862e-07) < SL (7.450580596923828e-11)` |
| EFA | `Short orders require: TP (7.450580596923828) < LIMIT (3.088017247632146e-07) < SL (7.450580596923828e-11)` |
| XLF | `Short orders require: TP (7.450580596923828) < LIMIT (1.2314600919973105e-07) < SL (7.450580596923828e-11)` |

## Diferencias restantes del control

En la reproducción corregida no quedan excepciones de órdenes cortas. El control aún marca estas diferencias, con sus mensajes exactos:

| Celdas | Mensaje |
|---|---|
| EWJ, QQQ, XLI, XLK, XLP, XLY | `sin validar: fila 1, columna exit_time` |
| EFA, IWM, VNQ | `sin validar: distinta cantidad de trades (motor=21, port=22)` |
| SPY | `sin validar: distinta cantidad de trades (motor=20, port=22)` |
| XLV | `sin validar: distinta cantidad de trades (motor=20, port=22)` |
| XLF | `sin validar: fila 1, columna exit_price` |

Quitar esas diferencias no cambia el veredicto: la salida de etapa 2 conserva cortes independientes por menos de dos trades, P&L no positivo y drawdown mayor a 0,35. F2 sigue **CORTA**. Los caminos long-only no cambiaron.

## Verificación

`NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mpl-cache python -m pytest -q tests/test_bt_adapter.py tests/test_compare_control.py tests/test_strategy_registration.py tests/test_golden_snapshot.py`: **39 passed**.
