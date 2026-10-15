# QuantAgent-gfs1.1.2.10: test de sensibilidad fuera del alcance autorizado

Se intentó reescalar SL/TP antes del llenado en `cap_open_entries`, usando
apertura / (signal.entry_price o cierre), sin slippage. También se actualizaron
los niveles de ActivePosition y del CSV. El diff intentado tenía 60 líneas.

Comando ejecutado sobre el intento:
`timeout 20m python -m pytest -q -m 'not api' tests/test_bt_adapter*.py tests/test_compare_control.py`

Resultado: **31 passed, 1 failed**, en 238.67 s.
Falló `tests/test_compare_control.py::test_compare_control_sensibilidad_apertura_siguiente_gap`:
en línea 261, PnL obtenido **-1101.61**, esperado **-1543.87 ± 0.01**.
Los 18 tests del adaptador pasaron, incluidos los nuevos casos LONG/SHORT
(SL/TP 101.92/107.12 para LONG con apertura 104), y cierre sin cambio.

El cálculo manual del test usa `trade["exit_price"]` del motor al cierre
también para apertura. Al reescalar el stop cambia el precio de salida;
mantener ese precio común ya no representa el comportamiento solicitado.
`tests/test_compare_control.py` está en los tests dirigidos, pero no en
«Archivos relevantes». No se modificó para ampliar el alcance por cuenta propia.

Decisión que destraba: autorizar también ese archivo y actualizar el cálculo
manual de apertura para usar su stop reescalado y su propia salida.
Después, repetir los tests dirigidos, smoke y la comparación de las 48 celdas
con ambos llenados; las 48 celdas no se ejecutaron en este intento.

La implementación se retiró del árbol: esta entrega sólo agrega este bloqueo
y actualiza Beads. No se guardó rama WIP porque el contrato de aislamiento
prohíbe commit/push; el wrapper recibe únicamente la entrega de bloqueo.
El ticket queda in_progress para que el PM ajuste el alcance.

Verificación del árbol final, sin implementación: el mismo comando dio
**29 passed, 1 failed**, en 242.09 s. Falló
`test_control_mismo_llenado_motor[None]`: esperaba `coinciden 6 de 6`, obtuvo
`sin validar: fila 1, columna exit_time`. La base ya está roja: el adaptador
todavía no sigue el reescalado del motor. Smoke final: exit 0, 9 trades,
PnL 561.18. No se presenta la entrega documental como una corrección de tests.
