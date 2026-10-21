# QuantAgent-01kb: falta aislamiento por fuente

El ticket exige invalidar la historia de `(symbol, timeframe, source)` al detectar
un cambio relativo de `close` mayor a 1e-4 en la vela solapada.

Se revisaron `quantagent/data/provider.py` y `quantagent/models.py:MarketData`.
`MarketData` no tiene columna `source`; las consultas y escrituras del proveedor
solo distinguen símbolo y timeframe. No se intentó borrar datos ni cambiar código.
Implementar el borrado así podría eliminar historia de otra fuente o un snapshot.

La columna corresponde a Q08 (`QuantAgent-1y0i`) y el aislamiento de lecturas y
escrituras a Q09 (`QuantAgent-lrf8`), posteriores a Q07 en la cola L18.
Agregar esos cambios acá excedería los archivos autorizados y el alcance de T7.

Decisión que destraba: ejecutar Q08 y Q09 antes de retomar Q07 y registrar ambas
dependencias en Beads. Alternativa: redefinir explícitamente T7 para una base
de fuente única, aceptando el borrado por símbolo y timeframe.

No se implementaron los dos tests de aceptación porque falta resolver la clave
del borrado. El ticket queda `in_progress`, pendiente de decisión del PM.
