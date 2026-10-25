# Partición de QuantAgent-e4jc.4

Estimación previa: 150 líneas. El cambio combina PnL de posición corta en la cartera, simetría de
costos, regresión long-only y extensión del control cruzado con `backtesting.py`. No se implementó
código.

| Hijo | Objetivo | Aceptación binaria | Líneas | Revisión |
|---|---|---|---|---|
| QuantAgent-e4jc.4a | PnL de celda corta en cartera | Test sintético demuestra ganancia al bajar y pérdida al subir, con valores esperados | 45 | leer |
| QuantAgent-e4jc.4b | Costos simétricos y preservar long-only | Costos/PnL neto iguales para largo y corto de igual nocional y variación; referencia long-only sin cambio | 45 | leer |
| QuantAgent-e4jc.4c | Control corto con `backtesting.py` | Motor y control coinciden para corto con costos y para una referencia long-only | 60 | probar |

Orden obligatorio: .4a → .4b → .4c, con dependencias en Beads. Cada hijo estima menos de 100 líneas.
Las tres tareas conservan el alcance del padre y no incluyen paper ni la corrida de F2. El padre queda
`in_progress` hasta integrar los hijos; F2-b y F2-c mantienen su dependencia del padre.
