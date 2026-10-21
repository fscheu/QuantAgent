# QuantAgent-lrf8 — partición T5

La implementación completa se estima en 250 líneas con tests y adaptaciones:
proveedor (80), cableado del motor (75), loaders y CLI (95).
El contrato §3.2 exige partir antes de escribir código si supera 100.

Se conserva la decisión del PM del 2026-10-21 02:54: default legacy,
fuente externa explícita por source.name y offline por source_name.

| Hijo | Cambio y aceptación binaria | Revisión | Estimación |
|---|---|---|---|
| QuantAgent-lrf8.1 | DataProvider separa lectura y escritura de snapshot:x y alpaca con igual clave; sin fuente explícita conserva legacy | leer | 80 |
| QuantAgent-lrf8.2 | Las dos consultas del backtest y su DataProvider usan la fuente configurada; fechas y precios de alpaca no contaminan snapshot:x | leer | 75 |
| QuantAgent-lrf8.3 | Loaders escriben fixture/snapshot:nombre y ambos caminos del CLI leen esa fuente; referencias intactas | decidir | 95 |

Orden obligatorio: .1 → .2 → .3. Hasta .3 los loaders y el CLI conservan
legacy, para que cada entrega siga funcionando. .3 adapta los tests que
cargan fixtures directamente para declarar su fuente al construir el backtest.
Si alguna estimación supera 100 al revisar esos tests, ese hijo se vuelve a partir.

Cada hijo exige tests dirigidos y smoke verdes; .2 y .3 también oráculos.
En .3 se verifican las tres referencias M1 y rsi/SPY 2007–2018,
snapshot etf-1d-2026-10, costos etf, tamaño 100%: 187 trades y -9141.21.
No se modifican referencias ni lógica de ejecución del motor.
Usar NUMBA_CACHE_DIR y MPLCONFIGDIR en /tmp; nunca NUMBA_DISABLE_JIT.

El padre queda in_progress hasta completar los hijos; T7 sigue bloqueado por él.
Esta entrega solo modifica planificación y Beads; no implementa T5.
