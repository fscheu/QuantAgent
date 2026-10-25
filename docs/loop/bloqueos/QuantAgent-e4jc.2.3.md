# Bloqueo: QuantAgent-e4jc.2.3

Los tests dirigidos recolectaron 316 casos con `NUMBA_CACHE_DIR=/tmp/numba-cache`; terminaron con 302 pasados y
14 fallidos. Uno espera una lista de aliases anterior a la estrategia `rebalanceo-fin-de-mes`; otros fallan
porque las órdenes de los fixtures son rechazadas por el límite de posición de 10 % (`Position too large`).
La primera corrida sin `NUMBA_CACHE_DIR` no pudo recolectar 11 archivos: Numba no encontró locator de caché
para `vectorbt/utils/random_.py`.

`NUMBA_CACHE_DIR=/tmp/numba-cache scripts/loop/smoke.sh` terminó con exit 0 (9 trades, PnL 561.18). El
comando de `try.sh` no pudo ejecutarse: el checkout aislado no permite escribir `.git/FETCH_HEAD` y la rama
`loop/QuantAgent-e4jc.2.3` todavía no existe en el worktree. No quedó un cambio de código porque los tests
dirigidos no terminaron en 0 failed.

Destraba: resolver los 14 fallos del entorno/tests o confirmar la configuración esperada del límite de posición;
después correr tests dirigidos y `try.sh` en un checkout writable.
