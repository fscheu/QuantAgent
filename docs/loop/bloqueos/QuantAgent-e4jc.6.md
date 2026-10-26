# Bloqueo QuantAgent-e4jc.6

No se pudo ejecutar F2 hasta la etapa 2 en este worktree. `local-data/` solo tiene
su README; no están el snapshot requerido `etf-1d-2026-10-amplio` ni los datos de
la hipótesis. Además, pasar directamente la propuesta al comando `correr` da
`Error: Falta el campo 'autor'`; se necesita el YAML admitido que usa la corrida.
La decisión que destraba el ticket es proveer en la VM esos artefactos (o su ruta
de generación) para guardar el resultado real y redactar el veredicto y la
correlación con 1/N.

Primer intento: `python -m quantagent.cli backtest hipotesis correr --help` falló
al importar vectorbt porque Numba no encontró una ubicación de caché escribible.
Con `NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mpl-cache`, el CLI cargó.
Esto permite cargar el CLI. El siguiente comando no ejecuta la hipótesis: falla
porque el YAML de catálogo no está admitido y no tiene `autor`:

`NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mpl-cache python -m quantagent.cli backtest hipotesis correr catalogo/propuestas/tendencia-larga-corta.yaml --snapshot etf-1d-2026-10-amplio --carpeta local-data/intentos/f2 --hasta-etapa 2`

La línea `try.sh` no se pudo verificar: requiere fetch/una rama que no existe en
este worktree, y `.git` es de solo lectura. No se generó veredicto ni se modificó
el candado. No se pidió el OK de reserva.
