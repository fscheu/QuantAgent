# Bloqueo — QuantAgent-cq4u

El punto de control S09 no se puede cerrar con el estado de datos disponible.

## Intentado

- Generé los informes CLI de `cartera-transversal`, `cartera-estacionalidad` y `cartera-tendencia-momentum` con curvas guardadas; los tres terminaron con `Informe en /tmp/cq4u-<id>.html`.
- Para `cartera-nucleo-curva`, el mismo comando termina con `Error: Hipótesis desconocida: cartera-nucleo-curva`.
- El YAML y las curvas guardadas existen, pero `/home/azureuser/repos/projects/QuantAgent/local-data/hipotesis.csv` no tiene filas de esa hipótesis. No modifiqué `local-data` ni abrí ventanas.
- `armar_pedido(n_intentos=109)` terminó en seco: 6899 caracteres, sin Sharpe, DSR ni ganancias.
- `NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mpl-cache QUANTAGENT_SNAPSHOT_DIR=/home/azureuser/repos/projects/QuantAgent/local-data python -m pytest -q -m oraculo`: 35 passed, 1668 deselected.
- `NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mpl-cache QUANTAGENT_SNAPSHOT_DIR=/home/azureuser/repos/projects/QuantAgent/local-data scripts/loop/smoke.sh`: exit 0.

## Qué destraba

El PM necesita una copia consistente de `hipotesis.csv` que incluya el estado de `cartera-nucleo-curva`, o decidir formalmente que el informe ya publicado en `QuantAgent-ro3w-IM-criterio-paper-curva.md` cuenta como cuarta fila. Con esa fuente se puede completar el IM con las cuatro carteras y reproducirlo sin registrar intentos.
