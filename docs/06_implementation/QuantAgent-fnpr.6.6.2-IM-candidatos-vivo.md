# M5-55b: candidatos semanales en vivo

`quantagent.fabrica.candidatos_vivo.candidatos_en_vivo(as_of,
etfs_habilitados, data_provider, fundamentales, composicion="sp500-2026-10",
base_dir=None)` devuelve `(candidatos, descartes)`. Obtiene la última sesión
NYSE cerrada mediante `ultima_sesion_cerrada_nyse(as_of)`, sin consultar el
reloj del sistema. Pide un año de barras diarias al proveedor inyectado para
los ETFs habilitados y los miembros de sus sectores. Normaliza fechas,
conserva el último cierre por día y excluye barras posteriores a la sesión.
El fin del pedido es el día siguiente, como en `_panel_cierres`.

Calcula `metricas_fundamentales` sólo para esa sesión y los miembros presentes
en el snapshot, y delega el ranking a `candidatos_semanales`, sin modificarlo.
La composición se pasa como ruta absoluta para respetar `base_dir` también
dentro del screener. No usa snapshots históricos de precios como fallback:
el diccionario incluye incluso las series vacías de las acciones solicitadas.

`descartes` contiene diccionarios `{simbolo, motivo}` para acciones sin
fundamentales en el snapshot o sin precios útiles. Reutiliza el criterio de
retornos del screener: menos de 64 cierres válidos o último cierre con más de
cinco días de antigüedad impiden ambos retornos. Con 64 a 126 cierres puede
faltar sólo el retorno de seis meses; se conserva la tolerancia original de
una métrica faltante. Los descartes no enumeran exclusiones por ranking,
cuartil o métricas contables insuficientes. Una acción puede tener dos motivos.

El llamador elige el nombre de `fundamentales`, por ejemplo a partir de
`snapshot_vigente` del [refresco M5-55a](QuantAgent-fnpr.6.6.1-IM-datos-vivos-h3.md).
Si es `None`, inexistente o su manifiesto no contiene símbolos, devuelve una
tabla vacía con las columnas del screener y un descarte global con
`simbolo=None` y motivo explícito. No consulta precios en ese caso; el
llamador puede continuar sólo con ETFs. Este adaptador no refresca ni elige
snapshots y no oculta errores del proveedor o corrupción de los Parquet.

Limitación: **sp500-2026-10 llega hasta 2026-08-18**. `sp500.miembros` usa la
última fila anterior o igual a la fecha solicitada, por lo que después de ese
día se mantienen los últimos miembros conocidos. Actualizar la composición
queda fuera del ticket, igual que integrar el pipeline, guardar CSV o bajar
datos reales.

Las pruebas usan composición y fundamentales temporales, precios sintéticos
y un proveedor de fixture que implementa `get_ohlc`. Comparan toda la tabla
con una llamada directa al screener y cubren ausencias de precio y snapshot,
historia insuficiente o vieja, sectores habilitados y reloj antes/después del
cierre. Comandos desde el worktree:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fnpr662-mpl NUMBA_CACHE_DIR=/tmp/fnpr662-numba /home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest tests/test_candidatos_vivo.py tests/test_screener*.py tests/test_sp500.py tests/test_fundamentales*.py -q
PATH="/home/azureuser/repos/projects/QuantAgent/.venv/bin:$PATH" PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fnpr662-mpl NUMBA_CACHE_DIR=/tmp/fnpr662-numba bash scripts/loop/smoke.sh
```

Entrega sin stage ni commit, en `loop/QuantAgent-fnpr.6.6.2`, para revisión e
integración por la sesión lanzadora. Sin push, PR ni uso de Beads.
